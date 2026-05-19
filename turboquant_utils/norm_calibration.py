"""Norm calibration: learn per-row scaling corrections using activation data.

After quantization, each TurboQuantLinear layer stores per-row norms that
were computed analytically (L2 norm of the original weight rows).  Quantization
introduces a systematic bias because the codebook is designed for N(0,1) but
real rotated weights are only approximately Gaussian.

This module optimises a per-output-row scaling factor ``alpha`` (parameterised
as ``exp(log_alpha)`` to stay positive) that minimises a combination of:

    MSE + lambda * (angular_loss + KLD_loss)

between the reference (fp) layer output and the quantized layer output on
calibration data.  After optimisation ``alpha`` is folded into the stored
norms so inference cost is unchanged.

Usage:
    from turboquant_model.norm_calibration import calibrate_norms

    calibrate_norms(tq_model, fp_model, tokenizer, device="cuda")
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from turboquant_model.module import TurboQuantLinear

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(y_ref: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE, averaged over the batch."""
    return (y_ref - y_hat).pow(2).mean()


def negative_log_cosine_loss(y_ref: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Angular distortion: -log(mean cosine similarity).

    Penalises directional deviation independently of magnitude.
    """
    cos = F.cosine_similarity(y_ref, y_hat, dim=-1)
    return -torch.log(cos.clamp(min=1e-8).mean())


def layer_kld_loss(y_ref: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Forward KL divergence on softmax distributions over the output dim.

    KL(ref || hat) — treats each token's output vector as a distribution over
    features, a proxy for how the downstream softmax (logits) would shift.
    Uses forward-only KL (not symmetric) for numerical stability.
    """
    log_p = F.log_softmax(y_hat.float(), dim=-1)
    q = F.softmax(y_ref.float(), dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean", log_target=False)


def combined_loss(
    y_ref: torch.Tensor,
    y_hat: torch.Tensor,
    lam: float = 1.0,
) -> torch.Tensor:
    """MSE + lambda * (angular + KLD) combined loss."""
    mse = mse_loss(y_ref, y_hat)
    angular = negative_log_cosine_loss(y_ref, y_hat)
    kld = layer_kld_loss(y_ref, y_hat)
    return mse + lam * (angular + kld)


# ---------------------------------------------------------------------------
# Activation collection via forward hooks
# ---------------------------------------------------------------------------

class _ActivationCollector:
    """Forward hook that records (input, output) pairs for a layer."""

    def __init__(self, max_rows: int = 8192):
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []
        self.max_rows = max_rows
        self._total = 0

    def __call__(self, module, args, output):
        if self._total >= self.max_rows:
            return
        x = args[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        y = output.detach()
        if y.dim() == 3:
            y = y.reshape(-1, y.shape[-1])
        self.inputs.append(x.cpu())
        self.outputs.append(y.cpu())
        self._total += x.shape[0]

    def get(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        X = torch.cat(self.inputs, dim=0)[: self.max_rows]
        Y = torch.cat(self.outputs, dim=0)[: self.max_rows]
        return X.to(device), Y.to(device)


# ---------------------------------------------------------------------------
# Per-layer calibration
# ---------------------------------------------------------------------------

def _calibrate_single_layer(
    tq_layer: TurboQuantLinear,
    X: torch.Tensor,
    Y_ref: torch.Tensor,
    lam: float = 1.0,
    lr: float = 1e-3,
    n_iters: int = 200,
    batch_size: int = 64,
) -> dict:
    """Optimise per-row alpha for a single TurboQuantLinear.

    Returns dict with before/after MSE and cosine similarity.
    """
    device = tq_layer.indices_packed.device
    X = X.to(device).float()
    Y_ref = Y_ref.to(device).float()

    M = tq_layer.out_features
    log_alpha = torch.zeros(M, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW([log_alpha], lr=lr, weight_decay=0.0)

    # Before metrics
    with torch.no_grad():
        Y_hat0 = tq_layer(X[:batch_size])
        before_mse = mse_loss(Y_ref[:batch_size], Y_hat0).item()
        before_cos = F.cosine_similarity(Y_ref[:batch_size], Y_hat0, dim=-1).mean().item()

    n = X.shape[0]
    for _ in range(n_iters):
        idx = torch.randint(0, n, (min(batch_size, n),))
        x_b = X[idx]
        y_b = Y_ref[idx]

        with torch.no_grad():
            y_raw = tq_layer(x_b)

        alpha = log_alpha.exp()
        y_hat = y_raw * alpha[None, :]

        loss = combined_loss(y_b, y_hat, lam=lam)

        if torch.isnan(loss) or torch.isinf(loss):
            continue  # skip bad batches

        optimizer.zero_grad()
        loss.backward()
        # Clamp gradients to prevent divergence
        torch.nn.utils.clip_grad_norm_([log_alpha], max_norm=1.0)
        optimizer.step()

        # If optimizer produced NaN, reset to zero (no calibration)
        with torch.no_grad():
            nan_mask = log_alpha.isnan()
            if nan_mask.any():
                log_alpha[nan_mask] = 0.0
                # Also reset optimizer state for NaN entries
                state = optimizer.state[log_alpha]
                if 'exp_avg' in state:
                    state['exp_avg'][nan_mask] = 0.0
                    state['exp_avg_sq'][nan_mask] = 0.0
            log_alpha.clamp_(-2.0, 2.0)

    # Fold alpha into norms
    alpha_final = log_alpha.detach().exp()
    # Final safety: replace any remaining NaN/inf with 1.0 (neutral)
    bad = alpha_final.isnan() | alpha_final.isinf()
    if bad.any():
        alpha_final[bad] = 1.0
        logger.warning(f"Reset {bad.sum().item()} NaN/inf alpha values")
    if tq_layer.weight_norms.dim() == 1:
        tq_layer.weight_norms.data *= alpha_final
    else:
        tq_layer.weight_norms.data *= alpha_final.unsqueeze(1)

    if tq_layer.has_residual:
        p2_norms = getattr(tq_layer, "pass2_weight_norms", None)
        if p2_norms is not None:
            if p2_norms.dim() == 1:
                p2_norms.data *= alpha_final
            else:
                p2_norms.data *= alpha_final.unsqueeze(1)

    # After metrics
    with torch.no_grad():
        Y_hat1 = tq_layer(X[:batch_size])
        after_mse = mse_loss(Y_ref[:batch_size], Y_hat1).item()
        after_cos = F.cosine_similarity(Y_ref[:batch_size], Y_hat1, dim=-1).mean().item()

    return {
        "before_mse": before_mse,
        "before_cos": before_cos,
        "after_mse": after_mse,
        "after_cos": after_cos,
        "alpha_delta_l2": (alpha_final - 1.0).norm().item(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    """Parameters for norm calibration."""
    n_samples: int = 4
    seq_length: int = 2048
    lam: float = 1.0
    lr: float = 1e-3
    n_iters: int = 50
    batch_size: int = 64
    per_group: bool = True  # per-group alpha (M,G) vs per-row alpha (M,)


def collect_calibration_data(
    fp_model: nn.Module,
    tq_model: nn.Module,
    tokenizer,
    device: str,
    n_samples: int = 128,
    seq_length: int = 2048,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Run calibration text through the FP model and collect per-layer (X, Y) pairs.

    Only layers that have a matching TurboQuantLinear in `tq_model` are recorded.

    Returns:
        dict mapping layer name -> (X: [N, K], Y_ref: [N, M])
    """
    from datasets import load_dataset

    fp_linears = {n: m for n, m in fp_model.named_modules() if isinstance(m, nn.Linear)}
    tq_linears = {n: m for n, m in tq_model.named_modules() if isinstance(m, TurboQuantLinear)}
    matched = {n for n in tq_linears if n in fp_linears}

    collectors: dict[str, _ActivationCollector] = {}
    handles = []
    for name in matched:
        coll = _ActivationCollector(max_rows=n_samples * 4)
        collectors[name] = coll
        handles.append(fp_linears[name].register_forward_hook(coll))

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]

    n_collected = 0
    for i in range(0, len(input_ids) - seq_length, seq_length):
        if n_collected >= n_samples:
            break
        ids = input_ids[i : i + seq_length].unsqueeze(0).to(device)
        with torch.no_grad():
            fp_model(ids)
        n_collected += 1

    for h in handles:
        h.remove()

    data = {}
    for name, coll in collectors.items():
        if coll._total > 0:
            X, Y = coll.get(device="cpu")
            data[name] = (X, Y)

    logger.info(f"Collected calibration data for {len(data)} layers "
                f"({n_collected} sequences)")
    return data


def calibrate_norms(
    tq_model: nn.Module,
    fp_model: nn.Module,
    tokenizer,
    device: str = "cuda",
    config: CalibrationConfig | None = None,
) -> list[dict]:
    """Calibrate all TurboQuantLinear norms in `tq_model` (per-layer).

    1. Collects activations from `fp_model` on WikiText-103 validation data.
    2. For each matched layer, optimises per-row scaling alpha to minimise
       MSE + lambda * (angular + KLD) between fp output and TQ output.
    3. Folds alpha into the stored norms (zero runtime cost).

    Fused kernels are temporarily disabled during calibration (gradient flow
    requires the pure-PyTorch forward path) and restored afterwards.

    Args:
        tq_model: quantized model (modified in-place)
        fp_model: reference floating-point model
        tokenizer: HuggingFace tokenizer
        device: compute device
        config: calibration hyperparameters

    Returns:
        list of per-layer stats dicts
    """
    if config is None:
        config = CalibrationConfig()

    # Temporarily disable fused kernels (need pure-PyTorch forward for gradient)
    saved_flags: list[tuple[TurboQuantLinear, bool, bool, bool]] = []
    for m in tq_model.modules():
        if isinstance(m, TurboQuantLinear):
            saved_flags.append((m, m.use_cutile, m.use_triton, m.use_metal))
            m.use_cutile = False
            m.use_triton = False
            m.use_metal = False

    t0 = time.time()
    calib_data = collect_calibration_data(
        fp_model, tq_model, tokenizer, device,
        n_samples=config.n_samples,
        seq_length=config.seq_length,
    )

    results = []
    tq_linears = {n: m for n, m in tq_model.named_modules() if isinstance(m, TurboQuantLinear)}

    for name in tq_linears:
        if name not in calib_data:
            continue
        tq_layer = tq_linears[name]
        X, Y_ref = calib_data[name]

        logger.info(f"Calibrating {name} ({X.shape[0]} samples) ...")
        stats = _calibrate_single_layer(
            tq_layer, X, Y_ref,
            lam=config.lam,
            lr=config.lr,
            n_iters=config.n_iters,
            batch_size=config.batch_size,
        )
        stats["layer"] = name
        results.append(stats)

        logger.info(
            f"  MSE {stats['before_mse']:.6f} -> {stats['after_mse']:.6f}, "
            f"cos {stats['before_cos']:.6f} -> {stats['after_cos']:.6f}"
        )

    elapsed = time.time() - t0
    logger.info(f"Norm calibration complete: {len(results)} layers in {elapsed:.1f}s")

    # Restore fused kernel flags
    for m, cutile, triton, metal in saved_flags:
        m.use_cutile = cutile
        m.use_triton = triton
        m.use_metal = metal

    return results


# ---------------------------------------------------------------------------
# Block-wise end-to-end calibration
# ---------------------------------------------------------------------------

def _prepare_calibration_tokens(
    tokenizer,
    n_samples: int,
    seq_length: int,
    device: str,
) -> list[torch.Tensor]:
    """Prepare calibration token batches from WikiText-103 validation."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]

    batches = []
    for i in range(0, len(input_ids) - seq_length, seq_length):
        if len(batches) >= n_samples:
            break
        batches.append(input_ids[i : i + seq_length].unsqueeze(0).to(device))
    return batches


def _find_decoder_blocks(model: nn.Module):
    """Find the list of transformer decoder blocks in a HF model.

    Looks for common attribute paths: model.layers, model.model.layers, etc.
    """
    for attr_path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = model
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            return list(obj)
    raise ValueError("Cannot find decoder blocks in model architecture")


def _fold_alpha(layer: TurboQuantLinear, alpha: torch.Tensor) -> None:
    """Fold alpha scaling into stored norms.

    alpha can be (M,) for per-row or (M, G) for per-group.
    """
    bad = alpha.isnan() | alpha.isinf()
    if bad.any():
        alpha[bad] = 1.0
        logger.warning(f"Reset {bad.sum().item()} NaN/inf alpha values")
    if alpha.dim() == layer.weight_norms.dim():
        # Per-group alpha matches norms shape exactly
        layer.weight_norms.data *= alpha
    elif layer.weight_norms.dim() == 1:
        layer.weight_norms.data *= alpha
    else:
        # Per-row alpha (M,) with per-group norms (M, G)
        layer.weight_norms.data *= alpha.unsqueeze(1)
    if layer.has_residual:
        p2 = getattr(layer, "pass2_weight_norms", None)
        if p2 is not None:
            if alpha.dim() == p2.dim():
                p2.data *= alpha
            elif p2.dim() == 1:
                p2.data *= alpha if alpha.dim() == 1 else alpha.mean(dim=1)
            else:
                p2.data *= alpha.unsqueeze(1) if alpha.dim() == 1 else alpha


def calibrate_norms_blockwise(
    tq_model: nn.Module,
    fp_model: nn.Module,
    tokenizer,
    device: str = "cuda",
    config: CalibrationConfig | None = None,
) -> list[dict]:
    """Block-wise end-to-end norm calibration.

    Instead of optimizing each layer independently, this processes transformer
    blocks sequentially.  For each block all per-row alpha parameters are
    optimized jointly so that the block output (after RMSNorm, attention, MLP,
    etc.) matches the FP reference block output.  After each block the
    calibrated TQ model is re-run to obtain updated inputs for the next block,
    capturing error propagation effects.

    Optimised to run the FP model only once (block outputs are static) and to
    merge the after-metrics forward with the next-block capture forward,
    reducing full model forwards from ~3N to N+2 (N = number of blocks).

    Args:
        tq_model: quantized model (modified in-place)
        fp_model: reference floating-point model
        tokenizer: HuggingFace tokenizer
        device: compute device
        config: calibration hyperparameters

    Returns:
        list of per-block stats dicts
    """
    if config is None:
        config = CalibrationConfig()

    # NOTE: We do NOT disable fused kernels globally here because the full
    # model forward (used for hook capture / after-metrics) would OOM on the
    # lm_head.  Instead we toggle fused kernels OFF only for the TQ linears
    # inside the block being optimised, and restore them afterwards.
    #
    # requires_grad_(False) is deferred until after the initial captures
    # to avoid any interaction with triton kernel autotuning.

    t0 = time.time()

    blocks_tq = _find_decoder_blocks(tq_model)
    blocks_fp = _find_decoder_blocks(fp_model)
    assert len(blocks_tq) == len(blocks_fp), "Block count mismatch"
    n_blocks = len(blocks_tq)

    # Prepare calibration tokens
    token_batches = _prepare_calibration_tokens(
        tokenizer, config.n_samples, config.seq_length, device,
    )
    # Use a subset for block calibration (keep memory manageable)
    n_cal = min(config.batch_size, len(token_batches))
    cal_ids = torch.cat(token_batches[:n_cal], dim=0)  # [n_cal, seq_length]

    # ------------------------------------------------------------------
    # Pre-capture ALL FP block outputs in a single forward pass.
    # The FP model never changes, so one pass suffices for all blocks.
    # ------------------------------------------------------------------
    # Use backbone-only forward (skip lm_head) to avoid allocating the
    # full [batch, seq, vocab] logits tensor which can exceed GPU memory.
    fp_backbone = fp_model.model if hasattr(fp_model, "model") else fp_model
    tq_backbone = tq_model.model if hasattr(tq_model, "model") else tq_model

    fp_block_outputs: list[torch.Tensor | None] = [None] * n_blocks
    fp_hooks = []
    for i, blk_fp in enumerate(blocks_fp):
        def _make_fp_hook(idx: int):
            def _hook(module, args, kwargs, output):
                out0 = output[0] if isinstance(output, tuple) else output
                fp_block_outputs[idx] = out0.detach().cpu()
            return _hook
        fp_hooks.append(blk_fp.register_forward_hook(_make_fp_hook(i), with_kwargs=True))

    with torch.no_grad():
        fp_backbone(input_ids=cal_ids)
    for h in fp_hooks:
        h.remove()
    del fp_hooks

    # Move FP model to CPU to free ~1.5GB GPU memory for optimization.
    # It's not needed until lm_head calibration at the end (if ever).
    fp_model.cpu()
    torch.cuda.empty_cache()
    print(f"    Captured {n_blocks} FP block outputs ({time.time() - t0:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # Initial TQ forward to capture block-0 inputs / output.
    # ------------------------------------------------------------------
    cur: dict = {}

    def _capture_hook(module, args, kwargs, output):
        cur["args"] = tuple(a.detach() for a in args if isinstance(a, torch.Tensor))
        cur["kwargs"] = {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        out0 = output[0] if isinstance(output, tuple) else output
        cur["output"] = out0.detach()

    h0 = blocks_tq[0].register_forward_hook(_capture_hook, with_kwargs=True)
    with torch.no_grad():
        tq_backbone(input_ids=cal_ids)
    h0.remove()
    torch.cuda.empty_cache()

    # Now disable requires_grad on all model params (only alpha gets grad)
    for p in tq_model.parameters():
        p.requires_grad_(False)

    results = []

    for block_idx in range(n_blocks):
        bt0 = time.time()
        block_tq = blocks_tq[block_idx]
        target_output = fp_block_outputs[block_idx].to(device)

        block_input_args = cur["args"]
        block_input_kwargs = cur["kwargs"]

        # Before metrics (from the capture we already have)
        before_mse = mse_loss(target_output, cur["output"]).item()
        before_cos = F.cosine_similarity(
            target_output.reshape(-1, target_output.shape[-1]),
            cur["output"].reshape(-1, cur["output"].shape[-1]),
            dim=-1,
        ).mean().item()

        # ---- Create alpha params for TQ linears in this block ----
        tq_linears = {
            n: m for n, m in block_tq.named_modules()
            if isinstance(m, TurboQuantLinear)
        }
        if not tq_linears:
            logger.info(f"Block {block_idx}: no TQ layers, skipping")
            # Still need to advance `cur` for the next block
            if block_idx < n_blocks - 1:
                nxt: dict = {}
                def _nxt_hook(module, args, kwargs, output):
                    nxt["args"] = tuple(a.detach() for a in args if isinstance(a, torch.Tensor))
                    nxt["kwargs"] = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
                    out0 = output[0] if isinstance(output, tuple) else output
                    nxt["output"] = out0.detach()
                h_n = blocks_tq[block_idx + 1].register_forward_hook(_nxt_hook, with_kwargs=True)
                with torch.no_grad():
                    tq_backbone(input_ids=cal_ids)
                h_n.remove()
                cur = nxt
            continue

        log_alphas: dict[str, torch.Tensor] = {}
        orig_norms: dict[str, torch.Tensor] = {}
        for name, layer in tq_linears.items():
            if config.per_group and layer.weight_norms.dim() == 2:
                # Per-group alpha: (M, G)
                log_alphas[name] = torch.zeros_like(
                    layer.weight_norms, dtype=torch.float32, requires_grad=True,
                )
                orig_norms[name] = layer.weight_norms.data.clone()
            else:
                # Per-row alpha: (M,)
                log_alphas[name] = torch.zeros(
                    layer.out_features, device=device,
                    dtype=torch.float32, requires_grad=True,
                )

        # ---- Patch TQ linear forwards with alpha scaling ----
        # Disable fused kernels in this block for gradient flow
        block_saved_flags: list[tuple[TurboQuantLinear, bool, bool, bool]] = []
        for layer in tq_linears.values():
            block_saved_flags.append((layer, layer.use_cutile, layer.use_triton, layer.use_metal))
            layer.use_cutile = layer.use_triton = layer.use_metal = False

        saved_forwards: dict[str, object] = {}
        for name, layer in tq_linears.items():
            saved_forwards[name] = layer.forward

            if name in orig_norms:
                # Per-group: inject alpha into weight_norms before forward
                def _make_patched_pg(orig_fn, alpha_param, lyr, on):
                    def _patched(x):
                        # Replace norms with alpha-scaled version (differentiable)
                        lyr.weight_norms = on * alpha_param.exp()
                        y = orig_fn(x)
                        return y.to(x.dtype)
                    return _patched

                layer.forward = _make_patched_pg(
                    layer.forward, log_alphas[name], layer, orig_norms[name],
                )
            else:
                # Per-row: scale output (original approach)
                def _make_patched(orig_fn, alpha_param):
                    def _patched(x):
                        y = orig_fn(x)
                        return (y * alpha_param.exp()[None, :]).to(x.dtype)
                    return _patched

                layer.forward = _make_patched(layer.forward, log_alphas[name])

        # ---- Optimize ----
        # Use a small subset of samples for the gradient loop to limit
        # GPU memory (full block forward+backward with grad is expensive).
        opt_n = min(4, block_input_args[0].shape[0])
        batch_dim = block_input_args[0].shape[0]
        opt_args = tuple(a[:opt_n] for a in block_input_args)

        def _slice_val(v):
            """Recursively slice tensors/tuples to opt_n along batch dim."""
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == batch_dim:
                return v[:opt_n]
            if isinstance(v, (tuple, list)):
                sliced = [_slice_val(item) for item in v]
                return type(v)(sliced)
            return v

        opt_kwargs = {}
        for k, v in block_input_kwargs.items():
            # past_key_values is a stateful KV cache sized for the full batch;
            # it can't be sliced, so drop it for the sub-batch gradient loop.
            if k == "past_key_values":
                opt_kwargs[k] = None
            else:
                opt_kwargs[k] = _slice_val(v)
        opt_target = target_output[:opt_n]

        optimizer = torch.optim.AdamW(
            list(log_alphas.values()), lr=config.lr, weight_decay=0.0,
        )

        for _it in range(config.n_iters):
            output = block_tq(*opt_args, **opt_kwargs)
            pred = output[0] if isinstance(output, tuple) else output
            loss = combined_loss(opt_target, pred, lam=config.lam)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            for la in log_alphas.values():
                torch.nn.utils.clip_grad_norm_([la], max_norm=1.0)
            optimizer.step()

            if _it == 0 or (_it + 1) % 50 == 0:
                print(f"      iter {_it + 1}/{config.n_iters} loss={loss.item():.6f}", flush=True)

            with torch.no_grad():
                for la in log_alphas.values():
                    nan_mask = la.isnan()
                    if nan_mask.any():
                        la[nan_mask] = 0.0
                        state = optimizer.state.get(la)
                        if state and "exp_avg" in state:
                            state["exp_avg"][nan_mask] = 0.0
                            state["exp_avg_sq"][nan_mask] = 0.0
                    la.clamp_(-2.0, 2.0)

        # ---- Restore forwards, fold alphas, restore fused kernels ----
        for name, layer in tq_linears.items():
            layer.forward = saved_forwards[name]
            alpha = log_alphas[name].detach().exp()
            # Restore original norms buffer if per-group patching replaced it
            if name in orig_norms:
                layer.weight_norms = orig_norms[name]
            _fold_alpha(layer, alpha)
        for layer, cutile, triton, metal in block_saved_flags:
            layer.use_cutile = cutile
            layer.use_triton = triton
            layer.use_metal = metal

        # ----------------------------------------------------------
        # After-metrics + next-block capture in ONE TQ forward pass.
        # This halves the number of full model forwards needed.
        # ----------------------------------------------------------
        after_cap: dict = {}
        next_cap: dict = {}

        def _after_hook(module, args, kwargs, output):
            out0 = output[0] if isinstance(output, tuple) else output
            after_cap["output"] = out0.detach()

        h_after = block_tq.register_forward_hook(_after_hook, with_kwargs=True)

        h_next = None
        if block_idx < n_blocks - 1:
            def _make_next_hook(cap_dict):
                def _hook(module, args, kwargs, output):
                    cap_dict["args"] = tuple(
                        a.detach() for a in args if isinstance(a, torch.Tensor)
                    )
                    cap_dict["kwargs"] = {
                        k: v.detach() if isinstance(v, torch.Tensor) else v
                        for k, v in kwargs.items()
                    }
                    out0 = output[0] if isinstance(output, tuple) else output
                    cap_dict["output"] = out0.detach()
                return _hook
            h_next = blocks_tq[block_idx + 1].register_forward_hook(
                _make_next_hook(next_cap), with_kwargs=True,
            )

        with torch.no_grad():
            tq_backbone(input_ids=cal_ids)

        h_after.remove()
        if h_next is not None:
            h_next.remove()
            cur = next_cap  # advance to next block

        after_mse = mse_loss(target_output, after_cap["output"]).item()
        after_cos = F.cosine_similarity(
            target_output.reshape(-1, target_output.shape[-1]),
            after_cap["output"].reshape(-1, after_cap["output"].shape[-1]),
            dim=-1,
        ).mean().item()

        stats = {
            "block": block_idx,
            "n_layers": len(tq_linears),
            "before_mse": before_mse,
            "before_cos": before_cos,
            "after_mse": after_mse,
            "after_cos": after_cos,
        }
        results.append(stats)
        bt_elapsed = time.time() - bt0
        print(
            f"    Block {block_idx + 1}/{n_blocks}: "
            f"MSE {before_mse:.6f}->{after_mse:.6f}  "
            f"cos {before_cos:.6f}->{after_cos:.6f}  "
            f"({bt_elapsed:.1f}s)",
            flush=True,
        )

    # Also calibrate lm_head if it's a TQ layer
    lm_head = getattr(tq_model, "lm_head", None)
    if isinstance(lm_head, TurboQuantLinear):
        logger.info("Calibrating lm_head via per-layer method ...")
        fp_model.to(device)  # bring FP model back for lm_head calibration
        calib_data = collect_calibration_data(
            fp_model, tq_model, tokenizer, device,
            n_samples=config.n_samples,
            seq_length=config.seq_length,
        )
        head_name = None
        for n, m in tq_model.named_modules():
            if m is lm_head:
                head_name = n
                break
        if head_name and head_name in calib_data:
            X, Y_ref = calib_data[head_name]
            stats = _calibrate_single_layer(
                lm_head, X, Y_ref,
                lam=config.lam, lr=config.lr,
                n_iters=config.n_iters, batch_size=config.batch_size,
            )
            stats["layer"] = head_name
            results.append(stats)

    elapsed = time.time() - t0
    logger.info(f"Block-wise calibration complete: {len(results)} blocks in {elapsed:.1f}s")

    return results
