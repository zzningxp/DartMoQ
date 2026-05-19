"""Model-level quantization, saving, and loading.

quantize_model:  Replace all nn.Linear → TurboQuantLinear (single-pass or residual)
save_quantized / load_quantized: Serialize/deserialize quantized models to disk
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from turboquant_model.codebook import get_codebook
from turboquant_model.rotation import generate_rotation_matrix
from turboquant_model.quantize import pack_4bit
from turboquant_model.module import TurboQuantLinear, SharedScratchPool, QuantizedEmbedding
from turboquant_model.rotation import (
    generate_rotation_matrix,
    hadamard_rotate,
    hadamard_rotate_inverse,
)
from turboquant_model.norm_compression import factorize_norms, reconstruct_norms

logger = logging.getLogger(__name__)


def _entropy_compress_indices(
    packed: torch.Tensor, bit_width: int, N: int,
) -> torch.Tensor:
    """Compress packed indices with rANS, returning a 1D uint8 tensor."""
    from turboquant_model.entropy_codec import compress_indices
    from turboquant_model.quantize import unpack_4bit

    indices = unpack_4bit(packed, N)
    compressed_bytes, _ = compress_indices(indices, bit_width)
    return torch.frombuffer(bytearray(compressed_bytes), dtype=torch.uint8).clone()


def _entropy_decompress_indices(
    compressed_tensor: torch.Tensor, bit_width: int, M: int, N: int,
) -> torch.Tensor:
    """Decompress rANS bytes back to packed 4-bit indices."""
    from turboquant_model.entropy_codec import decompress_indices
    from turboquant_model.quantize import pack_4bit

    data = bytes(compressed_tensor.cpu().numpy())
    indices = decompress_indices(data, bit_width, (M, N))
    return pack_4bit(indices)


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant weight quantization."""

    bit_width: int = 4
    group_size: Optional[int] = 128
    seed: int = 42
    skip_embeddings: bool = False
    skip_lm_head: bool = False
    # Residual
    residual_bit_width: Optional[int] = None
    residual_seed: int = 1042
    # Rotation method: "qr" (Haar random orthogonal) or "hadamard" (fast Walsh-Hadamard + signs)
    rotation: str = "qr"
    # Rotation strategy for residual passes:
    #   "different" — pass 1 uses seed, pass 2 uses residual_seed (default, best quality)
    #   "shared"    — both passes use the same seed (enables merge_and_requantize)
    #   "alternating" — even passes use seed, odd passes use residual_seed (for multi-pass)
    #   "per_layer"  — each layer gets a unique seed (seed + layer_idx * 10007) to decorrelate errors
    rotation_strategy: str = "different"

    # Advanced features
    norm_codec: str = "fp32"         # norm compression: "fp32", "fp16", "factored_int8"
    entropy_coding: bool = False     # rANS entropy coding of indices
    cpu_offload_pass2: bool = False  # offload residual pass2 to CPU (pipelined H2D)

    # Embedding quantization: "none", "int8" (per-row symmetric), "int4" (per-group Q4_K-style)
    embedding_quant: str = "none"
    embedding_group_size: int = 32   # group size for int4 mode

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TurboQuantConfig":
        with open(path) as f:
            return cls(**json.load(f))

    @property
    def total_bits(self) -> int:
        return self.bit_width + (self.residual_bit_width or 0)


# ---------------------------------------------------------------------------
# Quantize model
# ---------------------------------------------------------------------------


@torch.no_grad()
def quantize_model(model: nn.Module, config: TurboQuantConfig) -> nn.Module:
    """Quantize all nn.Linear layers, replacing them with TurboQuantLinear.

    Supports single-pass and residual (two-pass) quantization.
    All layers use on-the-fly dequantization at inference.

    Args:
        model: HuggingFace model (or any nn.Module with Linear layers)
        config: quantization configuration

    Returns:
        model with TurboQuantLinear modules (modified in-place)
    """
    centroids, boundaries = get_codebook(config.bit_width)
    if config.residual_bit_width:
        r_centroids, r_boundaries = get_codebook(config.residual_bit_width)

    replaced = 0
    total_orig = 0
    total_compressed = 0

    # Collect modules to replace
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if config.skip_embeddings and "embed" in name.lower():
            continue
        if config.skip_lm_head and "lm_head" in name.lower():
            continue
        replacements.append((name, module))

    for layer_idx, (name, module) in enumerate(replacements):
        W = module.weight.data
        M, N = W.shape
        device = W.device

        group_size = config.group_size or N

        # Per-layer rotation: each layer gets a unique base seed
        if config.rotation_strategy == "per_layer":
            layer_seed = config.seed + layer_idx * 10007
        else:
            layer_seed = config.seed

        # --- Pass 1: Quantize weight ---
        pass1_packed, pass1_norms, pass1_codebook = _quantize_weight(
            W, config.bit_width, group_size, layer_seed, centroids, boundaries, device,
            rotation=config.rotation,
        )

        # --- Create TurboQuantLinear ---
        tq = TurboQuantLinear(
            in_features=N,
            out_features=M,
            bias=module.bias is not None,
            bit_width=config.bit_width,
            group_size=group_size,
            device=device,
            rotation=config.rotation,
        )
        tq.indices_packed.copy_(pass1_packed)
        tq.weight_norms.copy_(pass1_norms)
        tq.codebook.copy_(centroids.to(device))
        tq.set_rotation(layer_seed)

        if module.bias is not None:
            tq.bias.copy_(module.bias.data)

        # --- Pass 2: Residual quantization ---
        if config.residual_bit_width:
            # Reconstruct pass 1 to compute residual
            W_hat1 = tq.dequantize().float()
            residual = W.float() - W_hat1

            # Determine residual rotation seed based on strategy
            if config.rotation_strategy == "shared":
                pass2_seed = layer_seed
            elif config.rotation_strategy == "per_layer":
                pass2_seed = config.residual_seed + layer_idx * 10007
            else:  # "different" or "alternating" — both use residual_seed for pass 2
                pass2_seed = config.residual_seed

            pass2_packed, pass2_norms, pass2_codebook = _quantize_weight(
                residual, config.residual_bit_width, group_size,
                pass2_seed, r_centroids, r_boundaries, device,
                rotation=config.rotation,
            )
            tq.set_pass2(
                indices_packed=pass2_packed,
                weight_norms=pass2_norms,
                codebook=r_centroids.to(device),
                seed=pass2_seed,
            )

            # Offload pass2 to CPU if requested
            if config.cpu_offload_pass2:
                tq.offload_pass2_to_cpu()

        # Replace in model
        _replace_module(model, name, tq)

        orig_bytes = M * N * 2  # bf16
        total_orig += orig_bytes
        total_compressed += tq.memory_bytes()
        replaced += 1

    mode = "residual" if config.residual_bit_width else "single-pass"
    bits = f"{config.bit_width}" if not config.residual_bit_width else f"{config.bit_width}+{config.residual_bit_width}"
    compression_ratio = (
        f"{total_orig / total_compressed:.1f}x" if total_compressed > 0 else "N/A"
    )
    logger.info(
        f"Quantized {replaced} layers ({mode}, {bits}-bit): "
        f"{total_orig / 1024**2:.1f}MB → {total_compressed / 1024**2:.1f}MB "
        f"({compression_ratio} compression)"
    )

    # Quantize embeddings if requested
    if config.embedding_quant != "none":
        emb_replaced = 0
        emb_orig = 0
        emb_compressed = 0
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Embedding):
                continue
            V, D = module.weight.shape
            qe = QuantizedEmbedding.from_float(
                module, mode=config.embedding_quant, group_size=config.embedding_group_size,
            )
            _replace_module(model, name, qe)
            emb_orig += V * D * 2  # bf16
            emb_compressed += qe.memory_bytes()
            emb_replaced += 1
        if emb_replaced > 0:
            ratio = f"{emb_orig / emb_compressed:.1f}x" if emb_compressed > 0 else "N/A"
            logger.info(
                f"Quantized {emb_replaced} embedding(s) ({config.embedding_quant}): "
                f"{emb_orig / 1024**2:.1f}MB → {emb_compressed / 1024**2:.1f}MB "
                f"({ratio} compression)"
            )

    # Clear caches populated during quantization (dequantize() for residual
    # computation caches unpacked int32 indices that are 8× the packed size).
    for module in model.modules():
        if isinstance(module, TurboQuantLinear):
            module._cached_indices = None
            module._cached_pass2_indices = None
            module._rotation_cache.clear()

    # Set up next-layer prefetch chain if offloading
    if config.cpu_offload_pass2:
        enable_prefetch_chain(model)

    return model


@torch.no_grad()
def quantize_model_advanced(model: nn.Module, config: TurboQuantConfig) -> nn.Module:
    """Quantize with norm factorization and entropy coding support.

    Supports all features from quantize_model plus:
    - norm_codec: compress norms via factorization ("fp16", "factored_int8")
    - entropy_coding: flag only (actual compression measured separately)
    - Non-4-bit quantization via per-group variable bit-width path

    Args:
        model: HuggingFace model
        config: quantization config with advanced options

    Returns:
        model with TurboQuantLinear modules (modified in-place)
    """
    centroids_cache = {}
    boundaries_cache = {}

    def _get_codebook_cached(bw):
        if bw not in centroids_cache:
            c, b = get_codebook(bw)
            centroids_cache[bw] = c
            boundaries_cache[bw] = b
        return centroids_cache[bw], boundaries_cache[bw]

    replaced = 0
    total_orig = 0
    total_compressed = 0

    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if config.skip_embeddings and "embed" in name.lower():
            continue
        if config.skip_lm_head and "lm_head" in name.lower():
            continue
        replacements.append((name, module))

    for layer_idx, (name, module) in enumerate(replacements):
        W = module.weight.data
        M, N = W.shape
        device = W.device
        group_size = config.group_size or N
        n_groups = math.ceil(N / group_size)

        # Per-layer rotation: each layer gets a unique base seed
        if config.rotation_strategy == "per_layer":
            layer_seed = config.seed + layer_idx * 10007
        else:
            layer_seed = config.seed

        group_bit_widths = [config.bit_width] * n_groups

        # --- Quantize with per-group bit-widths ---
        packed, norms, group_codebooks, indices_uint8 = _quantize_weight_variable(
            W, group_bit_widths, group_size, layer_seed, device,
            rotation=config.rotation, codebook_cache=(_get_codebook_cached,),
        )

        # --- Create TurboQuantLinear ---
        tq = TurboQuantLinear(
            in_features=N,
            out_features=M,
            bias=module.bias is not None,
            bit_width=config.bit_width,
            group_size=group_size,
            device=device,
            rotation=config.rotation,
        )

        # Use uint8 path for non-4-bit
        if config.bit_width != 4:
            tq.set_variable_bit_widths(
                group_bit_widths=group_bit_widths,
                group_codebooks=group_codebooks,
                indices_uint8=indices_uint8,
            )
        else:
            tq.indices_packed.copy_(packed)
            tq.codebook.copy_(centroids_cache[config.bit_width].to(device))

        tq.weight_norms.copy_(norms)
        tq.set_rotation(layer_seed)

        if module.bias is not None:
            tq.bias.copy_(module.bias.data)

        # --- Apply norm compression ---
        if config.norm_codec != "fp32" and norms.dim() == 2:
            tq.apply_norm_codec(config.norm_codec)

        _replace_module(model, name, tq)

        orig_bytes = M * N * 2  # bf16
        total_orig += orig_bytes
        total_compressed += tq.memory_bytes()
        replaced += 1

    nf_str = f"+{config.norm_codec}" if config.norm_codec != "fp32" else ""
    ec_str = "+EC" if config.entropy_coding else ""
    logger.info(
        f"Quantized {replaced} layers ({config.bit_width}-bit{nf_str}{ec_str}): "
        f"{total_orig / 1024**2:.1f}MB → {total_compressed / 1024**2:.1f}MB "
        f"({total_orig / total_compressed:.1f}x compression)"
    )

    return model


def _quantize_weight(
    W: torch.Tensor,
    bit_width: int,
    group_size: int,
    seed: int,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
    device: torch.device,
    rotation: str = "qr",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a single weight matrix and return packed data.

    Returns: (indices_packed, norms, codebook)
    """
    M, N = W.shape
    W = W.float()

    all_norms = []
    all_indices = []

    bnd = boundaries.to(device)
    ctr = centroids.to(device)

    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms
        all_norms.append(norms.squeeze(1))

        if rotation == "hadamard":
            Y = hadamard_rotate(W_norm, seed=seed + g_start)
        else:
            Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(device)
            Y = W_norm @ Pi.T
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        indices = torch.searchsorted(bnd, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(ctr) - 1).reshape(M, g_dim)
        all_indices.append(indices)

    full_indices = torch.cat(all_indices, dim=1)
    norms_out = torch.stack(all_norms, dim=1) if len(all_norms) > 1 else all_norms[0]

    if N % 2 != 0:
        full_indices = torch.nn.functional.pad(full_indices, (0, 1), value=0)

    packed = pack_4bit(full_indices)
    return packed, norms_out, ctr


def _quantize_weight_variable(
    W: torch.Tensor,
    group_bit_widths: list[int],
    group_size: int,
    seed: int,
    device: torch.device,
    rotation: str = "qr",
    codebook_cache: tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Quantize with per-group variable bit-widths.

    Returns: (indices_packed_4bit, norms, group_codebooks, indices_uint8)
        - indices_packed_4bit: standard 4-bit packed (for uniform case)
        - norms: (M, G) float32
        - group_codebooks: list of (2^b_g,) tensors per group
        - indices_uint8: (M, N) uint8 (for variable bit-width case)
    """
    M, N = W.shape
    W = W.float()

    get_cb = codebook_cache[0] if codebook_cache else lambda bw: get_codebook(bw)

    all_norms = []
    all_indices = []
    group_codebooks = []

    for g_idx, g_start in enumerate(range(0, N, group_size)):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]
        bw = group_bit_widths[g_idx]

        centroids, boundaries = get_cb(bw)
        ctr = centroids.to(device)
        bnd = boundaries.to(device)
        group_codebooks.append(ctr)

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms
        all_norms.append(norms.squeeze(1))

        if rotation == "hadamard":
            Y = hadamard_rotate(W_norm, seed=seed + g_start)
        else:
            Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(device)
            Y = W_norm @ Pi.T
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        indices = torch.searchsorted(bnd, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(ctr) - 1).reshape(M, g_dim)
        all_indices.append(indices)

    full_indices = torch.cat(all_indices, dim=1)
    norms_out = torch.stack(all_norms, dim=1) if len(all_norms) > 1 else all_norms[0]

    # uint8 indices for variable bit-width
    indices_uint8 = full_indices.to(torch.uint8)

    # Also produce 4-bit packed (for uniform case)
    if N % 2 != 0:
        full_indices = torch.nn.functional.pad(full_indices, (0, 1), value=0)
    packed = pack_4bit(full_indices)

    return packed, norms_out, group_codebooks, indices_uint8


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


@torch.no_grad()
def save_quantized(model: nn.Module, config: TurboQuantConfig, save_dir: str | Path):
    """Save quantized model to disk in safetensors format.

    Directory structure:
        save_dir/
        ├── turboquant_config.json
        ├── model.safetensors          # all quantized layer tensors + codebook
        ├── non_quantized.safetensors  # non-linear params (embeddings, norms, etc.)
        └── config.json                # (optional) HuggingFace model config

    When ``config.entropy_coding`` is True, indices are rANS-compressed and
    stored as ``{layer}.indices_ec`` (1-D uint8) with shape metadata in
    ``{layer}.indices_ec_shape`` (int32 tensor [M, N]).
    """
    from safetensors.torch import save_file

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config.save(save_dir / "turboquant_config.json")

    # Save HF model config
    if hasattr(model, "config"):
        model.config.save_pretrained(save_dir)

    tensors = {}
    codebook_saved = False
    tq_param_prefixes = set()

    for name, module in model.named_modules():
        if isinstance(module, TurboQuantLinear):
            safe = name.replace(".", "_")

            if config.entropy_coding:
                ec = _entropy_compress_indices(
                    module.indices_packed.cpu(), config.bit_width, module.in_features,
                )
                tensors[f"{safe}.indices_ec"] = ec.contiguous()
                tensors[f"{safe}.indices_ec_shape"] = torch.tensor(
                    [module.out_features, module.in_features], dtype=torch.int32,
                )
            else:
                tensors[f"{safe}.indices"] = module.indices_packed.cpu().contiguous()

            # Save norms: factored (compact) or full
            if (
                config.norm_codec in ("factored_int8", "factored_int4")
                and hasattr(module, "_factored_norms")
                and module._factored_norms is not None
            ):
                fn = module._factored_norms
                tensors[f"{safe}.norms.row_scale"] = fn.row_scale.cpu().contiguous()
                tensors[f"{safe}.norms.group_scale"] = fn.group_scale.cpu().contiguous()
                tensors[f"{safe}.norms.residual"] = fn.residual_int8.cpu().contiguous()
                tensors[f"{safe}.norms.residual_scale"] = torch.tensor(
                    [fn.residual_scale], dtype=torch.float32,
                )
                tensors[f"{safe}.norms.residual_bits"] = torch.tensor(
                    [fn.residual_bits], dtype=torch.int32,
                )
            elif config.norm_codec == "fp16":
                tensors[f"{safe}.norms"] = module.weight_norms.cpu().half().contiguous()
            else:
                tensors[f"{safe}.norms"] = module.weight_norms.cpu().contiguous()

            if module.bias is not None:
                tensors[f"{safe}.bias"] = module.bias.cpu().contiguous()

            if module.has_residual:
                # Resolve pass2 tensors (may be CPU-offloaded or on GPU)
                if module.is_pass2_offloaded:
                    p2_indices = module._pass2_cpu_indices_packed
                    p2_norms = module._pass2_cpu_weight_norms
                    p2_codebook = module._pass2_cpu_codebook
                else:
                    p2_indices = module.pass2_indices_packed.cpu()
                    p2_norms = module.pass2_weight_norms.cpu()
                    p2_codebook = module.pass2_codebook.cpu()

                if config.entropy_coding:
                    ec2 = _entropy_compress_indices(
                        p2_indices.cpu(),
                        config.residual_bit_width or config.bit_width,
                        module.in_features,
                    )
                    tensors[f"{safe}.pass2_indices_ec"] = ec2.contiguous()
                    tensors[f"{safe}.pass2_indices_ec_shape"] = torch.tensor(
                        [module.out_features, module.in_features], dtype=torch.int32,
                    )
                else:
                    tensors[f"{safe}.pass2_indices"] = p2_indices.cpu().contiguous()
                tensors[f"{safe}.pass2_norms"] = p2_norms.cpu().contiguous()
                tensors[f"{safe}.pass2_codebook"] = p2_codebook.cpu().clone()

            if not codebook_saved:
                tensors["codebook"] = module.codebook.cpu().clone()
                codebook_saved = True

            tq_param_prefixes.add(name + ".")

    # Save quantized embeddings
    qe_prefixes = set()
    for name, module in model.named_modules():
        if isinstance(module, QuantizedEmbedding):
            safe = name.replace(".", "_")
            tensors[f"{safe}.qe_mode"] = torch.tensor(
                [ord(c) for c in module.mode], dtype=torch.uint8,
            )
            tensors[f"{safe}.qe_meta"] = torch.tensor(
                [module.num_embeddings, module.embedding_dim, module.group_size,
                 module.padding_idx if module.padding_idx is not None else -1],
                dtype=torch.int64,
            )
            if module.mode == "int8":
                tensors[f"{safe}.weight_int8"] = module.weight_int8.cpu().contiguous()
                tensors[f"{safe}.weight_scale"] = module.weight_scale.cpu().contiguous()
            elif module.mode == "int4":
                tensors[f"{safe}.weight_packed"] = module.weight_packed.cpu().contiguous()
                tensors[f"{safe}.weight_scale"] = module.weight_scale.cpu().contiguous()
                tensors[f"{safe}.weight_min"] = module.weight_min.cpu().contiguous()
            qe_prefixes.add(name + ".")

    save_file(tensors, save_dir / "model.safetensors")

    # Collect non-quantized parameters
    all_quant_prefixes = tq_param_prefixes | qe_prefixes
    non_quantized = {}
    for pname, param in model.named_parameters():
        is_quant = any(pname.startswith(prefix) for prefix in all_quant_prefixes)
        if not is_quant:
            non_quantized[pname] = param.data.cpu().contiguous()

    for bname, buf in model.named_buffers():
        is_quant = any(bname.startswith(prefix) for prefix in all_quant_prefixes)
        if not is_quant and bname not in non_quantized:
            non_quantized[bname] = buf.cpu().contiguous()

    save_file(non_quantized, save_dir / "non_quantized.safetensors")

    total = sum(f.stat().st_size for f in save_dir.rglob("*") if f.is_file())
    logger.info(f"Saved quantized model to {save_dir} ({total / 1024**2:.1f} MB)")


@torch.no_grad()
def load_quantized(
    model_name_or_path: str,
    quantized_dir: str | Path,
    device: str = "cuda",
    cpu_offload_pass2: bool | None = None,
) -> nn.Module:
    """Load a pre-quantized model from disk.

    Supports both safetensors format (model.safetensors) and legacy
    .pt format (layers/*.pt).

    Args:
        model_name_or_path: HF model name or path (for architecture)
        quantized_dir: directory with saved quantized weights
        device: target device
        cpu_offload_pass2: Override CPU offload for pass2 residual weights.
            ``None`` (default) uses the value from the saved config.
            ``True`` forces offload on (reduces GPU VRAM).
            ``False`` forces offload off (keeps everything on GPU).

    Returns:
        model with TurboQuantLinear modules, on-the-fly mode
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    quantized_dir = Path(quantized_dir)
    config = TurboQuantConfig.load(quantized_dir / "turboquant_config.json")

    # Override CPU offload setting if caller specified
    if cpu_offload_pass2 is not None:
        config.cpu_offload_pass2 = cpu_offload_pass2

    # Load architecture
    if (quantized_dir / "config.json").exists():
        model_config = AutoConfig.from_pretrained(quantized_dir)
    else:
        model_config = AutoConfig.from_pretrained(model_name_or_path)

    model = AutoModelForCausalLM.from_config(model_config).to(torch.bfloat16).to(device)

    # Detect format: safetensors vs legacy .pt
    safetensors_path = quantized_dir / "model.safetensors"
    use_safetensors = safetensors_path.exists()

    if use_safetensors:
        from safetensors.torch import load_file
        tensors = load_file(str(safetensors_path), device=device)
        codebook = tensors["codebook"]
    else:
        tensors = None
        codebook = torch.load(quantized_dir / "codebook.pt", map_location=device, weights_only=True)

    layers_dir = quantized_dir / "layers"

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if config.skip_embeddings and "embed" in name.lower():
            continue
        if config.skip_lm_head and "lm_head" in name.lower():
            continue

        safe = name.replace(".", "_")

        if use_safetensors:
            indices_key = f"{safe}.indices"
            indices_ec_key = f"{safe}.indices_ec"
            if indices_key not in tensors and indices_ec_key not in tensors:
                continue
        else:
            indices_path = layers_dir / f"{safe}.indices.pt"
            if not indices_path.exists():
                continue

        M, N = module.weight.shape
        group_size = config.group_size or N

        tq = TurboQuantLinear(
            in_features=N,
            out_features=M,
            bias=module.bias is not None,
            bit_width=config.bit_width,
            group_size=group_size,
            device=device,
            rotation=config.rotation,
        )

        if use_safetensors:
            ec_key = f"{safe}.indices_ec"
            if ec_key in tensors:
                shape_t = tensors[f"{safe}.indices_ec_shape"]
                ec_M, ec_N = int(shape_t[0]), int(shape_t[1])
                tq.indices_packed = _entropy_decompress_indices(
                    tensors[ec_key], config.bit_width, ec_M, ec_N,
                ).to(device)
            else:
                tq.indices_packed = tensors[f"{safe}.indices"]

            # Load norms: factored or full
            norms_row_key = f"{safe}.norms.row_scale"
            norms_full_key = f"{safe}.norms"
            if norms_row_key in tensors:
                from turboquant_model.norm_compression import FactoredNorms, reconstruct_norms
                res_bits_key = f"{safe}.norms.residual_bits"
                res_bits = int(tensors[res_bits_key][0]) if res_bits_key in tensors else 8
                fn = FactoredNorms(
                    row_scale=tensors[f"{safe}.norms.row_scale"],
                    group_scale=tensors[f"{safe}.norms.group_scale"],
                    residual_int8=tensors[f"{safe}.norms.residual"],
                    residual_scale=float(tensors[f"{safe}.norms.residual_scale"][0]),
                    residual_bits=res_bits,
                )
                tq.weight_norms = reconstruct_norms(fn).to(device)
                tq._factored_norms = fn
                tq._use_factored_norms = True
            elif norms_full_key in tensors:
                tq.weight_norms = tensors[norms_full_key].float().to(device)
        else:
            tq.indices_packed = torch.load(layers_dir / f"{safe}.indices.pt", map_location=device, weights_only=True)
            tq.weight_norms = torch.load(layers_dir / f"{safe}.norms.pt", map_location=device, weights_only=True)

        tq.codebook = codebook

        if module.bias is not None:
            if use_safetensors:
                bias_key = f"{safe}.bias"
                if bias_key in tensors:
                    tq.bias = tensors[bias_key]
            else:
                bias_path = layers_dir / f"{safe}.bias.pt"
                if bias_path.exists():
                    tq.bias = torch.load(bias_path, map_location=device, weights_only=True)

        tq.set_rotation(config.seed)

        # Load residual pass if present
        if use_safetensors:
            pass2_key = f"{safe}.pass2_indices"
            pass2_ec_key = f"{safe}.pass2_indices_ec"
            if pass2_ec_key in tensors:
                shape_t = tensors[f"{safe}.pass2_indices_ec_shape"]
                ec_M, ec_N = int(shape_t[0]), int(shape_t[1])
                p2_bw = config.residual_bit_width or config.bit_width
                pass2_packed = _entropy_decompress_indices(
                    tensors[pass2_ec_key], p2_bw, ec_M, ec_N,
                ).to(device)
                tq.set_pass2(
                    indices_packed=pass2_packed,
                    weight_norms=tensors[f"{safe}.pass2_norms"],
                    codebook=tensors[f"{safe}.pass2_codebook"],
                    seed=config.residual_seed,
                )
            elif pass2_key in tensors:
                tq.set_pass2(
                    indices_packed=tensors[pass2_key],
                    weight_norms=tensors[f"{safe}.pass2_norms"],
                    codebook=tensors[f"{safe}.pass2_codebook"],
                    seed=config.residual_seed,
                )
        else:
            pass2_path = layers_dir / f"{safe}.pass2_indices.pt"
            if pass2_path.exists():
                tq.set_pass2(
                    indices_packed=torch.load(pass2_path, map_location=device, weights_only=True),
                    weight_norms=torch.load(layers_dir / f"{safe}.pass2_norms.pt", map_location=device, weights_only=True),
                    codebook=torch.load(layers_dir / f"{safe}.pass2_codebook.pt", map_location=device, weights_only=True),
                    seed=config.residual_seed,
                )

        # Offload pass2 to CPU if requested
        if config.cpu_offload_pass2 and tq.has_residual:
            tq.offload_pass2_to_cpu()

        _replace_module(model, name, tq)

    # Restore quantized embeddings
    if use_safetensors:
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Embedding):
                continue
            safe = name.replace(".", "_")
            mode_key = f"{safe}.qe_mode"
            if mode_key not in tensors:
                continue
            mode = "".join(chr(c) for c in tensors[mode_key].tolist())
            meta = tensors[f"{safe}.qe_meta"]
            V, D, gs = int(meta[0]), int(meta[1]), int(meta[2])
            pad_idx = int(meta[3])
            qe = QuantizedEmbedding(
                num_embeddings=V, embedding_dim=D, mode=mode,
                group_size=gs,
                padding_idx=pad_idx if pad_idx >= 0 else None,
                device=device,
            )
            if mode == "int8":
                qe.weight_int8.copy_(tensors[f"{safe}.weight_int8"])
                qe.weight_scale.copy_(tensors[f"{safe}.weight_scale"])
            elif mode == "int4":
                qe.weight_packed.copy_(tensors[f"{safe}.weight_packed"])
                qe.weight_scale.copy_(tensors[f"{safe}.weight_scale"])
                qe.weight_min.copy_(tensors[f"{safe}.weight_min"])
            _replace_module(model, name, qe)

    # Load non-quantized parameters
    non_quantized_st = quantized_dir / "non_quantized.safetensors"
    if non_quantized_st.exists():
        from safetensors.torch import load_file
        remaining = load_file(str(non_quantized_st), device=device)
    else:
        remaining = torch.load(quantized_dir / "non_quantized.pt", map_location=device, weights_only=True)

    for pname, tensor in remaining.items():
        parts = pname.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        target = getattr(parent, parts[-1], None)
        if target is not None:
            if isinstance(target, nn.Parameter):
                target.data.copy_(tensor)
            elif isinstance(target, torch.Tensor):
                target.copy_(tensor)

    model.eval()

    # Set up next-layer prefetch chain if offloading
    if config.cpu_offload_pass2:
        enable_prefetch_chain(model)

    logger.info(f"Loaded quantized model from {quantized_dir}")
    return model


# ---------------------------------------------------------------------------
# Prefetch chain for CPU-offloaded residual pass
# ---------------------------------------------------------------------------


def enable_prefetch_chain(model: nn.Module) -> int:
    """Link offloaded TurboQuantLinear layers with shared double-buffered scratch.

    Allocates a ``SharedScratchPool`` with two GPU scratch slots sized to the
    largest offloaded layer, then assigns alternating slot indices (ping-pong)
    so one slot is written by H2D while the other is consumed by a kernel.

    Also links layers so each one's ``forward()`` starts an async H2D copy
    of the *next* offloaded layer's pass2 data, overlapping the copy with
    the current layer's kernel execution.

    **Important:** The chain is built in *execution order*, not module-tree
    order.  A single forward pass with dummy input is run to discover the
    actual call sequence.  Module-tree (DFS) order can differ from execution
    order in hybrid architectures (e.g. Qwen3.5 where ``out_proj`` appears
    before ``in_proj_*`` in the tree but executes after them).

    VRAM overhead: 2 × max_layer_pass2_size (constant, regardless of N layers).
    Without this, no GPU scratch exists and offloaded layers fall back to
    synchronous copy.

    Call this after ``quantize_model`` or ``load_quantized`` with
    ``cpu_offload_pass2=True``.

    Args:
        model: model with TurboQuantLinear modules (some offloaded)

    Returns:
        Number of links created (= offloaded_layers - 1, or 0 if < 2)
    """
    offloaded_set: set[int] = set()
    for module in model.modules():
        if isinstance(module, TurboQuantLinear) and module.is_pass2_offloaded:
            offloaded_set.add(id(module))

    if not offloaded_set:
        return 0

    # --- Discover execution order via a forward hook ---
    exec_order: list[TurboQuantLinear] = []
    seen_ids: set[int] = set()
    hooks = []

    def _record_hook(mod, _input, _output):
        mid = id(mod)
        if mid in offloaded_set and mid not in seen_ids:
            exec_order.append(mod)
            seen_ids.add(mid)

    for module in model.modules():
        if isinstance(module, TurboQuantLinear):
            hooks.append(module.register_forward_hook(_record_hook))

    # Dummy forward to discover order (using the model's own device)
    device = next(model.parameters()).device
    dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            pass  # some models may error on length-1 input; order is still valid

    for h in hooks:
        h.remove()

    # Clear caches populated during the dummy forward (avoids holding
    # ~8× the packed indices as int32 in VRAM).
    for module in model.modules():
        if isinstance(module, TurboQuantLinear):
            module._cached_indices = None
            module._cached_pass2_indices = None
            module._rotation_cache.clear()

    offloaded = exec_order
    if not offloaded:
        # Fallback: use module-tree order
        offloaded = [m for m in model.modules()
                     if isinstance(m, TurboQuantLinear) and m.is_pass2_offloaded]

    # Find device and max element counts across all offloaded layers
    device = offloaded[0].indices_packed.device
    max_idx_numel = 0
    max_nrm_numel = 0
    max_cb_numel = 0

    for layer in offloaded:
        cpu_idx = layer._pass2_cpu_indices_packed
        cpu_nrm = layer._pass2_cpu_weight_norms
        cpu_cb = layer._pass2_cpu_codebook
        max_idx_numel = max(max_idx_numel, cpu_idx.numel())
        max_nrm_numel = max(max_nrm_numel, cpu_nrm.numel())
        max_cb_numel = max(max_cb_numel, cpu_cb.numel())

    # Allocate shared double-buffered scratch pool (flat 1-D buffers)
    pool = SharedScratchPool(max_idx_numel, max_nrm_numel, max_cb_numel, device)

    # Share a single copy stream and scratch pool across all layers
    shared_stream = offloaded[0]._copy_stream
    if shared_stream is None:
        shared_stream = torch.cuda.Stream(device=device)

    for i, layer in enumerate(offloaded):
        layer._copy_stream = shared_stream
        layer._scratch_pool = pool
        layer._scratch_idx = i % 2  # ping-pong

    # Chain: layer[i] prefetches layer[i+1]
    # Use object.__setattr__ to avoid nn.Module child-module registration,
    # which would create a recursive chain visible to named_modules()/state_dict().
    for i in range(len(offloaded) - 1):
        object.__setattr__(offloaded[i], '_next_offloaded_layer', offloaded[i + 1])

    # Last layer doesn't prefetch anyone
    object.__setattr__(offloaded[-1], '_next_offloaded_layer', None)

    pool_mb = pool.memory_bytes() / (1024 * 1024)
    logger.info(
        f"Prefetch chain: {len(offloaded)} offloaded layers, "
        f"shared scratch pool {pool_mb:.1f} MB VRAM "
        f"({len(offloaded) - 1} prefetch links)"
    )
    return len(offloaded) - 1


def disable_prefetch_chain(model: nn.Module) -> None:
    """Remove next-layer prefetch links and shared scratch from all layers."""
    for module in model.modules():
        if isinstance(module, TurboQuantLinear):
            object.__setattr__(module, '_next_offloaded_layer', None)
            module._prefetch_event = None
            module._scratch_pool = None
            module._scratch_idx = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _replace_module(model: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
