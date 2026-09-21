#!/usr/bin/env python3
"""Save/load for quantized checkpoints (wxa16 packed format on disk).

Save: after quantization, write the packed weights plus the remaining fp16
      parameters into safetensors, and the non-tensor metadata (seeds,
      group sizes, rotations, expert offsets, router conventions, ...) into
      meta.json.
Load: build an empty structure on the base model path with device_map="meta",
      swap in the quantized modules, then fill tensors with
      load_state_dict(assign=True) — skipping calibration and in-place
      quantization entirely.

Saved seeds are the actual values used at quantization time (MoE uses the
fixed 42+bit / 42+bit+1000 convention, linears use args.seed + layer_idx);
loading does not depend on the formulas — the QR rotation matrices are
regenerated from the seeds at inference time, bit-for-bit identical to the
pre-save state. Behavioral conventions (router_mode / shared_mode /
mlp_returns_tuple) are stored in meta.json and restored on load; see
quantization/common/adapters.py.
"""

import gc
import json
import os
import shutil
import time

import torch

from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization.wxa16.bit_partitioned_moe import WxA16BitPartitionedGroupMoE, WxA16Weights
from quantization.wxa8.linear import is_uniform_codebook
from quantization.wxa16.linear import WxA16Linear

DEV = torch.device('cuda:0')


# ---------------------------------------------------------------------------
# Metadata collection (save side)
# ---------------------------------------------------------------------------

def _linear_meta(linear: WxA16Linear) -> dict:
    """Collect non-state_dict metadata plus buffer shapes of a WxA16Linear."""
    return {
        "type": "wxa16_linear",
        "in_features": linear.in_features,
        "out_features": linear.out_features,
        "bit_width": linear.bit_width,
        "group_size": linear.group_size,
        "seed": linear.seed,
        "rotation": linear.rotation,
        "orig_dtype": str(linear.orig_dtype),
        "has_bias": linear.bias is not None,
        "indices_packed_shape": list(linear.packed_indices.shape),
        "codebook_shape": list(linear.codebook.shape),
        "norms_shape": list(linear.norms.shape),
        "bias_shape": list(linear.bias.shape) if linear.bias is not None else None,
        # Needed by the WxA8 attention path: only uniform codebooks can be
        # safely converted to int8 (old checkpoints lack this field and are
        # treated as "lloydmax"; conversion then keeps them on W8A16)
        "codebook_type": getattr(linear, "codebook_type", "lloydmax"),
    }


def _packed_meta(packed: dict) -> dict:
    """Collect non-tensor metadata plus tensor shapes of one packed dict
    (gate_up or down)."""
    return {
        "seed": packed.get("seed"),
        "group_size": packed.get("group_size"),
        "shape": list(packed["shape"]),
        "bit_width": packed.get("bit_width"),
        "rotation": packed.get("rotation"),
        "orig_dtype": packed.get("orig_dtype"),
        "indices_packed_shape": list(packed["indices_packed"].shape),
        "codebook_shape": list(packed["codebook"].shape),
        "norms_shape": list(packed["norms"].shape),
    }


def _moe_meta(moe: WxA16BitPartitionedGroupMoE) -> dict:
    """Collect non-state_dict metadata of a WxA16BitPartitionedGroupMoE.

    expert_offsets lives in a plain dict of LongTensors (outside state_dict,
    and layer.to('cpu') does not move it), so it must be converted to CPU
    lists for JSON.
    """
    bits = {}
    for bit_str, weights in moe.bit_weights.items():
        bits[bit_str] = {
            "gate_up": _packed_meta(weights.gate_up_packed),
            "down": _packed_meta(weights.down_packed),
        }

    expert_offsets = {}
    for bit_str, offsets in moe.expert_offsets.items():
        expert_offsets[bit_str] = offsets.cpu().tolist()

    return {
        "type": "wxa16_moe",
        "num_experts": moe.num_experts,
        "hidden_size": moe.hidden_size,
        "intermediate_size": moe.intermediate_size,
        "top_k": moe.top_k,
        "bit_list": list(moe.bit_list),
        "inter_size_by_bit": {str(k): v for k, v in moe.inter_size_by_bit.items()},
        "expert_offsets": expert_offsets,
        "bits": bits,
        "enable_timing": moe.enable_timing,
        # Behavioral conventions (quantization/common/adapters.py);
        # restored on the load side from these fields
        "router_mode": moe.router_mode,
        "shared_mode": moe.shared_mode,
        "mlp_returns_tuple": moe.mlp_returns_tuple,
    }


def collect_quant_metadata(model, quant_args: dict = None, model_id: str = None,
                           base_model: str = None) -> dict:
    """Walk the model and collect quantized-module attributes by dotted path
    (= state_dict key prefix)."""
    meta = {
        "base_model": base_model,
        "model_class": model.__class__.__name__,
        "quant_args": quant_args or {},
        "modules": {},
        "layers": {},
    }
    if model_id:
        meta["model_id"] = model_id

    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, WxA16Linear):
            meta["modules"][name] = _linear_meta(module)
        elif isinstance(module, WxA16BitPartitionedGroupMoE):
            meta["layers"][name] = _moe_meta(module)

    return meta


# ---------------------------------------------------------------------------
# qmeta seeds (redundant safetensors copy + cross-check)
# ---------------------------------------------------------------------------

def _build_qmeta_tensors(meta: dict) -> dict:
    """Convert quantized-module seeds into scalar int64 tensors prefixed
    with "qmeta.".

    safetensors only stores tensors (strings such as rotation/orig_dtype and
    the expert_offsets lists remain in meta.json); the seeds are written as a
    redundant copy and cross-checked against meta.json on load.
    """
    qmeta = {}
    for path, m in meta.get("modules", {}).items():
        if m.get("seed") is not None:
            qmeta[f"qmeta/{path}/seed"] = torch.tensor(m["seed"], dtype=torch.int64)
    for path, lm in meta.get("layers", {}).items():
        for bit_str, bm in lm["bits"].items():
            for which in ("gate_up", "down"):
                seed_val = bm[which].get("seed")
                if seed_val is not None:
                    qmeta[f"qmeta/{path}/bits/{bit_str}/{which}/seed"] = torch.tensor(seed_val, dtype=torch.int64)
    qa_seed = meta.get("quant_args", {}).get("seed")
    if qa_seed is not None:
        qmeta["qmeta/quant_args/seed"] = torch.tensor(qa_seed, dtype=torch.int64)
    return qmeta


def _check_qmeta_seeds(meta: dict, qmeta_tensors: dict):
    """Cross-check the qmeta seeds in safetensors against meta.json
    (guards against mismatched files)."""
    expected = {}
    for path, m in meta.get("modules", {}).items():
        if m.get("seed") is not None:
            expected[f"qmeta/{path}/seed"] = m["seed"]
    for path, lm in meta.get("layers", {}).items():
        for bit_str, bm in lm["bits"].items():
            for which in ("gate_up", "down"):
                seed_val = bm[which].get("seed")
                if seed_val is not None:
                    expected[f"qmeta/{path}/bits/{bit_str}/{which}/seed"] = seed_val
    qa_seed = meta.get("quant_args", {}).get("seed")
    if qa_seed is not None:
        expected["qmeta/quant_args/seed"] = qa_seed

    if not qmeta_tensors:
        print("  [WARN] no qmeta seeds in safetensors (old format or not written)")
        return

    mismatched = []
    missing = []
    for key, want in expected.items():
        got = qmeta_tensors.get(key)
        if got is None:
            missing.append(key)
        elif int(got.item()) != want:
            mismatched.append((key, want, int(got.item())))

    if mismatched:
        print(f"  [WARN] qmeta seeds disagree with meta.json in {len(mismatched)} places: "
              f"{mismatched[:5]}... (files may be mismatched)")
    elif missing:
        print(f"  [WARN] {len(missing)} qmeta seed keys missing: {missing[:5]}...")
    else:
        print(f"  [OK] safetensors qmeta seeds match meta.json ({len(expected)} keys)")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _print_size_breakdown(sd: dict, total_bytes: int):
    """Print per-category checkpoint sizes (qmeta scalar keys excluded)."""
    cats = {
        "MoE packed (bit_weights)": 0,
        "Attn/Shared packed (indices/codebook/norms)": 0,
        "fp16 remaining params": 0,
    }
    for name, t in sd.items():
        if name.startswith("qmeta/"):
            continue
        nbytes = t.numel() * t.element_size()
        if "bit_weights" in name:
            cats["MoE packed (bit_weights)"] += nbytes
        elif name.endswith(("packed_indices", "codebook", "norms")):
            cats["Attn/Shared packed (indices/codebook/norms)"] += nbytes
        else:
            cats["fp16 remaining params"] += nbytes

    print("\n  [Quantized Checkpoint Size Breakdown]")
    for cat, nbytes in cats.items():
        print(f"    {cat}: {nbytes / 1024**3:.2f}GB")
    print(f"    total: {total_bytes / 1024**3:.2f}GB")


def _copy_base_aux_files(base_model_path: str, save_dir: str, verbose: bool = True):
    """Copy config/tokenizer/custom modeling files from the base model dir so
    the checkpoint directory is self-contained.

    Custom code is copied according to config.json's auto_map references
    (deepseek family), with a fallback that copies every .py file in the dir
    (custom modeling code may import helper files indirectly).
    """
    copied = []
    for fname in ("config.json", "generation_config.json",
                  "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "vocab.json", "merges.txt",
                  "tokenizer.model"):
        src = os.path.join(base_model_path, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(save_dir, fname))
            copied.append(fname)

    config_path = os.path.join(base_model_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for target in cfg.get("auto_map", {}).values():
                py_file = target.split(".")[0] + ".py"
                src = os.path.join(base_model_path, py_file)
                if os.path.isfile(src) and py_file not in copied:
                    shutil.copy(src, os.path.join(save_dir, py_file))
                    copied.append(py_file)
        except Exception as e:
            print(f"  [WARN] auto_map parsing failed, skipping custom code copy: {e}")

    if os.path.isdir(base_model_path):
        for fname in sorted(os.listdir(base_model_path)):
            if fname.endswith(".py") and fname not in copied:
                src = os.path.join(base_model_path, fname)
                if os.path.isfile(src):
                    shutil.copy(src, os.path.join(save_dir, fname))
                    copied.append(fname)

    if verbose:
        print(f"Copied base model aux files ({len(copied)}): {copied[:8]}{'...' if len(copied) > 8 else ''}")


def save_quantized_model(model, save_dir: str, base_model_path: str = None,
                         quant_args: dict = None, verbose: bool = True) -> dict:
    """Save the quantized model as safetensors + meta.json.

    Only CPU copies are made; the model's own device state is not modified,
    so evaluation after saving is unaffected.
    """
    tick0 = time.time()
    os.makedirs(save_dir, exist_ok=True)

    if base_model_path:
        _copy_base_aux_files(base_model_path, save_dir, verbose=verbose)

    meta = collect_quant_metadata(
        model,
        quant_args=quant_args,
        model_id=getattr(model, "model_id", None),
        base_model=base_model_path,
    )
    meta_path = os.path.join(save_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"Saved metadata: {meta_path} (modules={len(meta['modules'])}, layers={len(meta['layers'])})")

    # Move state_dict tensors to CPU one key at a time
    # (safetensors requires CPU + contiguous)
    print("Collecting state_dict to CPU...")
    sd = {}
    seen_ids = set()
    for name, t in model.state_dict().items():
        if id(t) in seen_ids:
            raise AssertionError(f"state_dict contains a shared tensor {name} "
                                 f"(the current model should have no ties)")
        seen_ids.add(id(t))
        sd[name] = t.detach().cpu() if t.device.type != "cpu" else t.detach()

    # Redundant seeds written into safetensors ("qmeta." scalar int64 tensors)
    sd.update(_build_qmeta_tensors(meta))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sd_path = os.path.join(save_dir, "model.safetensors")
    print(f"Writing safetensors to {sd_path} (this may take a while)...")
    save_file(sd, sd_path)
    total_bytes = os.path.getsize(sd_path)

    if verbose:
        _print_size_breakdown(sd, total_bytes)

    del sd
    gc.collect()
    print(f"Saved quantized checkpoint to {save_dir} in {time.time() - tick0:.2f}s, "
          f"total {total_bytes / 1024**3:.2f}GB")
    return meta


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _split_path(root, path: str):
    """Locate (parent module, attribute name) from a dotted path."""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _materialize_meta_buffers(model) -> list:
    """Materialize meta non-persistent buffers left over after assign load.

    RoPE-related buffers (inv_freq / cos_cached / sin_cached /
    original_inv_freq) are registered with persistent=False, so
    load_state_dict neither matches them strictly nor fills them, and
    model.to(DEV) then fails with "Cannot copy out of meta tensor". They are
    recomputed deterministically:
      - inv_freq: standard RoPE formula 1/(base**(arange(0,dim,2)/dim)),
        with base taken from module.base -> config.rope_theta -> 10000
        (transformers-style modules prefer compute_default_rope_parameters)
      - cos/sin caches: recomputed by the module's own _set_cos_sin_cache
        (deepseek-family Yarn scaling lives inside the module), with seq_len
        taken from the meta buffer shape
    Two passes are required: the cos/sin computation depends on inv_freq.
    """
    handled = []
    unknown = []

    # Pass 1: inv_freq first (_set_cos_sin_cache depends on it)
    for name, buf in list(model.named_buffers()):
        if not buf.is_meta:
            continue
        parent_name, _, buf_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if buf_name in ("inv_freq", "original_inv_freq") and hasattr(parent, "compute_default_rope_parameters"):
            inv_freq, _ = parent.compute_default_rope_parameters(parent.config, None)
            parent.register_buffer("inv_freq", inv_freq, persistent=False)
            parent.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
            handled.append(name)
        elif buf_name == "inv_freq":
            dim = buf.shape[0] * 2
            base = (getattr(parent, "base", None)
                    or getattr(getattr(parent, "config", None), "rope_theta", None)
                    or 10000.0)
            inv_freq = 1.0 / (float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            parent.register_buffer("inv_freq", inv_freq, persistent=False)
            handled.append(name)

    # Pass 2: cos/sin caches via the module's own method
    # (Yarn scaling and friends are handled inside the module)
    for name, buf in list(model.named_buffers()):
        if not buf.is_meta:
            continue
        parent_name, _, buf_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if hasattr(parent, "_set_cos_sin_cache"):
            seq_len = buf.shape[0] if buf.dim() >= 1 else getattr(parent, "max_seq_len_cached", 4096)
            parent._set_cos_sin_cache(seq_len, torch.device("cpu"), torch.get_default_dtype())
            handled.append(name)
        else:
            unknown.append(name)

    if handled:
        print(f"Materialized meta buffers ({len(handled)}): {handled[:8]}{'...' if len(handled) > 8 else ''}")
    if unknown:
        print(f"  [WARN] meta buffers still not materialized: {unknown} "
              f"(first forward may fail)", flush=True)
    return handled


def restore_quant_metadata(model, meta: dict, state_dict: dict = None):
    """Swap in the quantized modules according to the metadata and fill tensors.

    Ordering requirements:
      1. replace WxA16Linear inside shared experts / attention first
         (still under the original mlp structure)
      2. then replace the mlp as a whole with WxA16BitPartitionedGroupMoE
         (reusing the already-swapped shared modules)
      3. finally load_state_dict(strict=True, assign=True)
    """
    # Step 1: replace WxA16Linear (attention and shared-expert internals)
    for path, m in meta.get("modules", {}).items():
        if m.get("type") != "wxa16_linear":
            continue
        parent, attr = _split_path(model, path)
        setattr(parent, attr, WxA16Linear.from_metadata(m))

    # Step 2: replace the MoE as a whole
    for path, m in meta.get("layers", {}).items():
        parent, attr = _split_path(model, path)
        old_mlp = getattr(parent, attr)
        new_moe = WxA16BitPartitionedGroupMoE.from_metadata(
            m,
            gate=old_mlp.gate,
            shared_expert=getattr(old_mlp, "shared_expert", None),
            shared_expert_gate=getattr(old_mlp, "shared_expert_gate", None),
            shared_experts=getattr(old_mlp, "shared_experts", None),
        )
        setattr(parent, attr, new_moe)

        # Drop references from the old structure (same cleanup as the
        # quantize path in dartmoq_sequential)
        if hasattr(old_mlp, "gate"):
            del old_mlp.gate
        if hasattr(old_mlp, "shared_expert"):
            del old_mlp.shared_expert
        if hasattr(old_mlp, "shared_expert_gate"):
            del old_mlp.shared_expert_gate
        if hasattr(old_mlp, "shared_experts"):
            del old_mlp.shared_experts
        del old_mlp

    # Step 3: fill tensors (strict match: keys must correspond one-to-one).
    # The "qmeta." seed scalars are not part of the module structure;
    # strip them and cross-check against meta.json
    if state_dict is not None:
        qmeta_tensors = {k: v for k, v in state_dict.items() if k.startswith("qmeta/")}
        sd_weights = {k: v for k, v in state_dict.items() if not k.startswith("qmeta/")}
        res = model.load_state_dict(sd_weights, strict=True, assign=True)
        if res.missing_keys:
            raise AssertionError(f"load_state_dict missing keys: {res.missing_keys[:10]}...")
        if res.unexpected_keys:
            raise AssertionError(f"load_state_dict unexpected keys: {res.unexpected_keys[:10]}...")
        _materialize_meta_buffers(model)
        _check_qmeta_seeds(meta, qmeta_tensors)

    return model


def convert_model_to_wxa8(model):
    """Switch a loaded WxA16 model to the WxA8 inference path in place.

    WxA8 shares the packed storage format with WxA16 (the checkpoint is
    common to both); the codebooks are converted to INT8 at load time, so
    this only swaps __class__ — zero tensor copies, zero extra memory.

    Conversion scope:
      - MoE: always converted to WxA8BitPartitionedGroupMoE
      - 8-bit linears (attention / shared experts): converted only when the
        codebook is uniform (codebook_type="uniform") AND in_features is a
        multiple of group_size (the A8 kernels group the K dimension by 128).
        Lloyd-Max codebooks (including old checkpoints) stay on W8A16 with a
        warning — stuffing them into an int8 grid would collapse ~61 levels
        (~7.6 effective bits); non-aligned linears (e.g. the dense first
        layer's down_proj with in_features=10944 in dsv1/dsv2) also stay on
        W8A16 and use the A16 fallback path.
    """
    from quantization.wxa8 import WxA8BitPartitionedGroupMoE, WxA8Linear

    n_moe = 0
    n_attn = 0
    n_kept = 0
    n_misaligned = 0
    for sub in model.modules():
        if isinstance(sub, WxA16BitPartitionedGroupMoE):
            WxA8BitPartitionedGroupMoE.from_wxa16(sub)
            n_moe += 1
        elif isinstance(sub, WxA16Linear):
            if sub.bit_width == 8 and is_uniform_codebook(sub.codebook):
                if sub.in_features % sub.group_size == 0:
                    WxA8Linear.from_wxa16(sub)
                    n_attn += 1
                else:
                    n_misaligned += 1
            elif sub.bit_width == 8:
                n_kept += 1
            # non-8-bit linears do not occur in the main flow
            # (attention/shared are always 8-bit)
    print(f"Converted to WxA8: {n_moe} MoE layers, {n_attn} attention/shared linears")
    if n_kept:
        print(f"  [WARN] {n_kept} 8-bit linears kept on W8A16: non-uniform (Lloyd-Max) "
              f"codebook would collapse to ~7.6 effective bits in int8. "
              f"Re-quantize with a uniform codebook to enable the conversion.")
    if n_misaligned:
        print(f"  [INFO] {n_misaligned} 8-bit linears kept on W8A16: in_features not a "
              f"multiple of group_size ({128}) — the A8 kernels group the K "
              f"dimension by group_size, so these use the A16 fallback path.")
    return model


def load_quantized_model(base_model_path: str = None, quant_dir: str = None,
                         standby_cpu: bool = False, seqlen: int = 2048,
                         inference_quant_mode: str = "wxa16"):
    """Load a quantized checkpoint, skipping calibration and in-place quantization.

    Args:
        base_model_path: original model path (defaults to quant_dir, since the
            checkpoint directory is self-contained — config/tokenizer/custom
            modeling files are copied at save time)
        quant_dir: save directory (contains model.safetensors + meta.json)
        standby_cpu: keep the model on CPU after loading (for the per-layer
            sequential eval used by large models)
        seqlen: model sequence length (fixed 2048, same as the original
            loading path)
        inference_quant_mode: "wxa16" (default) or "wxa8". wxa8 switches the
            MoE to the INT8-activation path in place after restore (the
            checkpoint itself is unchanged)
    """
    if base_model_path is None:
        # Prefer the base_model path recorded in meta.json (also covers old
        # checkpoints that did not copy the tokenizer files)
        meta_path_tmp = os.path.join(quant_dir, "meta.json")
        if os.path.isfile(meta_path_tmp):
            with open(meta_path_tmp, "r", encoding="utf-8") as f:
                meta_tmp = json.load(f)
            saved_base = meta_tmp.get("base_model")
            if saved_base and os.path.isdir(saved_base):
                base_model_path = saved_base
        if base_model_path is None:
            base_model_path = quant_dir
    print(f"Loading quantized checkpoint from: {quant_dir}")
    print(f"Base model path (config/tokenizer/remote code): {base_model_path}")
    tick0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print("Building model structure on meta device (no weights loaded)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="meta",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"Meta structure built in {time.time() - tick0:.2f}s")

    meta_path = os.path.join(quant_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    sd_path = os.path.join(quant_dir, "model.safetensors")
    print(f"Loading quantized tensors: {sd_path} ({os.path.getsize(sd_path) / 1024**3:.2f}GB)...")
    sd = load_file(sd_path)
    print(f"Loaded {len(sd)} tensors in {time.time() - tick0:.2f}s")

    restore_quant_metadata(model, meta, sd)

    # WxA8 inference mode: switch the MoE to the INT8-activation path in
    # place (checkpoint format unchanged; the same 2bpw checkpoint runs in
    # both wxa16 and wxa8)
    if inference_quant_mode == "wxa8":
        convert_model_to_wxa8(model)
    elif inference_quant_mode != "wxa16":
        raise ValueError(f"unknown inference_quant_mode: {inference_quant_mode}")

    del sd
    gc.collect()

    # Attributes matching eval_dartmoq.load_model
    model.seqlen = seqlen
    model.model_id = meta.get("model_id") or os.path.basename(base_model_path.rstrip("/"))
    model.eval()

    if not standby_cpu:
        print(f"Moving quantized model to {DEV}...")
        model.to(DEV)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Quantized model loaded in {time.time() - tick0:.2f}s")
    return model, tokenizer
