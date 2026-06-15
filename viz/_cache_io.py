"""Shared cache-IO helpers.

The DartMoQ pipeline ([dartmoq_layer_reconstruct.py:80](../dartmoq_layer_reconstruct.py#L80))
caches per-(model, layer, bit) sensitivity tensors under

    intermediate_result/quant_outlier_{quantmode}/{rank_mode}/{model_id}/{model_id}_L{layer}_b{bit}.pt

Each `.pt` file is a `List[Tensor]` whose length equals the number of experts and
each element has shape `(n_neurons,)`. We never recompute these from scratch in the
viz module — all figures are derived from these cached tensors so that the entire
plot pipeline is decoupled from quantization runtime.

Conventions used by the viz modules
-----------------------------------
- `quantmode` ∈ {"gptq", "turboquant"} — selects the top-level cache directory
- `rank_mode` ∈ {"gptq_quant_outlier", "turboquant_innerproduct", ...} — selects
  the sensitivity-metric subdirectory
- `bits` are integers in {0,1,2,3,4}; bit 0 may be a real cached file or may be
  re-extrapolated via dp_utils.extrapolate_0bit_loss
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


INTERMEDIATE_RESULT_DIR = "intermediate_result"
CACHE_ROOT_PATTERN = os.path.join(INTERMEDIATE_RESULT_DIR, "quant_outlier_{quantmode}", "{rank_mode}", "{model_id}")
EXPERT_ACTIVATE_ROOT = os.path.join(INTERMEDIATE_RESULT_DIR, "expert_activate")
expert_cosine_ROOT = os.path.join(INTERMEDIATE_RESULT_DIR, "expert_cosine")


# -------- model registry ----------------------------------------------------
# Maps a short id (used inside cache paths) -> human-readable label for plot legends.
KNOWN_MODELS = {
    "deepseek-v1-moe-16b": "DeepSeekMoE-V1-16B",
    "deepseek-v2-lite":    "DeepSeek-V2-Lite",
    "moonlight":           "Moonlight-16B-A3B",
    "olmoe-7b-1b":         "OLMoE-1B-7B",
    "qwen3-30b-a3b":       "Qwen3-30B-A3B",
}

# (substring-in-path, short_id). Kept in sync with `eval_dartmoq.load_model`.
# When the user passes `--model /home/...../deepseek-moe-16b-base/`, we strip
# directory parts and lowercase, then look for any of these substrings. First
# match wins, so order from most-specific to least-specific.
_PATH_TO_ID = [
    ("olmoe",            "olmoe-7b-1b"),
    ("deepseek-moe-16b", "deepseek-v1-moe-16b"),
    ("deepseek-v2-lite", "deepseek-v2-lite"),
    ("moonlight",        "moonlight"),
    ("qwen3-30b-a3b",    "qwen3-30b-a3b"),
]


def resolve_model_id(name_or_path: str) -> str:
    """Accept either a short cache id (e.g. ``olmoe-7b-1b``) or a full model
    path (e.g. ``/home/user/models/OLMoE-1B-7B-0924/``), and return the short
    cache id that the pipeline uses for ``intermediate_result/quant_outlier_*/{rank}/{id}/``.

    Raises ``ValueError`` if no mapping is found, so the caller fails loudly
    rather than silently searching a non-existent cache directory.
    """
    if name_or_path in KNOWN_MODELS:
        return name_or_path

    needle = name_or_path.rstrip("/\\").lower()
    needle = os.path.basename(needle) or needle  # strip leading dirs
    # Try the basename first, then the full string — covers both `--model foo/`
    # and `--model /a/b/foo`.
    for haystack in (needle, name_or_path.lower()):
        for substr, short_id in _PATH_TO_ID:
            if substr in haystack:
                return short_id

    raise ValueError(
        f"cannot map '{name_or_path}' to a known cache id; "
        f"known ids = {sorted(KNOWN_MODELS)}; "
        f"path substrings = {[s for s, _ in _PATH_TO_ID]}. "
        f"If this is a new model, add it to viz/_cache_io.py."
    )


# Reverse map: short cache id -> on-disk model path (used by viz/dump_*.py
# scripts that need to load the actual model). Kept in sync with the paths in
# `run.sh`. If `$HOME/models/` is not where your checkpoints live, override
# with the `DARTMOQ_MODELS_ROOT` environment variable.
_MODELS_ROOT = os.environ.get("DARTMOQ_MODELS_ROOT",
                              os.path.expanduser("~/models"))
_ID_TO_PATH = {
    "deepseek-v1-moe-16b": "deepseek-moe-16b-base",
    "deepseek-v2-lite":    "DeepSeek-V2-Lite",
    "moonlight":           "Moonlight-16B-A3B",
    "olmoe-7b-1b":         "OLMoE-1B-7B-0924-Instruct",
    "qwen3-30b-a3b":       "Qwen3-30B-A3B",
}


def resolve_model_path(name_or_path: str) -> str:
    """Inverse of resolve_model_id. Accepts either a short id (looks up the
    on-disk path under ``$DARTMOQ_MODELS_ROOT`` / default ``~/models``) or an
    already-valid path (returns it unchanged after sanity-checking it exists).
    """
    # already a usable path?
    if os.path.isdir(name_or_path):
        return name_or_path

    # short id → ~/models/<canonical-dirname>
    short_id = name_or_path if name_or_path in _ID_TO_PATH else resolve_model_id(name_or_path)
    dirname = _ID_TO_PATH.get(short_id)
    if dirname is None:
        raise ValueError(
            f"no on-disk path registered for short id '{short_id}'. "
            f"Add it to _ID_TO_PATH in viz/_cache_io.py, or pass the full path."
        )
    path = os.path.join(_MODELS_ROOT, dirname)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"resolved '{short_id}' → '{path}' but that directory does not exist. "
            f"Either copy the model there, or set DARTMOQ_MODELS_ROOT to the "
            f"directory that contains '{dirname}'."
        )
    return path


# -------- structure --------------------------------------------------------
@dataclass
class LayerSensitivity:
    """Per-layer sensitivity bundle for a single (model, quantmode, rank_mode)."""
    model_id: str
    quantmode: str
    rank_mode: str
    layer_idx: int
    # bit -> list[expert_idx] -> 1-D np.array(n_neurons)
    by_bit: Dict[int, List[np.ndarray]] = field(default_factory=dict)

    @property
    def n_experts(self) -> int:
        any_bit = next(iter(self.by_bit))
        return len(self.by_bit[any_bit])

    @property
    def n_neurons(self) -> int:
        any_bit = next(iter(self.by_bit))
        return self.by_bit[any_bit][0].shape[0]

    def bits_sorted(self) -> List[int]:
        return sorted(self.by_bit.keys())


def cache_dir(quantmode: str, rank_mode: str, model_id: str) -> str:
    return CACHE_ROOT_PATTERN.format(
        quantmode=quantmode, rank_mode=rank_mode, model_id=model_id
    )


def load_layer(
    model_id: str,
    layer_idx: int,
    quantmode: str,
    rank_mode: str,
    bits: Iterable[int] = (1, 2, 3, 4),
    dir_suffix: str = "",
    device: str = "cpu",
) -> Optional[LayerSensitivity]:
    """Load all cached `.pt` files for one (model, quantmode, rank_mode, layer)."""
    base = cache_dir(quantmode, rank_mode, model_id) + dir_suffix
    if not os.path.isdir(base):
        return None

    out = LayerSensitivity(
        model_id=model_id, quantmode=quantmode,
        rank_mode=rank_mode, layer_idx=layer_idx,
    )
    for b in bits:
        path = os.path.join(base, f"{model_id}_L{layer_idx}_b{b}.pt")
        if not os.path.exists(path):
            continue
        raw = torch.load(path, map_location=device)
        out.by_bit[b] = [
            t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
            for t in raw
        ]
    return out if out.by_bit else None


def discover_layers(quantmode: str, rank_mode: str, model_id: str,
                    dir_suffix: str = "") -> List[int]:
    """Return the sorted list of layer indices that have at least one cached bit."""
    base = cache_dir(quantmode, rank_mode, model_id) + dir_suffix
    if not os.path.isdir(base):
        return []
    pat = re.compile(rf"{re.escape(model_id)}_L(\d+)_b\d+\.pt$")
    seen = set()
    for fn in os.listdir(base):
        m = pat.match(fn)
        if m:
            seen.add(int(m.group(1)))
    return sorted(seen)


def discover_models(quantmode: str, rank_mode: str) -> List[str]:
    """Return all model ids that have cached data for (quantmode, rank_mode)."""
    base = os.path.join(INTERMEDIATE_RESULT_DIR, f"quant_outlier_{quantmode}", rank_mode)
    if not os.path.isdir(base):
        return []
    return sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and not d.endswith("_bak")
        and "-whole-" not in d  # legacy directories
    )


def load_all_layers(
    model_id: str, quantmode: str, rank_mode: str,
    bits: Iterable[int] = (1, 2, 3, 4),
    layer_start: Optional[int] = None,
    num_layers: Optional[int] = None,
    layer_end: Optional[int] = None,  # keep for backward compatibility
    dir_suffix: str = "",
) -> List[LayerSensitivity]:
    """Load every cached layer for one (model, quantmode, rank_mode)."""
    layers = discover_layers(quantmode, rank_mode, model_id, dir_suffix=dir_suffix)
    if layer_start == -1:
        if num_layers is not None:
            layers = layers[-num_layers:] if num_layers > 0 else []
    elif layer_start is not None:
        layers = [li for li in layers if li >= layer_start]
    if layer_end is not None:  # backward compatibility
        layers = [li for li in layers if li < layer_end]

    out = []
    for li in layers:
        if num_layers is not None and len(out) >= num_layers:
            break
        ls = load_layer(model_id, li, quantmode, rank_mode, bits, dir_suffix=dir_suffix)
        if ls is not None:
            out.append(ls)
    return out


# -------- aggregations ------------------------------------------------------
def expert_total_loss(layer: LayerSensitivity, bit: int) -> np.ndarray:
    """Sum of per-neuron loss for each expert at a given bit -> shape (n_experts,)."""
    return np.array([rates.sum() for rates in layer.by_bit[bit]])


def neuron_loss_matrix(layer: LayerSensitivity, bit: int) -> np.ndarray:
    """Stack experts into a (n_experts, n_neurons) matrix at a given bit."""
    return np.stack(layer.by_bit[bit], axis=0)


def model_label(model_id: str) -> str:
    return KNOWN_MODELS.get(model_id, model_id)


# -------- plotting house style ---------------------------------------------
def apply_paper_style() -> None:
    """Tighter, paper-friendly matplotlib defaults. Call once at top of each script."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 130,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# -------- expert_cosine cache -------------------------------------------------
def expert_cosine_cache_path(
    model_id: str,
    quantmode: str,
    rank_mode: str,
    bit: int,
    task_name: str,
    layers: Optional[Sequence[int]] = None,
    experts: Optional[Sequence[int]] = None,
    seeds: Optional[Sequence[int]] = None,
    preserve_fracs: Optional[Sequence[float]] = None,
) -> str:
    """Get cache path for expert_cosine summary."""
    sanitized_task = _slug(task_name)
    parts = [model_id, quantmode, rank_mode, f"b{bit}"]
    if layers:
        layer_str = "_".join(str(l) for l in sorted(layers))
        parts.append(f"L{layer_str}")
    if experts:
        expert_str = "_".join(str(e) for e in sorted(experts))
        parts.append(f"E{expert_str}")
    if seeds:
        seed_str = "_".join(str(s) for s in sorted(seeds))
        parts.append(f"S{seed_str}")
    if preserve_fracs:
        frac_str = "_".join(f"{f:.4f}".rstrip('0').rstrip('.') if '.' in f"{f:.4f}" else f"{f:.4f}" for f in preserve_fracs)
        parts.append(f"F{frac_str}")
    parts.append(sanitized_task)
    filename = "_".join(parts) + ".json"
    return os.path.join(expert_cosine_ROOT, filename)


def _validate_cache_compatibility(
    cache: dict,
    seeds: Optional[Sequence[int]] = None,
    preserve_fracs: Optional[Sequence[float]] = None,
    strategies: Optional[Sequence[str]] = None,
) -> bool:
    """Validate if cache is compatible with requested parameters."""
    # Check seeds if provided
    if seeds is not None:
        cache_seeds = cache.get("seeds", [])
        if sorted(cache_seeds) != sorted(seeds):
            return False

    # Check preserve_fracs if provided
    if preserve_fracs is not None:
        cache_fracs = cache.get("preserve_fracs", [])
        # Convert both to float for comparison
        cache_fracs_float = [float(f) for f in cache_fracs]
        req_fracs_float = [float(f) for f in preserve_fracs]
        if cache_fracs_float != req_fracs_float:
            return False

    # Check strategies if provided (check first target)
    if strategies is not None and cache.get("targets"):
        first_target = cache["targets"][0]
        cache_strategies = set(first_target.get("strategies", {}).keys())
        req_strategies = set(strategies)
        if not req_strategies.issubset(cache_strategies):
            return False

    return True


def load_expert_cosine_cache(
    model_id: str,
    quantmode: str,
    rank_mode: str,
    bit: int,
    task_name: str,
    layers: Optional[Sequence[int]] = None,
    experts: Optional[Sequence[int]] = None,
    seeds: Optional[Sequence[int]] = None,
    preserve_fracs: Optional[Sequence[float]] = None,
    strategies: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """Load expert_cosine summary from cache if exists and compatible."""
    # First try with all parameters
    cache_path = expert_cosine_cache_path(
        model_id, quantmode, rank_mode, bit, task_name,
        layers, experts, seeds, preserve_fracs
    )
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            import json
            cache = json.load(f)
            if _validate_cache_compatibility(cache, seeds, preserve_fracs, strategies):
                return cache

    # If not found or incompatible, try legacy path (without seeds/fracs)
    legacy_path = expert_cosine_cache_path(
        model_id, quantmode, rank_mode, bit, task_name, layers, experts
    )
    if os.path.exists(legacy_path):
        with open(legacy_path, "r") as f:
            import json
            cache = json.load(f)
            if _validate_cache_compatibility(cache, seeds, preserve_fracs, strategies):
                return cache

    return None


def save_expert_cosine_cache(
    summary: dict,
    model_id: str,
    quantmode: str,
    rank_mode: str,
    bit: int,
    task_name: str,
    layers: Optional[Sequence[int]] = None,
    experts: Optional[Sequence[int]] = None,
) -> str:
    """Save expert_cosine summary to cache."""
    seeds = summary.get("seeds")
    preserve_fracs = summary.get("preserve_fracs")
    cache_path = expert_cosine_cache_path(
        model_id, quantmode, rank_mode, bit, task_name,
        layers, experts, seeds, preserve_fracs
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        import json
        json.dump(summary, f, indent=2)
    print(f"[expert_cosine] cache saved to {cache_path}")
    return cache_path


def _slug(text: object) -> str:
    """Sanitize text for use in filenames."""
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "unknown"


__all__ = [
    "LayerSensitivity", "KNOWN_MODELS", "INTERMEDIATE_RESULT_DIR",
    "EXPERT_ACTIVATE_ROOT", "expert_cosine_ROOT",
    "cache_dir", "load_layer", "load_all_layers",
    "discover_layers", "discover_models",
    "resolve_model_id", "resolve_model_path",
    "expert_cosine_cache_path", "load_expert_cosine_cache", "save_expert_cosine_cache",
    "expert_total_loss", "neuron_loss_matrix",
    "model_label", "apply_paper_style",
]
