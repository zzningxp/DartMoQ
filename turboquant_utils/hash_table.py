"""TurboQuant-Model: Hash-based extreme weight compression (<1 bpw).

Replaces per-element codebook indices with deterministic multi-head hash
lookup into a global shared table.  Per-group pre-rotation statistics (mean
and standard deviation, quantized to 8-bit each) are incorporated into the
hash key so that groups with similar original distributions collide and share
a reconstruction vector.

Reference:
    TurboQuant-Model: Extreme Weight Compression via Per-Group Hashing
    with Pre-Rotation Statistics
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from turboquant_model.rotation import (
    generate_rotation_matrix,
    hadamard_rotate,
    hadamard_rotate_inverse,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TABLE_SIZE: int = 262_147  # prime near 2^18
DEFAULT_GROUP_SIZE: int = 128
DEFAULT_NUM_HEADS: int = 4

# Large primes for multi-head hashing (one per head)
_HASH_PRIMES: tuple[int, ...] = (
    6_700_417,
    15_485_863,
    32_452_843,
    49_979_687,
    67_867_979,
    86_028_121,
)


# ---------------------------------------------------------------------------
# Pre-rotation statistics
# ---------------------------------------------------------------------------


def compute_group_stats(
    W: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-group mean and std of the *original* (pre-rotation) weights.

    Args:
        W: (M, N) weight matrix (float).
        group_size: number of consecutive columns per group.

    Returns:
        means: (M, n_groups) float32 — per-group mean.
        stds:  (M, n_groups) float32 — per-group std.
    """
    W = W.float()
    M, N = W.shape
    n_groups = math.ceil(N / group_size)
    means = torch.zeros(M, n_groups, device=W.device)
    stds = torch.zeros(M, n_groups, device=W.device)

    for g in range(n_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, N)
        W_g = W[:, g_start:g_end]
        means[:, g] = W_g.mean(dim=1)
        stds[:, g] = W_g.std(dim=1).clamp(min=1e-8)

    return means, stds


def quantize_stats(
    means: torch.Tensor,
    stds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize mean to signed int8 and std to unsigned int8 (log-scale).

    Args:
        means: (M, n_groups) float32.
        stds:  (M, n_groups) float32 (positive).

    Returns:
        mu_q:    (M, n_groups) int8  — quantized mean.
        sigma_q: (M, n_groups) uint8 — quantized log-std.
    """
    # --- Mean: symmetric linear quantize to [-127, 127] ---
    mu_max = means.abs().amax().clamp(min=1e-8)
    mu_q = (means / mu_max * 127.0).round().clamp(-127, 127).to(torch.int8)

    # --- Std: log-scale quantize to [0, 255] ---
    log_sigma = torch.log(stds.clamp(min=1e-8))
    ls_min = log_sigma.amin()
    ls_max = log_sigma.amax()
    ls_range = (ls_max - ls_min).clamp(min=1e-8)
    sigma_q = ((log_sigma - ls_min) / ls_range * 255.0).round().clamp(0, 255).to(torch.uint8)

    return mu_q, sigma_q


# ---------------------------------------------------------------------------
# Hash key construction & multi-head lookup
# ---------------------------------------------------------------------------


def build_hash_keys(
    layer_idx: int,
    n_rows: int,
    n_groups: int,
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
) -> torch.Tensor:
    """Construct a single integer hash key per group.

    key = layer_idx * 1_000_003
        + row_idx   * 1_009
        + group_id  * 7
        + int(mu_q) * 131
        + int(sigma_q) * 31

    Args:
        layer_idx: layer index (scalar).
        n_rows: M (number of rows).
        n_groups: G (groups per row).
        mu_q:    (M, G) int8.
        sigma_q: (M, G) uint8.

    Returns:
        keys: (M, G) int64.
    """
    row_ids = torch.arange(n_rows, device=mu_q.device).unsqueeze(1)       # (M, 1)
    group_ids = torch.arange(n_groups, device=mu_q.device).unsqueeze(0)   # (1, G)

    keys = (
        layer_idx * 1_000_003
        + row_ids * 1_009
        + group_ids * 7
        + mu_q.long() * 131
        + sigma_q.long() * 31
    )
    return keys


def multi_head_lookup(
    keys: torch.Tensor,
    table: torch.Tensor,
    num_heads: int = DEFAULT_NUM_HEADS,
) -> torch.Tensor:
    """Compute multi-head hash indices and average the retrieved vectors.

    h_j(key) = (key * prime_j) % table_size,  j = 1..num_heads
    v = mean_j T[h_j(key)]

    Args:
        keys:  (M, G) int64 hash keys.
        table: (table_size, group_size) float — the shared hash table.
        num_heads: number of independent hash functions.

    Returns:
        vectors: (M, G, group_size) float — reconstructed group vectors.
    """
    table_size = table.shape[0]
    M, G = keys.shape
    group_dim = table.shape[1]

    accumulated = torch.zeros(M, G, group_dim, device=table.device, dtype=table.dtype)

    for j in range(num_heads):
        prime = _HASH_PRIMES[j]
        indices = ((keys.long() * prime) % table_size).abs()  # (M, G)
        vectors = table[indices.reshape(-1)]                   # (M*G, group_dim)
        accumulated += vectors.reshape(M, G, group_dim)

    return accumulated / num_heads


# ---------------------------------------------------------------------------
# Hash Table dataclass and training
# ---------------------------------------------------------------------------


@dataclass
class HashTableConfig:
    """Configuration for hash-based weight compression."""

    table_size: int = DEFAULT_TABLE_SIZE
    group_size: int = DEFAULT_GROUP_SIZE
    num_heads: int = DEFAULT_NUM_HEADS
    lr: float = 1e-3
    n_steps: int = 200
    rotation: str = "qr"


class HashWeightTable(nn.Module):
    """Learnable global hash table for weight reconstruction.

    The table is (table_size, group_size) and stores fp16 vectors.
    Training minimises MSE between original rotated-group weights and
    the multi-head hash lookup reconstruction.
    """

    def __init__(
        self,
        table_size: int = DEFAULT_TABLE_SIZE,
        group_size: int = DEFAULT_GROUP_SIZE,
        num_heads: int = DEFAULT_NUM_HEADS,
    ):
        super().__init__()
        self.table_size = table_size
        self.group_size = group_size
        self.num_heads = num_heads

        # Learnable table — initialise with small random values
        self.table = nn.Parameter(
            torch.randn(table_size, group_size) * 0.01
        )

    def lookup(self, keys: torch.Tensor) -> torch.Tensor:
        """Multi-head lookup. keys: (M, G) int64 → (M, G, group_size)."""
        return multi_head_lookup(keys, self.table, self.num_heads)

    def memory_bytes(self) -> int:
        """Storage cost of the table in bytes (fp16)."""
        return self.table_size * self.group_size * 2  # fp16


# ---------------------------------------------------------------------------
# Rotated-domain target computation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _compute_rotated_groups(
    W: torch.Tensor,
    group_size: int,
    seed: int,
    rotation: str = "qr",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-group rotated (normalised) weight vectors and norms.

    This mirrors the TurboQuant pre-processing: row-normalise within each
    group, rotate, and scale by sqrt(d).

    Args:
        W:          (M, N) weight matrix.
        group_size: columns per group.
        seed:       rotation seed.
        rotation:   "qr" or "hadamard".

    Returns:
        rotated: (M, n_groups, group_dim) — rotated+scaled group vectors.
        norms:   (M, n_groups) — per-group row norms.
    """
    W = W.float()
    M, N = W.shape
    n_groups = math.ceil(N / group_size)

    all_rotated = []
    all_norms = []

    for g in range(n_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms

        if rotation == "hadamard":
            Y = hadamard_rotate(W_norm, seed=seed + g_start)
        else:
            Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(W.device)
            Y = W_norm @ Pi.T

        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        # Pad to group_size if last group is smaller
        if g_dim < group_size:
            pad = torch.zeros(M, group_size - g_dim, device=W.device)
            Y_scaled = torch.cat([Y_scaled, pad], dim=1)

        all_rotated.append(Y_scaled)
        all_norms.append(norms.squeeze(1))

    rotated = torch.stack(all_rotated, dim=1)  # (M, n_groups, group_size)
    norms_t = torch.stack(all_norms, dim=1)    # (M, n_groups)
    return rotated, norms_t


# ---------------------------------------------------------------------------
# Reconstruction from hash table
# ---------------------------------------------------------------------------


@torch.no_grad()
def reconstruct_weights(
    table: torch.Tensor,
    keys: torch.Tensor,
    norms: torch.Tensor,
    num_heads: int,
    group_size: int,
    seed: int,
    M: int,
    N: int,
    rotation: str = "qr",
) -> torch.Tensor:
    """Reconstruct full weight matrix from hash table and per-group metadata.

    Args:
        table:      (table_size, group_size) shared hash table.
        keys:       (M, n_groups) int64 hash keys.
        norms:      (M, n_groups) float32 per-group norms.
        num_heads:  number of hash heads.
        group_size: columns per group.
        seed:       rotation seed.
        M, N:       original weight shape.
        rotation:   "qr" or "hadamard".

    Returns:
        W_recon: (M, N) float32 reconstructed weight matrix.
    """
    n_groups = keys.shape[1]
    vectors = multi_head_lookup(keys, table, num_heads)  # (M, n_groups, group_size)

    W_recon = torch.zeros(M, N, device=table.device)

    for g in range(n_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start

        Y_scaled = vectors[:, g, :g_dim]  # (M, g_dim)
        scale = math.sqrt(g_dim)
        Y_unscaled = Y_scaled / scale

        if rotation == "hadamard":
            W_g = hadamard_rotate_inverse(Y_unscaled, seed=seed + g_start)
        else:
            Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(table.device)
            W_g = Y_unscaled @ Pi

        W_recon[:, g_start:g_end] = W_g * norms[:, g:g + 1]

    return W_recon


# ---------------------------------------------------------------------------
# Train the hash table for a single layer
# ---------------------------------------------------------------------------


def train_hash_table(
    W: torch.Tensor,
    layer_idx: int,
    config: HashTableConfig,
    seed: int = 42,
    table: Optional[HashWeightTable] = None,
) -> tuple[HashWeightTable, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train (or fine-tune) the hash table to reconstruct a single layer's weights.

    The loss is MSE between rotated group vectors and multi-head hash
    reconstruction.  Only the table entries that are accessed receive
    gradient updates (sparse).

    Args:
        W:         (M, N) weight matrix (float).
        layer_idx: integer layer index.
        config:    HashTableConfig.
        seed:      rotation seed.
        table:     optional pre-existing HashWeightTable to fine-tune.

    Returns:
        table:   trained HashWeightTable.
        keys:    (M, n_groups) int64 hash keys.
        mu_q:    (M, n_groups) int8 quantized means.
        sigma_q: (M, n_groups) uint8 quantized stds.
        norms:   (M, n_groups) float32 group norms.
    """
    W = W.float()
    M, N = W.shape
    device = W.device

    # 1. Compute pre-rotation statistics
    means, stds = compute_group_stats(W, group_size=config.group_size)
    mu_q, sigma_q = quantize_stats(means, stds)

    # 2. Compute rotated-domain targets
    rotated_targets, norms = _compute_rotated_groups(
        W, group_size=config.group_size, seed=seed, rotation=config.rotation,
    )

    # 3. Build hash keys
    n_groups = mu_q.shape[1]
    keys = build_hash_keys(layer_idx, M, n_groups, mu_q, sigma_q)

    # 4. Create or reuse table
    if table is None:
        table = HashWeightTable(
            table_size=config.table_size,
            group_size=config.group_size,
            num_heads=config.num_heads,
        ).to(device)

    # 5. Train via MSE
    optimizer = torch.optim.Adam(table.parameters(), lr=config.lr)
    keys_dev = keys.to(device)
    targets_dev = rotated_targets.to(device)

    for step in range(config.n_steps):
        optimizer.zero_grad()
        reconstructed = table.lookup(keys_dev)  # (M, n_groups, group_size)
        loss = nn.functional.mse_loss(reconstructed, targets_dev)
        loss.backward()
        optimizer.step()

    return table, keys.cpu(), mu_q.cpu(), sigma_q.cpu(), norms.cpu()


# ---------------------------------------------------------------------------
# Packed representation for storage
# ---------------------------------------------------------------------------


def hash_compress(
    W: torch.Tensor,
    layer_idx: int,
    config: HashTableConfig,
    seed: int = 42,
    table: Optional[HashWeightTable] = None,
) -> dict:
    """Compress a weight matrix using hash-based encoding.

    Returns a packed dict containing all metadata needed to reconstruct
    the weights (excluding the shared global table which is stored once).

    Args:
        W:         (M, N) weight matrix.
        layer_idx: integer layer index.
        config:    HashTableConfig.
        seed:      rotation seed.
        table:     optional pre-existing table to fine-tune.

    Returns:
        dict with:
            table:     HashWeightTable (shared, to be saved once).
            keys:      (M, n_groups) int64 hash keys.
            mu_q:      (M, n_groups) int8 quantized means.
            sigma_q:   (M, n_groups) uint8 quantized stds.
            norms:     (M, n_groups) float32 group norms.
            shape:     (M, N).
            seed:      rotation seed.
            layer_idx: int.
            group_size: int.
            num_heads: int.
            table_size: int.
            rotation:  str.
    """
    table, keys, mu_q, sigma_q, norms = train_hash_table(
        W, layer_idx, config, seed, table,
    )

    return {
        "table": table,
        "keys": keys,
        "mu_q": mu_q,
        "sigma_q": sigma_q,
        "norms": norms,
        "shape": tuple(W.shape),
        "seed": seed,
        "layer_idx": layer_idx,
        "group_size": config.group_size,
        "num_heads": config.num_heads,
        "table_size": config.table_size,
        "rotation": config.rotation,
    }


@torch.no_grad()
def hash_decompress(packed: dict, device: Optional[torch.device] = None) -> torch.Tensor:
    """Reconstruct weight matrix from hash-compressed packed data.

    Args:
        packed: dict returned by ``hash_compress``.
        device: target device (default: CPU).

    Returns:
        W_recon: (M, N) float32 reconstructed weight matrix.
    """
    device = device or torch.device("cpu")
    table_obj: HashWeightTable = packed["table"]
    table_data = table_obj.table.data.to(device)
    keys = packed["keys"].to(device)
    norms = packed["norms"].to(device)
    M, N = packed["shape"]

    return reconstruct_weights(
        table=table_data,
        keys=keys,
        norms=norms,
        num_heads=packed["num_heads"],
        group_size=packed["group_size"],
        seed=packed["seed"],
        M=M,
        N=N,
        rotation=packed["rotation"],
    )


# ---------------------------------------------------------------------------
# BPW calculation
# ---------------------------------------------------------------------------


def compute_bpw(
    total_weights: int,
    table_size: int = DEFAULT_TABLE_SIZE,
    group_size: int = DEFAULT_GROUP_SIZE,
    stats_bits_per_group: int = 16,
) -> float:
    """Compute effective bits per weight for hash-based compression.

    Components:
      - Global table: table_size × group_size × 16 bits (fp16).
      - Per-group stats: 8 (mean) + 8 (std) = 16 bits per group.
      - Per-group norms: 32 bits (fp32) per group.

    The table is shared across all layers, so it is amortised over
    total_weights.

    Args:
        total_weights: total number of weight elements in the model.
        table_size:    number of entries in the hash table.
        group_size:    weights per group.
        stats_bits_per_group: bits for quantized stats (default 16).

    Returns:
        Effective bits per weight (float).
    """
    # Table cost (shared, amortised)
    table_bits = table_size * group_size * 16  # fp16 table entries

    # Per-group overhead
    n_groups = math.ceil(total_weights / group_size)
    stats_bits = n_groups * stats_bits_per_group  # μ_q + σ_q
    norm_bits = n_groups * 32                      # fp32 norms

    total_bits = table_bits + stats_bits + norm_bits
    return total_bits / total_weights
