"""Recursive polar decomposition for TurboQuant (PolarQuant-style).

Converts a rotated weight vector y ∈ R^d (d = 2^L) into:
  - One scalar final radius r = ||y||_2
  - (d-1) angles ψ^(ℓ) across L = log2(d) levels

The decomposition works like a binary tree of 2-D polar conversions:
  Level 1: pair original coordinates → d/2 angles + d/2 local radii
  Level 2: pair level-1 radii       → d/4 angles + d/4 local radii
  ...
  Level L: single pair              → 1 angle   + 1 final radius

Total angles: d/2 + d/4 + ... + 1 = d - 1.

Angle PDFs per level (for Lloyd-Max codebook design):
  Level 1 (finest): atan2 of two iid N(0,σ²) → Uniform[-π, π]
  Level ℓ ≥ 2:      atan2 of two iid chi(k) with k = 2^(ℓ-1)
                     → p(θ) ∝ (sin θ cos θ)^(k-1) on [0, π/2]
                     Concentrates around π/4 as k grows.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Recursive polar decomposition (forward / inverse)
# ---------------------------------------------------------------------------


def recursive_polar_decompose(
    y: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Decompose vectors into a binary tree of 2-D polar coordinates.

    Args:
        y: (..., d) tensor, d must be a power of 2.

    Returns:
        final_radius: (...,) — the L2 norm of each vector.
        angles: list of L tensors:
            angles[0]: (..., d//2)   finest level
            angles[L-1]: (..., 1)    coarsest level
    """
    d = y.shape[-1]
    L = int(math.log2(d))
    assert 2**L == d, f"d must be power of 2, got {d}"

    angles: list[torch.Tensor] = []
    current = y

    for _ in range(L):
        a = current[..., 0::2]
        b = current[..., 1::2]
        psi = torch.atan2(b, a)
        r_local = torch.sqrt(a**2 + b**2)
        angles.append(psi)
        current = r_local

    final_radius = current.squeeze(-1)
    return final_radius, angles


def recursive_polar_reconstruct(
    final_radius: torch.Tensor,
    angles: list[torch.Tensor],
) -> torch.Tensor:
    """Reconstruct Cartesian coordinates from polar decomposition.

    Args:
        final_radius: (...,) tensor.
        angles: list of L tensors (finest to coarsest).

    Returns:
        y: (..., d) reconstructed tensor.
    """
    L = len(angles)
    current = final_radius.unsqueeze(-1)  # (..., 1)

    for level in range(L - 1, -1, -1):
        psi = angles[level]
        a = current * torch.cos(psi)
        b = current * torch.sin(psi)
        current = torch.stack([a, b], dim=-1).reshape(
            *a.shape[:-1], a.shape[-1] * 2
        )

    return current


# ---------------------------------------------------------------------------
# Angle PDF computation (for Lloyd-Max codebook design)
# ---------------------------------------------------------------------------


def angle_pdf_level1(theta: np.ndarray) -> np.ndarray:
    """PDF of atan2(b, a) where a, b ~ iid N(0, σ²).

    Result: Uniform on [-π, π].
    """
    return np.ones_like(theta) / (2 * np.pi)


def angle_pdf_higher(theta: np.ndarray, k: int) -> np.ndarray:
    """PDF of atan2(b, a) where a, b ~ iid chi(k), θ ∈ [0, π/2].

    p(θ) ∝ (sin θ cos θ)^(k-1)  =  (sin 2θ / 2)^(k-1)

    This is a Beta(k/2, k/2) distribution under the transform t = sin²θ.
    """
    # Numerically stable: work in log space
    log_p = (k - 1) * (np.log(np.sin(theta) + 1e-30) + np.log(np.cos(theta) + 1e-30))
    log_p -= log_p.max()
    p = np.exp(log_p)
    # Normalize
    dtheta = theta[1] - theta[0] if len(theta) > 1 else 1.0
    p /= np.sum(p) * dtheta
    return p


# ---------------------------------------------------------------------------
# Lloyd-Max quantizer for arbitrary PDF (numerical)
# ---------------------------------------------------------------------------


def lloyd_max_arbitrary(
    pdf_func,
    lo: float,
    hi: float,
    n_levels: int,
    n_iters: int = 300,
    n_grid: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Lloyd-Max optimal quantizer for an arbitrary PDF via numerical integration.

    Args:
        pdf_func: callable θ → p(θ), must integrate to ~1 on [lo, hi].
        lo, hi: support of the PDF.
        n_levels: number of quantization levels (2^b).
        n_iters: Lloyd-Max iterations.
        n_grid: grid resolution for numerical integration.

    Returns:
        centroids: (n_levels,) optimal centroids.
        boundaries: (n_levels + 1,) quantization boundaries.
    """
    grid = np.linspace(lo, hi, n_grid)
    dx = grid[1] - grid[0]
    pdf_vals = pdf_func(grid)
    cdf_vals = np.cumsum(pdf_vals) * dx

    # Initialize boundaries with equal-probability partitioning
    boundaries = np.zeros(n_levels + 1)
    boundaries[0] = lo
    boundaries[-1] = hi
    for i in range(1, n_levels):
        target = i / n_levels
        idx = np.searchsorted(cdf_vals, target)
        boundaries[i] = grid[min(idx, len(grid) - 1)]

    centroids = np.zeros(n_levels)

    for _ in range(n_iters):
        # Update centroids: E[θ | θ ∈ bin_i]
        for i in range(n_levels):
            mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
            if i == n_levels - 1:
                mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            p_slice = pdf_vals[mask]
            x_slice = grid[mask]
            total_p = np.sum(p_slice) * dx
            if total_p > 1e-15:
                centroids[i] = np.sum(x_slice * p_slice) * dx / total_p
            else:
                centroids[i] = (boundaries[i] + boundaries[i + 1]) / 2

        # Update boundaries: midpoints of adjacent centroids
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

    return centroids, boundaries


# ---------------------------------------------------------------------------
# Precompute angle codebooks for all levels
# ---------------------------------------------------------------------------


_ANGLE_CODEBOOK_CACHE: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def get_angle_codebook(
    level: int, bit_width: int, d: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get Lloyd-Max codebook for angle quantization at a given level.

    Args:
        level: decomposition level (0 = finest, L-1 = coarsest).
        bit_width: bits per angle at this level.
        d: original vector dimension.

    Returns:
        centroids: (2^bit_width,) float32
        boundaries: (2^bit_width - 1,) float32 (inner boundaries)
    """
    L = int(math.log2(d))
    cache_key = (level, bit_width, d)

    if cache_key not in _ANGLE_CODEBOOK_CACHE:
        n_levels = 2**bit_width

        if level == 0:
            # Level 1 (finest): uniform on [-π, π]
            centroids, boundaries = lloyd_max_arbitrary(
                angle_pdf_level1, -math.pi, math.pi, n_levels
            )
        else:
            # Higher levels: concentrated around π/4
            k = 2**level  # chi degrees of freedom
            pdf_func = lambda theta, _k=k: angle_pdf_higher(theta, _k)
            centroids, boundaries = lloyd_max_arbitrary(
                pdf_func, 0.0, math.pi / 2, n_levels
            )

        _ANGLE_CODEBOOK_CACHE[cache_key] = (
            torch.tensor(centroids, dtype=torch.float32),
            torch.tensor(boundaries[1:-1], dtype=torch.float32),
        )

    return _ANGLE_CODEBOOK_CACHE[cache_key]


# ---------------------------------------------------------------------------
# Quantize / dequantize angles
# ---------------------------------------------------------------------------


def quantize_angles(
    angles: list[torch.Tensor],
    bit_alloc: list[int],
    d: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Quantize angles at each level using level-specific Lloyd-Max codebooks.

    Args:
        angles: list of L tensors from recursive_polar_decompose.
        bit_alloc: list of L bit-widths, one per level.
        d: original vector dimension.

    Returns:
        indices: list of L int tensors (quantization indices per level).
        q_angles: list of L float tensors (quantized angle values).
    """
    L = len(angles)
    assert len(bit_alloc) == L

    indices_list: list[torch.Tensor] = []
    q_angles_list: list[torch.Tensor] = []

    for level in range(L):
        psi = angles[level]
        bits = bit_alloc[level]
        centroids, boundaries = get_angle_codebook(level, bits, d)
        centroids = centroids.to(psi.device)
        boundaries = boundaries.to(psi.device)

        # Quantize via searchsorted
        flat = psi.reshape(-1)
        idx = torch.searchsorted(boundaries, flat)
        idx = idx.clamp(0, len(centroids) - 1)
        q_flat = centroids[idx]

        indices_list.append(idx.reshape(psi.shape))
        q_angles_list.append(q_flat.reshape(psi.shape))

    return indices_list, q_angles_list


def compute_bpw(bit_alloc: list[int], d: int, radius_bits: int = 16) -> float:
    """Compute effective bits-per-weight for a given bit allocation.

    Args:
        bit_alloc: list of L bit-widths per level.
        d: vector dimension.
        radius_bits: bits for the final radius (default 16 = fp16).

    Returns:
        bpw: float bits per original coordinate.
    """
    L = len(bit_alloc)
    total_bits = 0
    for level in range(L):
        n_angles = d // (2 ** (level + 1))
        total_bits += n_angles * bit_alloc[level]
    total_bits += radius_bits  # one radius per vector
    return total_bits / d


# ---------------------------------------------------------------------------
# Full polar quantization pipeline
# ---------------------------------------------------------------------------


@torch.no_grad()
def polar_quantize(
    W: torch.Tensor,
    bit_alloc: list[int],
    group_size: int = 128,
    seed: int = 42,
    rotation: str = "hadamard",
    radius_bits: int = 16,
) -> tuple[torch.Tensor, dict]:
    """Full polar quantization pipeline: rotate → decompose → quantize → reconstruct.

    Args:
        W: (M, N) weight matrix.
        bit_alloc: list of L bit-widths per level (L = log2(group_size)).
        group_size: must be power of 2.
        seed: rotation seed.
        rotation: "hadamard" or "qr".
        radius_bits: bits for final radius storage.

    Returns:
        W_approx: (M, N) dequantized approximation.
        info: dict with diagnostics (bpw, per-level errors, etc.).
    """
    from turboquant_model.rotation import (
        generate_rotation_matrix,
        hadamard_rotate,
        hadamard_rotate_inverse,
    )

    orig_dtype = W.dtype
    W = W.float()
    M, N = W.shape
    L = int(math.log2(group_size))
    assert len(bit_alloc) == L, f"bit_alloc length {len(bit_alloc)} != L={L}"

    W_approx = torch.zeros_like(W)
    bpw = compute_bpw(bit_alloc, group_size, radius_bits)

    level_mses = [0.0] * L
    n_groups_total = 0

    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        if g_dim < group_size:
            # Pad last group if needed
            W_g = torch.zeros(M, group_size, device=W.device)
            W_g[:, :g_dim] = W[:, g_start:g_end]
        else:
            W_g = W[:, g_start:g_end]

        # Step 1: Rotate
        if rotation == "hadamard":
            Y = hadamard_rotate(W_g, seed=seed + g_start)
        else:
            Pi = generate_rotation_matrix(group_size, seed=seed + g_start).to(W.device)
            Y = W_g @ Pi.T

        # Step 2: Recursive polar decomposition
        final_radius, angles = recursive_polar_decompose(Y)

        # Step 3: Quantize angles
        _, q_angles = quantize_angles(angles, bit_alloc, group_size)

        # Track per-level angle MSE
        for lv in range(L):
            level_mses[lv] += (angles[lv] - q_angles[lv]).pow(2).mean().item()
        n_groups_total += 1

        # Step 4: Quantize radius (simulate fp16 or int8)
        if radius_bits == 16:
            q_radius = final_radius.half().float()
        elif radius_bits == 8:
            # Simple min-max int8
            rmin = final_radius.min()
            rmax = final_radius.max()
            scale_r = (rmax - rmin) / 255.0
            q_radius = (
                ((final_radius - rmin) / scale_r).round() * scale_r + rmin
            )
        else:
            q_radius = final_radius  # lossless

        # Step 5: Reconstruct from quantized angles + radius
        Y_recon = recursive_polar_reconstruct(q_radius, q_angles)

        # Step 6: Inverse rotation
        if rotation == "hadamard":
            W_g_approx = hadamard_rotate_inverse(Y_recon, seed=seed + g_start)
        else:
            W_g_approx = Y_recon @ Pi

        W_approx[:, g_start:g_end] = W_g_approx[:, :g_dim]

    level_mses = [m / max(n_groups_total, 1) for m in level_mses]

    info = {
        "bpw": bpw,
        "bit_alloc": bit_alloc,
        "group_size": group_size,
        "radius_bits": radius_bits,
        "level_angle_mse": level_mses,
    }

    return W_approx.to(orig_dtype), info


# ---------------------------------------------------------------------------
# Optimal bit allocation via rate-distortion
# ---------------------------------------------------------------------------


def angle_distortion_at_bits(level: int, bits: int, d: int) -> float:
    """Estimate quantization distortion (MSE) for a given level and bit-width.

    Uses the precomputed Lloyd-Max codebook and the known angle PDF.
    """
    if bits == 0:
        # No quantization: use the mean angle as the single "centroid"
        if level == 0:
            return (math.pi**2) / 3  # Var(Uniform[-π,π])
        else:
            # Variance of the concentrated angle distribution
            k = 2**level
            # Approximate variance: 1 / (4k - 2) for Beta(k/2, k/2) mapped to [0, π/2]
            return (math.pi / 2) ** 2 / (4 * k - 2)

    centroids, boundaries = get_angle_codebook(level, bits, d)
    centroids_np = centroids.numpy()
    boundaries_full = np.concatenate([
        [-math.pi if level == 0 else 0.0],
        boundaries.numpy(),
        [math.pi if level == 0 else math.pi / 2],
    ])

    n_grid = 5000
    if level == 0:
        grid = np.linspace(-math.pi, math.pi, n_grid)
        pdf_vals = angle_pdf_level1(grid)
    else:
        grid = np.linspace(1e-6, math.pi / 2 - 1e-6, n_grid)
        k = 2**level
        pdf_vals = angle_pdf_higher(grid, k)

    dx = grid[1] - grid[0]
    total_mse = 0.0
    for i in range(len(centroids_np)):
        lo_b = boundaries_full[i]
        hi_b = boundaries_full[i + 1]
        mask = (grid >= lo_b) & (grid < hi_b)
        if i == len(centroids_np) - 1:
            mask = (grid >= lo_b) & (grid <= hi_b)
        total_mse += np.sum((grid[mask] - centroids_np[i]) ** 2 * pdf_vals[mask]) * dx

    return float(total_mse)


def partial_polar_decompose(
    y: torch.Tensor,
    n_levels: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Decompose only the first *n_levels* of the polar binary tree.

    After n_levels the remaining d / 2^n_levels local radii are returned
    as a 'residual Cartesian' tensor instead of being further decomposed.

    Args:
        y: (..., d) tensor, d must be a power of 2.
        n_levels: how many fine→coarse polar levels to perform (0 → identity).

    Returns:
        local_radii: (..., d // 2^n_levels) — intermediate radii.
        angles: list of n_levels tensors (finest first).
    """
    d = y.shape[-1]
    assert n_levels <= int(math.log2(d))

    angles: list[torch.Tensor] = []
    current = y

    for _ in range(n_levels):
        a = current[..., 0::2]
        b = current[..., 1::2]
        psi = torch.atan2(b, a)
        r_local = torch.sqrt(a**2 + b**2)
        angles.append(psi)
        current = r_local

    return current, angles


def partial_polar_reconstruct(
    local_radii: torch.Tensor,
    angles: list[torch.Tensor],
) -> torch.Tensor:
    """Reconstruct from a partial polar decomposition.

    Inverse of :func:`partial_polar_decompose`.
    """
    current = local_radii

    for level in range(len(angles) - 1, -1, -1):
        psi = angles[level]
        a = current * torch.cos(psi)
        b = current * torch.sin(psi)
        current = torch.stack([a, b], dim=-1).reshape(
            *a.shape[:-1], a.shape[-1] * 2
        )

    return current


@torch.no_grad()
def hybrid_polar_cartesian_quantize(
    W: torch.Tensor,
    n_polar_levels: int,
    angle_bits: list[int],
    cartesian_bits: int,
    group_size: int = 128,
    seed: int = 42,
    radius_bits: int = 16,
) -> tuple[torch.Tensor, dict]:
    """Hybrid polar + Cartesian quantization.

    Applies *n_polar_levels* of polar decomposition (fine → coarse), then
    quantizes the intermediate radii with standard Cartesian scalar
    quantization (Lloyd-Max for Gaussian).

    This avoids error amplification at coarse polar levels while exploiting
    the geometric angle structure at fine levels.

    Args:
        W: (M, N) weight matrix.
        n_polar_levels: number of polar levels (0 = pure Cartesian).
        angle_bits: list of n_polar_levels bit-widths for angle quantization.
        cartesian_bits: bit-width for Cartesian quantization of radii.
        group_size: must be power of 2.
        seed: rotation seed.
        radius_bits: bits for the final radius (fp16 by default).

    Returns:
        W_approx: (M, N) dequantized approximation.
        info: diagnostics dict.
    """
    from turboquant_model.codebook import get_codebook
    from turboquant_model.rotation import hadamard_rotate, hadamard_rotate_inverse

    assert len(angle_bits) == n_polar_levels

    orig_dtype = W.dtype
    W = W.float()
    M, N = W.shape
    L = int(math.log2(group_size))
    d_radii = group_size // (2**n_polar_levels)  # dim of intermediate radii

    # Cartesian codebook
    if cartesian_bits > 0:
        c_centroids, c_boundaries = get_codebook(cartesian_bits)
    else:
        c_centroids = c_boundaries = None

    W_approx = torch.zeros_like(W)
    angle_mses = [0.0] * n_polar_levels
    cart_mse_total = 0.0
    n_groups = 0

    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        if g_dim < group_size:
            W_g = torch.zeros(M, group_size, device=W.device)
            W_g[:, :g_dim] = W[:, g_start:g_end]
        else:
            W_g = W[:, g_start:g_end]

        # 1) Rotate
        Y = hadamard_rotate(W_g, seed=seed + g_start)

        # 2) Partial polar decomposition
        local_radii, angles = partial_polar_decompose(Y, n_polar_levels)

        # 3) Quantize angles (polar part)
        if n_polar_levels > 0:
            _, q_angles = quantize_angles(angles, angle_bits, group_size)
            for lv in range(n_polar_levels):
                angle_mses[lv] += (angles[lv] - q_angles[lv]).pow(2).mean().item()
        else:
            q_angles = []

        # 4) Quantize intermediate radii (Cartesian part)
        #    Normalize per-row, scalar-quantize, rescale
        radii_norms = local_radii.norm(dim=1, keepdim=True).clamp(min=1e-8)
        radii_normed = local_radii / radii_norms

        if cartesian_bits > 0 and c_centroids is not None:
            scale = math.sqrt(d_radii)
            r_scaled = radii_normed * scale
            dev = r_scaled.device
            idx = torch.searchsorted(
                c_boundaries.to(dev), r_scaled.reshape(-1)
            )
            idx = idx.clamp(0, len(c_centroids) - 1)
            r_q = c_centroids.to(dev)[idx].reshape(r_scaled.shape) / scale
        else:
            r_q = radii_normed  # lossless

        cart_mse_total += (radii_normed - r_q).pow(2).mean().item()

        # Quantize radii norms (fp16 or int8)
        if radius_bits == 16:
            q_norms = radii_norms.half().float()
        else:
            q_norms = radii_norms

        q_radii = r_q * q_norms

        # 5) Reconstruct polar → Cartesian
        Y_recon = partial_polar_reconstruct(q_radii, q_angles)

        # 6) Inverse rotation
        W_g_approx = hadamard_rotate_inverse(Y_recon, seed=seed + g_start)
        W_approx[:, g_start:g_end] = W_g_approx[:, :g_dim]
        n_groups += 1

    angle_mses = [m / max(n_groups, 1) for m in angle_mses]
    cart_mse_avg = cart_mse_total / max(n_groups, 1)

    # BPW calculation
    total_bits_per_group = 0
    for lv in range(n_polar_levels):
        n_angles = group_size // (2 ** (lv + 1))
        total_bits_per_group += n_angles * angle_bits[lv]
    total_bits_per_group += d_radii * cartesian_bits  # Cartesian radii
    total_bits_per_group += radius_bits  # one norm per row per group
    bpw = total_bits_per_group / group_size

    info = {
        "bpw": bpw,
        "n_polar_levels": n_polar_levels,
        "angle_bits": angle_bits,
        "cartesian_bits": cartesian_bits,
        "group_size": group_size,
        "d_radii": d_radii,
        "angle_mses": angle_mses,
        "cartesian_mse": cart_mse_avg,
    }
    return W_approx.to(orig_dtype), info


def optimize_bit_allocation(
    d: int,
    target_bpw: float,
    max_bits_per_level: int = 6,
    radius_bits: int = 16,
) -> list[int]:
    """Find the bit allocation that minimizes total angle distortion
    subject to a bpw budget constraint.

    Uses greedy marginal-gain allocation:
      1. Start with 1 bit per level.
      2. At each step, add 1 bit to the level with the greatest distortion reduction.
      3. Stop when the bpw budget is exhausted.

    Args:
        d: vector dimension (power of 2).
        target_bpw: target bits-per-weight.
        max_bits_per_level: cap on bits for any single level.
        radius_bits: bits for the final radius.

    Returns:
        bit_alloc: list of L optimal bit-widths.
    """
    L = int(math.log2(d))
    bit_alloc = [1] * L

    # Available bit budget (total angle bits)
    total_budget = int(target_bpw * d) - radius_bits
    current_bits = sum(d // (2 ** (lv + 1)) * bit_alloc[lv] for lv in range(L))

    # Precompute distortions at all levels and bit-widths
    distortions: dict[tuple[int, int], float] = {}
    for lv in range(L):
        for b in range(0, max_bits_per_level + 1):
            distortions[(lv, b)] = angle_distortion_at_bits(lv, b, d)

    while current_bits < total_budget:
        best_gain = -1.0
        best_level = -1

        for lv in range(L):
            if bit_alloc[lv] >= max_bits_per_level:
                continue
            n_angles = d // (2 ** (lv + 1))
            cost = n_angles  # adding 1 bit costs n_angles total bits
            if current_bits + cost > total_budget:
                continue
            old_d = distortions[(lv, bit_alloc[lv])]
            new_d = distortions[(lv, bit_alloc[lv] + 1)]
            gain = (old_d - new_d) * n_angles / cost  # weighted gain per bit
            if gain > best_gain:
                best_gain = gain
                best_level = lv

        if best_level < 0:
            break

        n_angles = d // (2 ** (best_level + 1))
        bit_alloc[best_level] += 1
        current_bits += n_angles

    return bit_alloc
