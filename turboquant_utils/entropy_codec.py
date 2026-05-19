"""CUDA-friendly entropy coding for quantized indices via rANS.

Exploits the known Gaussian bin probability distribution from Lloyd-Max
quantization for near-entropy-optimal compression.

The codec is designed for GPU-parallel decoding:
- Block-based encoding: each block of B symbols is independently decodable
- Precomputed decode tables fit in GPU shared memory (~4KB per bit-width)
- Decoding per symbol: 1 table lookup + 1 multiply + 1 shift + renormalize

Reference: Duda (2009), "Asymmetric numeral systems"
           Giesen (2014), "Interleaved entropy coders"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Gaussian bin probabilities
# ---------------------------------------------------------------------------


def gaussian_bin_probs(bit_width: int) -> np.ndarray:
    """Compute bin probabilities for Lloyd-Max quantization of N(0,1).

    Returns:
        probs: (2^bit_width,) array of bin probabilities
    """
    from turboquant_model.codebook import get_codebook
    from scipy.stats import norm

    _, boundaries = get_codebook(bit_width)
    boundaries = boundaries.numpy()
    full_bounds = np.concatenate([[-np.inf], boundaries, [np.inf]])
    probs = np.diff(norm.cdf(full_bounds))
    return probs


def compute_entropy(bit_width: int) -> float:
    """Theoretical entropy (bits per symbol) for a given bit-width.

    This is the lower bound on achievable BPW for index storage.
    """
    probs = gaussian_bin_probs(bit_width)
    H = -np.sum(probs * np.log2(probs + 1e-30))
    return float(H)


# ---------------------------------------------------------------------------
# rANS codec
# ---------------------------------------------------------------------------

# State parameters
PROB_BITS = 14          # probability precision
PROB_SCALE = 1 << PROB_BITS
STATE_LOWER = 1 << 16   # state lower bound
STATE_UPPER = 1 << 24   # state upper bound (STATE_LOWER << 8)
BLOCK_SIZE = 4096        # symbols per independently-decodable block


@dataclass
class ANSTable:
    """Precomputed frequency tables for rANS coding.

    These tables are the "CUDA-friendly entropy coding codebook":
    - Total size: n_symbols * 8 bytes (freqs + cumuls) ≈ 128 bytes for 4-bit
    - Fits easily in GPU shared memory / registers
    - Decoding: O(1) per symbol with cumulative frequency lookup
    """
    freqs: np.ndarray     # (n_symbols,) uint16 — quantized frequencies (sum = PROB_SCALE)
    cumuls: np.ndarray    # (n_symbols+1,) uint32 — cumulative frequencies
    n_symbols: int
    probs: np.ndarray     # (n_symbols,) float64 — exact probabilities
    entropy: float        # theoretical entropy (bits per symbol)
    bit_width: int

    def table_size_bytes(self) -> int:
        """Size of the decode table in bytes (for GPU shared memory budget)."""
        return self.n_symbols * 4 + (self.n_symbols + 1) * 4  # freqs + cumuls


def build_ans_table(bit_width: int) -> ANSTable:
    """Build rANS frequency tables for a given bit-width.

    Args:
        bit_width: quantization bit-width (2-5)

    Returns:
        ANSTable with precomputed frequency/cumulative tables
    """
    probs = gaussian_bin_probs(bit_width)
    n_symbols = len(probs)

    # Quantize probabilities to integers summing to PROB_SCALE
    freqs = np.maximum(1, np.round(probs * PROB_SCALE).astype(np.int64))

    # Adjust to sum exactly to PROB_SCALE
    while freqs.sum() > PROB_SCALE:
        # Reduce the symbol that is most over-represented
        err = freqs / probs.clip(min=1e-30) - PROB_SCALE
        idx = int(np.argmax(err))
        if freqs[idx] > 1:
            freqs[idx] -= 1
    while freqs.sum() < PROB_SCALE:
        err = probs * PROB_SCALE - freqs
        idx = int(np.argmax(err))
        freqs[idx] += 1

    assert freqs.sum() == PROB_SCALE, f"Frequency sum {freqs.sum()} != {PROB_SCALE}"

    # Build cumulative frequency table
    cumuls = np.zeros(n_symbols + 1, dtype=np.int64)
    cumuls[1:] = np.cumsum(freqs)

    entropy = float(-np.sum(probs * np.log2(probs + 1e-30)))

    return ANSTable(
        freqs=freqs.astype(np.uint16),
        cumuls=cumuls.astype(np.uint32),
        n_symbols=n_symbols,
        probs=probs,
        entropy=entropy,
        bit_width=bit_width,
    )


# ---------------------------------------------------------------------------
# rANS encoder/decoder
# ---------------------------------------------------------------------------


class rANSCodec:
    """Range ANS encoder/decoder with block-parallel support.

    Encoding is sequential (per block). Decoding can be parallelized
    across blocks on GPU — each block starts from a known state.

    Block layout:
        [4 bytes: final state] [variable: emitted bytes, reversed]

    GPU decoding algorithm per block (pseudocode):
        state = load_u32(block_start)
        for i in 0..BLOCK_SIZE:
            slot = state & (PROB_SCALE - 1)
            symbol = find_symbol(cumuls, slot)  // binary search or LUT
            state = freqs[symbol] * (state >> PROB_BITS) + slot - cumuls[symbol]
            while state < STATE_LOWER:
                state = (state << 8) | load_byte(byte_ptr--)
            output[i] = symbol
    """

    def __init__(self, table: ANSTable):
        self.table = table
        self.freqs = table.freqs.astype(np.int64)
        self.cumuls = table.cumuls.astype(np.int64)

        # Build reverse lookup table for fast symbol finding during decode
        self._build_decode_lut()

    def _build_decode_lut(self):
        """Build a lookup table: slot → symbol for O(1) decode."""
        self.decode_lut = np.zeros(PROB_SCALE, dtype=np.uint8)
        for s in range(self.table.n_symbols):
            c_lo = self.cumuls[s]
            c_hi = self.cumuls[s + 1]
            self.decode_lut[c_lo:c_hi] = s

    def encode(self, symbols: np.ndarray) -> bytes:
        """Encode a flat array of symbols into compressed bytes.

        Symbols are split into blocks of BLOCK_SIZE for parallel decoding.

        Args:
            symbols: (N,) uint8 array of symbol indices

        Returns:
            Compressed bytes with block headers
        """
        n = len(symbols)
        n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

        all_bytes = []

        # Encode header: n_blocks (4 bytes) + n_symbols_total (4 bytes)
        all_bytes.extend(n_blocks.to_bytes(4, 'little'))
        all_bytes.extend(n.to_bytes(4, 'little'))

        for b in range(n_blocks):
            start = b * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, n)
            block_symbols = symbols[start:end]
            block_data = self._encode_block(block_symbols)
            # Block header: block_data length (4 bytes)
            all_bytes.extend(len(block_data).to_bytes(4, 'little'))
            all_bytes.extend(block_data)

        return bytes(all_bytes)

    def _encode_block(self, symbols: np.ndarray) -> bytes:
        """Encode a single block of symbols."""
        state = STATE_LOWER  # initial state
        output = bytearray()

        # Process symbols in REVERSE order (ANS property)
        for s_idx in range(len(symbols) - 1, -1, -1):
            s = int(symbols[s_idx])
            freq = int(self.freqs[s])

            # Renormalize: emit bytes until state is in valid range
            max_state = freq << (24 - PROB_BITS)  # freq * (STATE_UPPER // PROB_SCALE)
            while state >= max_state:
                output.append(state & 0xFF)
                state >>= 8

            # Encode: state = (state // freq) * PROB_SCALE + (state % freq) + cumul[s]
            state = (state // freq) * PROB_SCALE + (state % freq) + int(self.cumuls[s])

        # Emit final state (4 bytes)
        for _ in range(4):
            output.append(state & 0xFF)
            state >>= 8

        return bytes(output)

    def decode(self, data: bytes) -> np.ndarray:
        """Decode compressed bytes back to symbol array.

        Args:
            data: compressed bytes from encode()

        Returns:
            symbols: (N,) uint8 array
        """
        pos = 0

        # Read header
        n_blocks = int.from_bytes(data[pos:pos+4], 'little')
        pos += 4
        n_total = int.from_bytes(data[pos:pos+4], 'little')
        pos += 4

        all_symbols = []

        for b in range(n_blocks):
            block_len = int.from_bytes(data[pos:pos+4], 'little')
            pos += 4
            block_data = data[pos:pos+block_len]
            pos += block_len

            n_syms = min(BLOCK_SIZE, n_total - b * BLOCK_SIZE)
            block_symbols = self._decode_block(block_data, n_syms)
            all_symbols.extend(block_symbols)

        return np.array(all_symbols[:n_total], dtype=np.uint8)

    def _decode_block(self, data: bytes, n_symbols: int) -> list[int]:
        """Decode a single block."""
        # Read final state from end of block
        byte_pos = len(data) - 1
        state = 0
        for _ in range(4):
            state = (state << 8) | data[byte_pos]
            byte_pos -= 1

        symbols = []
        for _ in range(n_symbols):
            # Find symbol from state
            slot = state & (PROB_SCALE - 1)  # state % PROB_SCALE
            s = int(self.decode_lut[slot])

            freq = int(self.freqs[s])
            cumul = int(self.cumuls[s])

            # Update state
            state = freq * (state >> PROB_BITS) + slot - cumul

            # Renormalize
            while state < STATE_LOWER and byte_pos >= 0:
                state = (state << 8) | data[byte_pos]
                byte_pos -= 1

            symbols.append(s)

        return symbols


# ---------------------------------------------------------------------------
# Cache for tables
# ---------------------------------------------------------------------------

_TABLE_CACHE: dict[int, ANSTable] = {}
_CODEC_CACHE: dict[int, rANSCodec] = {}


def get_ans_table(bit_width: int) -> ANSTable:
    """Get cached ANS table for a given bit-width."""
    if bit_width not in _TABLE_CACHE:
        _TABLE_CACHE[bit_width] = build_ans_table(bit_width)
    return _TABLE_CACHE[bit_width]


def get_codec(bit_width: int) -> rANSCodec:
    """Get cached rANS codec for a given bit-width."""
    if bit_width not in _CODEC_CACHE:
        _CODEC_CACHE[bit_width] = rANSCodec(get_ans_table(bit_width))
    return _CODEC_CACHE[bit_width]


# ---------------------------------------------------------------------------
# High-level compression API
# ---------------------------------------------------------------------------


def compress_indices(
    indices: torch.Tensor,
    bit_width: int,
) -> tuple[bytes, float]:
    """Compress quantized indices using rANS entropy coding.

    Args:
        indices: (...) int tensor of quantization indices
        bit_width: bit-width used for quantization

    Returns:
        compressed: compressed bytes
        bpw: actual bits per weight achieved
    """
    codec = get_codec(bit_width)
    flat = indices.reshape(-1).cpu().numpy().astype(np.uint8)

    compressed = codec.encode(flat)
    bpw = len(compressed) * 8 / len(flat)

    return compressed, bpw


def decompress_indices(
    compressed: bytes,
    bit_width: int,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Decompress rANS-coded indices back to tensor.

    Args:
        compressed: compressed bytes from compress_indices()
        bit_width: bit-width used for quantization
        shape: target tensor shape

    Returns:
        indices: int32 tensor of original shape
    """
    codec = get_codec(bit_width)
    flat = codec.decode(compressed)
    return torch.tensor(flat, dtype=torch.int32).reshape(shape)


def measure_compressed_bpw(
    indices: torch.Tensor,
    bit_width: int,
) -> tuple[float, float]:
    """Measure actual compressed BPW and theoretical entropy.

    Args:
        indices: quantized index tensor
        bit_width: quantization bit-width

    Returns:
        (compressed_bpw, theoretical_entropy)
    """
    table = get_ans_table(bit_width)
    flat = indices.reshape(-1).cpu().numpy().astype(np.uint8)

    # Compute empirical entropy
    counts = np.bincount(flat, minlength=table.n_symbols)
    probs = counts / counts.sum()
    empirical_entropy = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))

    # Actual compressed size
    _, compressed_bpw = compress_indices(indices, bit_width)

    return compressed_bpw, empirical_entropy


def get_cuda_decode_tables(bit_width: int) -> dict[str, torch.Tensor]:
    """Get decode tables as GPU-ready tensors for CUDA kernel integration.

    These tables constitute the "CUDA-friendly entropy coding codebook":
    - decode_lut: (PROB_SCALE,) uint8 — slot → symbol mapping
    - freqs: (n_symbols,) int32 — quantized frequencies
    - cumuls: (n_symbols+1,) int32 — cumulative frequencies

    Total shared memory: PROB_SCALE + n_symbols * 8 bytes
    For 4-bit (n=16): 16384 + 136 = 16520 bytes ≈ 16.1 KB

    For smaller shared memory, use PROB_BITS=10:
    1024 + 136 = 1160 bytes ≈ 1.1 KB
    """
    table = get_ans_table(bit_width)
    codec = get_codec(bit_width)

    return {
        "decode_lut": torch.tensor(codec.decode_lut, dtype=torch.uint8),
        "freqs": torch.tensor(table.freqs, dtype=torch.int32),
        "cumuls": torch.tensor(table.cumuls, dtype=torch.int32),
        "prob_bits": PROB_BITS,
        "state_lower": STATE_LOWER,
        "block_size": BLOCK_SIZE,
    }
