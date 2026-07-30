"""Quantisation + AMX tile packing.

Two weight formats, chosen per-tensor by accuracy sensitivity:

  q8c   int8, symmetric, one f32 scale per output channel.
        Attention projections and anything small where bandwidth is cheap but
        error is expensive.

  q4g   int4, symmetric, one f16 scale per (output channel, group of 64 inputs).
        The FFN bulk and the embedding / LM-head table, where parameters
        dominate DRAM traffic.

Both are stored **pre-packed into AMX tile order**: the runtime mmaps and
executes with no repacking.

Nibble layout
-------------
Canonical everywhere: int8 values are cut into contiguous blocks of 64, and
block element i (i < 32) is stored as

    packed[i] = (q[i] & 0xF) | ((q[i + 32] & 0xF) << 4)

so a single 32-byte AVX512 load yields lanes i and i+32 with one shift, and a
4-bit sign extend is `(x ^ 8) - 8`.

Group geometry
--------------
q4g uses group size 64 == TILE_K. One AMX B-tile (16 rows x 64B) covers exactly
64 K-values for 16 N-values, so every value in a tile shares one K-group and
each of the 16 N-channels contributes one scale. Scales are therefore stored
transposed as [K/64, N] — the 16 scales a tile needs are one contiguous load.

Rotation (`rotconv`)
--------------------
`down_proj` consumes the FFN intermediate, where Gemma's activation outliers
concentrate. We fold a Walsh-Hadamard rotation H into the weight
(W' = W @ H^T along K) and apply the matching H to the activation at runtime.
H is orthogonal so the product is unchanged, but the rotated activation is
near-Gaussian, which is what makes int8 activation quantisation of that tensor
safe. This is local to down_proj — it never touches the residual stream, so it
composes with AltUp/LAuReL with no global analysis.
"""

import numpy as np

TILE_N = 16
TILE_K = 64
GROUP = 64  # must equal TILE_K, see "Group geometry"


# ------------------------------------------------------------------ bf16 -> f32
def bf16_to_f32(raw_u16):
    """safetensors bfloat16 arrives as raw uint16; widen without torch."""
    return (raw_u16.astype(np.uint32) << 16).view(np.float32)


# ------------------------------------------------------------------ Hadamard
def hadamard(n):
    """Normalised Walsh-Hadamard matrix, n a power of two."""
    assert n & (n - 1) == 0, f"{n} is not a power of two"
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def apply_hadamard_k(w, hsize):
    """Rotate W[out, in] along `in` in blocks of hsize: W @ blockdiag(H)^T."""
    out, k = w.shape
    assert k % hsize == 0
    h = hadamard(hsize)
    return (w.reshape(out, k // hsize, hsize) @ h.T).reshape(out, k)


# ------------------------------------------------------------------ nibbles
def pack_nibbles(flat_i8):
    """Canonical 64-block half-split nibble packing (see module docstring)."""
    assert flat_i8.size % 64 == 0, flat_i8.size
    b = flat_i8.reshape(-1, 64)
    lo = (b[:, :32] & 0x0F).astype(np.uint8)
    hi = (b[:, 32:] & 0x0F).astype(np.uint8)
    return (lo | (hi << 4)).reshape(-1)


def unpack_nibbles(packed):
    """Inverse of pack_nibbles — reference used by the numerics tests."""
    p = packed.reshape(-1, 32)
    lo = (p & 0x0F).astype(np.int8)
    hi = ((p >> 4) & 0x0F).astype(np.int8)
    lo = ((lo ^ 8) - 8).astype(np.int8)
    hi = ((hi ^ 8) - 8).astype(np.int8)
    return np.concatenate([lo, hi], axis=1).reshape(-1)


# ------------------------------------------------------------------ AMX packing
def pack_b_tiles(q, n, k):
    """Pack an int8 B matrix (given as q[out=N, in=K]) into AMX tile order.

    Tiles are emitted [n_block][k_block]; each tile is 16 rows x 64 bytes
    holding (K=64, N=16) as [k/4][n][k%4] — the VNNI layout _tile_dpbssd wants.
    Streaming one n_block therefore walks K contiguously.
    """
    assert n % TILE_N == 0 and k % TILE_K == 0, (n, k)
    nb, kb = n // TILE_N, k // TILE_K
    t = q.reshape(nb, TILE_N, kb, TILE_K // 4, 4)
    t = t.transpose(0, 2, 3, 1, 4)  # [nb, kb, k/4, n, k%4]
    return np.ascontiguousarray(t).reshape(-1)


def quant_q8c(w):
    """int8 symmetric per output channel. w is [out, in] float32."""
    out, k = w.shape
    amax = np.abs(w).max(axis=1)
    amax[amax == 0] = 1.0
    scale = (amax / 127.0).astype(np.float32)
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return pack_b_tiles(q, out, k), scale


_MSE_TRIALS = np.linspace(0.82, 1.0, 10, dtype=np.float32)


def _q4_scale(wg):
    """Per-group int4 scale using all 16 levels, refined by an MSE search.

    Baseline follows the Q4_0 convention: the extreme element maps exactly onto
    -8, so the full [-8, 7] range is used rather than [-7, 7]. We then sweep a
    few shrink factors and keep whichever minimises reconstruction MSE — pure
    conversion-time cost, no runtime change (still one scale, no zero point).
    """
    amax = np.abs(wg).max(axis=2)
    idx = np.abs(wg).argmax(axis=2)
    ext = np.take_along_axis(wg, idx[:, :, None], axis=2)[:, :, 0]
    base = np.where(amax == 0, 1.0, ext / -8.0).astype(np.float32)

    best_s = base.copy()
    best_e = np.full(base.shape, np.inf, dtype=np.float32)
    for m in _MSE_TRIALS:
        s = base * m
        s = np.where(s == 0, 1.0, s)
        q = np.clip(np.rint(wg / s[:, :, None]), -8, 7)
        err = ((q * s[:, :, None] - wg) ** 2).sum(axis=2)
        take = err < best_e
        best_e = np.where(take, err, best_e)
        best_s = np.where(take, s, best_s)
    return best_s


def quant_q4g(w, group=GROUP):
    """int4 symmetric per (output channel, `group`-input group), AMX-tile packed.

    `group` must be a multiple of TILE_K so a scale group covers whole AMX tiles;
    the kernel drains its int32 accumulator every `group/64` tile steps, so this
    is the speed/accuracy dial (see bench/kernels/gemm_bench).

    Returns (packed_nibbles, scales[k/group, out] as f16).
    """
    out, k = w.shape
    assert group % TILE_K == 0 and k % group == 0, (group, k)
    wg = w.reshape(out, -1, group)
    scale = _q4_scale(wg)  # [out, g]
    q = np.clip(np.rint(wg / scale[:, :, None]), -8, 7).astype(np.int8).reshape(out, k)
    blob8 = pack_b_tiles(q, out, k)
    # scales transposed to [k/group, out] so a tile's 16 scales are contiguous
    return pack_nibbles(blob8), np.ascontiguousarray(scale.T).astype(np.float16)


# ------------------------------------------------------------- gather tables
def quant_q8_rows(w):
    """int8 symmetric per row, row-major (no tiling). For PLE / embedding
    tables, which the runtime gathers a row at a time rather than GEMMs."""
    amax = np.abs(w).max(axis=1)
    amax[amax == 0] = 1.0
    scale = (amax / 127.0).astype(np.float32)
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return q.reshape(-1), scale


def quant_q4_rows(w, gs=GROUP):
    """int4 per (row, 64-group), row-major. For gather tables."""
    rows, k = w.shape
    assert k % gs == 0
    wg = w.reshape(rows, -1, gs)
    scale = _q4_scale(wg)
    q = np.clip(np.rint(wg / scale[:, :, None]), -8, 7).astype(np.int8).reshape(rows, k)
    return pack_nibbles(q.reshape(-1)), scale.astype(np.float16)


# ------------------------------------------------------------------ error probe
def dequant_ref(w, fmt, group=GROUP):
    """Round-trip a weight through a format, for the conversion error report."""
    if fmt == "q8c":
        amax = np.abs(w).max(axis=1)
        amax[amax == 0] = 1.0
        s = (amax / 127.0)[:, None]
        return np.clip(np.rint(w / s), -127, 127) * s
    out, k = w.shape
    wg = w.reshape(out, -1, group)
    s = _q4_scale(wg).astype(np.float16).astype(np.float32)[:, :, None]
    return (np.clip(np.rint(wg / s), -8, 7) * s).reshape(out, k)


def rel_err(w, fmt, group=GROUP):
    d = dequant_ref(w, fmt, group)
    return float(np.linalg.norm(w - d) / (np.linalg.norm(w) + 1e-12))
