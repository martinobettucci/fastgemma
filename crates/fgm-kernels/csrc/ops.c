// Elementwise / reduction / attention primitives for the Gemma 4 forward pass.
// AVX-512 throughout; every routine works on f32 activations laid out row-major
// as [tokens, dim], which is what the AMX GEMM produces and consumes.

#define _GNU_SOURCE
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

// Largest head_dim in the models this engine targets (Gemma 4 full-attention
// layers use 512; sliding layers use 256). Used for a stack-resident quantised
// query, so it must bound every head_dim the attention kernels can be handed.
#define FGM_MAX_HEAD_DIM 512


// --------------------------------------------------------------------- exp
// GCC has no _mm512_exp_ps (that is an Intel-compiler SVML intrinsic), so this
// is the classic Cephes range-reduction: exp(x) = 2^n * exp(z) with
// z = x - n*ln2 kept small, 2^n built by exponent bit assembly.
static inline __m512 exp512_ps(__m512 x) {
  x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-88.0f)), _mm512_set1_ps(88.0f));
  __m512 fx = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)),
                                   _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 z = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693359375f), x);
  z = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.12194440e-4f), z);
  __m512 y = _mm512_set1_ps(1.9875691500E-4f);
  y = _mm512_fmadd_ps(y, z, _mm512_set1_ps(1.3981999507E-3f));
  y = _mm512_fmadd_ps(y, z, _mm512_set1_ps(8.3334519073E-3f));
  y = _mm512_fmadd_ps(y, z, _mm512_set1_ps(4.1665795894E-2f));
  y = _mm512_fmadd_ps(y, z, _mm512_set1_ps(1.6666665459E-1f));
  y = _mm512_fmadd_ps(y, z, _mm512_set1_ps(5.0000001201E-1f));
  y = _mm512_fmadd_ps(y, _mm512_mul_ps(z, z), z);
  y = _mm512_add_ps(y, _mm512_set1_ps(1.0f));
  __m512i n = _mm512_cvtps_epi32(fx);
  __m512 p2 = _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_add_epi32(n, _mm512_set1_epi32(127)), 23));
  return _mm512_mul_ps(y, p2);
}

void fgm_fwht(float *, int, int);
void fgm_quant_act(const float *, int, int, int8_t *, float *);

// Runtime switch between the scalar and vectorised P.V weight fill.
//
// Both paths live in one binary on purpose. Comparing them across separate
// runs failed: this box's AMX tile-state corruption rate drifts between runs
// (5/12, 6/12 and 7/12 trials corrupted at warm-up in three consecutive
// measurements), and every corrupted tile re-runs a whole GEMM block, so
// run-to-run differences of 25% appear with no code change at all. A switch
// lets the two arms alternate inside one measurement window, where the drift
// is shared instead of being attributed to whichever arm ran during it.
int fgm_scalar_wfill = 0;
void fgm_set_scalar_wfill(int on) { fgm_scalar_wfill = on; }

// ------------------------------------------------------------------ RMSNorm
// Gemma 4: normed = x * (mean(x^2) + eps)^-0.5, then * weight (NOT 1 + weight,
// unlike Gemma 2/3). Accumulated in f32, matching the reference which upcasts.
void fgm_rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
  __m512 acc = _mm512_setzero_ps();
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_loadu_ps(x + i);
    acc = _mm512_fmadd_ps(v, v, acc);
  }
  float ss = _mm512_reduce_add_ps(acc);
  for (; i < n; i++) ss += x[i] * x[i];
  float inv = 1.0f / sqrtf(ss / (float)n + eps);
  __m512 vi = _mm512_set1_ps(inv);
  for (i = 0; i + 16 <= n; i += 16)
    _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x + i), vi),
                                            _mm512_loadu_ps(w + i)));
  for (; i < n; i++) out[i] = x[i] * inv * w[i];
}

// No learnable scale (Gemma 4 v_norm).
void fgm_rmsnorm_noscale(float *out, const float *x, int n, float eps) {
  __m512 acc = _mm512_setzero_ps();
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_loadu_ps(x + i);
    acc = _mm512_fmadd_ps(v, v, acc);
  }
  float ss = _mm512_reduce_add_ps(acc);
  for (; i < n; i++) ss += x[i] * x[i];
  float inv = 1.0f / sqrtf(ss / (float)n + eps);
  __m512 vi = _mm512_set1_ps(inv);
  for (i = 0; i + 16 <= n; i += 16)
    _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), vi));
  for (; i < n; i++) out[i] = x[i] * inv;
}

// --------------------------------------------------------------------- GELU
// tanh approximation, matching gelu_pytorch_tanh:
//   0.5x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
static inline __m512 gelu_tanh_v(__m512 x) {
  const __m512 c0 = _mm512_set1_ps(0.7978845608028654f);
  const __m512 c1 = _mm512_set1_ps(0.044715f);
  const __m512 half = _mm512_set1_ps(0.5f);
  const __m512 one = _mm512_set1_ps(1.0f);
  __m512 x3 = _mm512_mul_ps(_mm512_mul_ps(x, x), x);
  __m512 u = _mm512_mul_ps(c0, _mm512_fmadd_ps(c1, x3, x));
  // tanh(u) = 1 - 2/(exp(2u)+1)
  __m512 e = exp512_ps(_mm512_add_ps(u, u));
  __m512 t = _mm512_sub_ps(one, _mm512_div_ps(_mm512_set1_ps(2.0f), _mm512_add_ps(e, one)));
  return _mm512_mul_ps(_mm512_mul_ps(half, x), _mm512_add_ps(one, t));
}

void fgm_gelu(float *x, int n) {
  int i = 0;
  for (; i + 16 <= n; i += 16) _mm512_storeu_ps(x + i, gelu_tanh_v(_mm512_loadu_ps(x + i)));
  for (; i < n; i++) {
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
  }
}

// out = gelu(gate) * up   (fused FFN activation)
void fgm_gelu_mul(float *out, const float *gate, const float *up, int n) {
  int i = 0;
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(out + i, _mm512_mul_ps(gelu_tanh_v(_mm512_loadu_ps(gate + i)),
                                            _mm512_loadu_ps(up + i)));
  for (; i < n; i++) {
    float v = gate[i];
    out[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))) * up[i];
  }
}

// --------------------------------------------------------------------- RoPE
// Half-split convention (HF): out[j] = x[j]cos - x[j+d/2]sin,
//                             out[j+d/2] = x[j+d/2]cos + x[j]sin.
// inv_freq entries that are zero (the "nope" tail of proportional RoPE) give
// cos=1, sin=0, so those dims pass through untouched.
void fgm_rope(float *x, int n_heads, int head_dim, const float *inv_freq, int pos) {
  const int h = head_dim / 2;
  for (int j = 0; j < h; j++) {
    float a = (float)pos * inv_freq[j];
    float c = cosf(a), s = sinf(a);
    for (int hh = 0; hh < n_heads; hh++) {
      float *p = x + hh * head_dim;
      float lo = p[j], hi = p[j + h];
      p[j] = lo * c - hi * s;
      p[j + h] = hi * c + lo * s;
    }
  }
}

// ------------------------------------------------------- fast Walsh-Hadamard
// In-place, normalised, on contiguous blocks of `hsz` (a power of two). This is
// the runtime half of rotconv: the converter folded H into the weights, so the
// activation must be rotated by the same H before every GEMM.
void fgm_fwht(float *x, int n, int hsz) {
  const float scale = 1.0f / sqrtf((float)hsz);
  for (int b = 0; b < n; b += hsz) {
    float *p = x + b;
    for (int len = 1; len < hsz; len <<= 1)
      for (int i = 0; i < hsz; i += len << 1)
        for (int j = i; j < i + len; j += 16) {
          int w = (len < 16) ? 1 : 16;
          if (w == 1) {
            for (int jj = j; jj < j + len && jj < i + len; jj++) {
              float u = p[jj], v = p[jj + len];
              p[jj] = u + v; p[jj + len] = u - v;
            }
            break;
          }
          __m512 u = _mm512_loadu_ps(p + j), v = _mm512_loadu_ps(p + j + len);
          _mm512_storeu_ps(p + j, _mm512_add_ps(u, v));
          _mm512_storeu_ps(p + j + len, _mm512_sub_ps(u, v));
        }
    for (int i = 0; i + 16 <= hsz; i += 16)
      _mm512_storeu_ps(p + i, _mm512_mul_ps(_mm512_loadu_ps(p + i), _mm512_set1_ps(scale)));
  }
}

// ------------------------------------------------- activation quantisation
// Per-row symmetric int8. Rows are tokens, so each token gets its own scale —
// which is what makes the Hadamard rotation pay off, since it removes the
// per-token outliers that would otherwise set the scale.
void fgm_quant_act(const float *x, int rows, int k, int8_t *q, float *scale) {
  for (int r = 0; r < rows; r++) {
    const float *src = x + (size_t)r * k;
    __m512 amax = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= k; i += 16)
      amax = _mm512_max_ps(amax, _mm512_abs_ps(_mm512_loadu_ps(src + i)));
    float m = _mm512_reduce_max_ps(amax);
    for (; i < k; i++) { float a = fabsf(src[i]); if (a > m) m = a; }
    float s = (m == 0.0f) ? 1.0f : m / 127.0f;
    float inv = 1.0f / s;
    scale[r] = s;
    int8_t *dst = q + (size_t)r * k;
    __m512 vi = _mm512_set1_ps(inv);
    for (i = 0; i + 16 <= k; i += 16) {
      __m512i v = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(src + i), vi));
      _mm_storeu_si128((__m128i *)(dst + i), _mm512_cvtsepi32_epi8(v));
    }
    for (; i < k; i++) {
      int v = (int)lrintf(src[i] * inv);
      dst[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }
  }
}

// ------------------------------------------------------------- gather tables
// int8 row-major table: out = row * scale
void fgm_gather_q8r(float *out, const int8_t *tbl, const float *sc, int row, int k) {
  const int8_t *p = tbl + (size_t)row * k;
  __m512 s = _mm512_set1_ps(sc[row]);
  int i = 0;
  for (; i + 16 <= k; i += 16) {
    __m512i v = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(p + i)));
    _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_cvtepi32_ps(v), s));
  }
  for (; i < k; i++) out[i] = (float)p[i] * sc[row];
}

// int4 row-major table, group 64, canonical half-split nibbles.
// scales are [rows, k/64] f16.
void fgm_gather_q4r(float *out, const uint8_t *tbl, const _Float16 *sc, int row, int k) {
  const int g = k / 64;
  const uint8_t *p = tbl + (size_t)row * (k / 2);
  const _Float16 *rs = sc + (size_t)row * g;
  const __m256i m4 = _mm256_set1_epi8(0x0F), k8 = _mm256_set1_epi8(8);
  for (int b = 0; b < g; b++) {
    __m256i packed = _mm256_loadu_si256((const __m256i *)(p + b * 32));
    __m256i lo = _mm256_and_si256(packed, m4);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), m4);
    lo = _mm256_sub_epi8(_mm256_xor_si256(lo, k8), k8);
    hi = _mm256_sub_epi8(_mm256_xor_si256(hi, k8), k8);
    __m512 s = _mm512_set1_ps((float)rs[b]);
    float *o = out + b * 64;
    _Alignas(32) int8_t tmp[64];
    _mm256_storeu_si256((__m256i *)tmp, lo);
    _mm256_storeu_si256((__m256i *)(tmp + 32), hi);
    for (int i = 0; i < 64; i += 16) {
      __m512i v = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(tmp + i)));
      _mm512_storeu_ps(o + i, _mm512_mul_ps(_mm512_cvtepi32_ps(v), s));
    }
  }
}


// ------------------------------------------------- GEMM activation preamble
// Every GEMM must rotate (rotconv FWHT), quantise to int8 and tile-pack its
// activation before the AMX kernel can touch it. That work was single-threaded
// while the GEMM itself was not, which is why a standalone GEMM benches far
// above what the same shape delivers inside the forward pass.
//
// Split by whole 16-row tile blocks so every stage stays aligned: the FWHT
// operates on row-contiguous blocks of `hsz`, quantisation is per row, and
// pack_a's unit is exactly a 16-row tile.
//
//   src  [m, lda] f32   activation, rows may be strided
//   rot  [m, k]   f32   scratch for the rotated copy
//   qa   [m, k]   i8    quantised
//   qs   [m]      f32   per-row scales
//   pa            i8    tile-packed output for the AMX kernel
//
// r0/r1 are row bounds; r0 must be a multiple of 16.
void fgm_prep_rows(const float *src, int lda, float *rot, int8_t *qa, float *qs,
                   int8_t *pa, int m, int k, int hsz, int r0, int r1) {
  if (r0 >= r1) return;
  const int n = r1 - r0;
  for (int r = r0; r < r1; r++)
    memcpy(rot + (size_t)r * k, src + (size_t)r * lda, (size_t)k * sizeof(float));
  if (hsz > 0) fgm_fwht(rot + (size_t)r0 * k, n * k, hsz);
  fgm_quant_act(rot + (size_t)r0 * k, n, k, qa + (size_t)r0 * k, qs + r0);

  // pack_a for this row range only: tile blocks [r0/16, ceil(r1/16))
  const int KB = k / 64;
  for (int mb = r0 / 16; mb * 16 < r1; mb++) {
    for (int kb = 0; kb < KB; kb++) {
      int8_t *d = pa + ((size_t)mb * KB + kb) * 1024;
      for (int rr = 0; rr < 16; rr++) {
        int row = mb * 16 + rr;
        if (row < m) memcpy(d + rr * 64, qa + (size_t)row * k + kb * 64, 64);
        else memset(d + rr * 64, 0, 64);
      }
    }
  }
}

// -------------------------------------------------------------- elementwise
void fgm_add(float *a, const float *b, int n) {
  int i = 0;
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(a + i, _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
  for (; i < n; i++) a[i] += b[i];
}

void fgm_mul(float *a, const float *b, int n) {
  int i = 0;
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(a + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
  for (; i < n; i++) a[i] *= b[i];
}

void fgm_scale(float *a, float s, int n) {
  __m512 v = _mm512_set1_ps(s);
  int i = 0;
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(a + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), v));
  for (; i < n; i++) a[i] *= s;
}

// logits = cap * tanh(logits / cap)
void fgm_softcap(float *x, int n, float cap) {
  const __m512 c = _mm512_set1_ps(cap), ci = _mm512_set1_ps(1.0f / cap);
  const __m512 one = _mm512_set1_ps(1.0f), two = _mm512_set1_ps(2.0f);
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 u = _mm512_mul_ps(_mm512_loadu_ps(x + i), ci);
    __m512 e = exp512_ps(_mm512_add_ps(u, u));
    __m512 t = _mm512_sub_ps(one, _mm512_div_ps(two, _mm512_add_ps(e, one)));
    _mm512_storeu_ps(x + i, _mm512_mul_ps(c, t));
  }
  for (; i < n; i++) x[i] = cap * tanhf(x[i] / cap);
}

// ---------------------------------------------------------------- attention
void fgm_attend_q8_heads(float *, const float *, const int8_t *, const float *,
                         const int8_t *, const float *, int, int, int, int, int,
                         float *, int, int, int, int);
// One query row against a contiguous int8 KV cache for one layer.
//   k_cache / v_cache: [n_ctx, kv_heads * head_dim] int8 with per-(pos) scale
//   q: [n_heads, head_dim] f32
//   out: [n_heads, head_dim] f32
// scaling is 1.0 in Gemma 4 (q_norm handles magnitude), and MQA means every
// query head reads the same KV row.
void fgm_attend_q8(float *out, const float *q, const int8_t *kc, const float *ks,
                   const int8_t *vc, const float *vs, int n_heads, int kv_heads,
                   int head_dim, int start, int end, float *scratch) {
  fgm_attend_q8_heads(out, q, kc, ks, vc, vs, n_heads, kv_heads, head_dim,
                      start, end, scratch, 0, n_heads, 0, end);
}

// Same, restricted to heads [h0, h1) so the work can be split across threads.
//
// `ring` is the cache capacity for sliding layers whose KV lives in a ring
// buffer (0 = linear, full-length cache). Absolute positions must be mapped
// through it exactly as the write path does via LayerKv::slot -- reading a ring
// cache by absolute position walks off the end of the allocation the moment the
// context passes the window, which is how this was found: a segfault at 8k that
// no test under 512 tokens could reach.
void fgm_attend_q8_heads(float *out, const float *q, const int8_t *kc, const float *ks,
                         const int8_t *vc, const float *vs, int n_heads, int kv_heads,
                         int head_dim, int start, int end, float *scratch,
                         int h0, int h1, int ring, int cap) {
  const int kvd = kv_heads * head_dim;
  const int grp = n_heads / kv_heads;
#define KVSLOT(t) ((ring) ? ((t) % (ring)) : (t))
  for (int h = h0; h < h1; h++) {
    const float *qh = q + (size_t)h * head_dim;
    const int kvh = h / grp;
    float *sc = scratch;
    float mx = -INFINITY;

    // Q.K^T in int8 with AVX512-VNNI.
    //
    // The f32 version of this loop spent three of every four issue slots on
    // format conversion: per 16 elements it did load + cvtepi8_epi32 +
    // cvtepi32_ps + fmadd, so 16 MACs cost 4 uops. Measured end to end it ran
    // at 49.6 G MAC/s, 18.5% of this box's f32 FMA peak -- and attention is
    // 61.9% of an 8192-token prefill, so that inefficiency was the single
    // largest cost in the engine.
    //
    // vpdpbusd does 64 MACs per instruction but wants (u8, i8). K is int8, so
    // bias it into u8 with an XOR of 0x80 -- on a two's-complement byte that is
    // exactly +128 -- and correct afterwards:
    //
    //     sum_i (k_i + 128) * q_i  =  sum_i k_i q_i  +  128 * sum_i q_i
    //
    // sum_i q_i is one scalar per (row, head), computed here while quantising
    // q. Biasing K rather than Q is what keeps this free of any change to the
    // KV cache: no per-position row sums to store, no second layout.
    //
    // Q is quantised per head. q_norm bounds its range before RoPE by
    // construction, so a single per-head scale is well conditioned.
    int8_t qq[FGM_MAX_HEAD_DIM];
    float qamax = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      float a = fabsf(qh[i]);
      if (a > qamax) qamax = a;
    }
    const float qscale = qamax / 127.0f;
    const float qinv = qamax > 0.0f ? 127.0f / qamax : 0.0f;
    int32_t qsum = 0;
    for (int i = 0; i < head_dim; i++) {
      int v = (int)lrintf(qh[i] * qinv);
      v = v > 127 ? 127 : (v < -127 ? -127 : v);
      qq[i] = (int8_t)v;
      qsum += v;
    }
    const int32_t qcorr = 128 * qsum;
    const __m512i kbias = _mm512_set1_epi8((char)0x80);

    // Four positions per iteration against four accumulators. With VNNI the
    // dot product itself is only ~8 instructions for head_dim 512, so a
    // per-position _mm512_reduce_add_epi32 (~10 cycles) would dominate what it
    // reduces. Combining four accumulators with an unpack/hadd tree costs ~13
    // ops instead of the ~28 four separate reductions would.
    int t = start;
    for (; t + 4 <= end; t += 4) {
      const int8_t *k0 = kc + (size_t)KVSLOT(t) * kvd + kvh * head_dim;
      const int8_t *k1 = kc + (size_t)KVSLOT(t + 1) * kvd + kvh * head_dim;
      const int8_t *k2 = kc + (size_t)KVSLOT(t + 2) * kvd + kvh * head_dim;
      const int8_t *k3 = kc + (size_t)KVSLOT(t + 3) * kvd + kvh * head_dim;
      __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
      __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
      for (int i = 0; i < head_dim; i += 64) {
        const __m512i qv = _mm512_loadu_si512((const void *)(qq + i));
        a0 = _mm512_dpbusd_epi32(a0, _mm512_xor_si512(_mm512_loadu_si512((const void *)(k0 + i)), kbias), qv);
        a1 = _mm512_dpbusd_epi32(a1, _mm512_xor_si512(_mm512_loadu_si512((const void *)(k1 + i)), kbias), qv);
        a2 = _mm512_dpbusd_epi32(a2, _mm512_xor_si512(_mm512_loadu_si512((const void *)(k2 + i)), kbias), qv);
        a3 = _mm512_dpbusd_epi32(a3, _mm512_xor_si512(_mm512_loadu_si512((const void *)(k3 + i)), kbias), qv);
      }
      __m256i b0 = _mm256_add_epi32(_mm512_castsi512_si256(a0), _mm512_extracti64x4_epi64(a0, 1));
      __m256i b1 = _mm256_add_epi32(_mm512_castsi512_si256(a1), _mm512_extracti64x4_epi64(a1, 1));
      __m256i b2 = _mm256_add_epi32(_mm512_castsi512_si256(a2), _mm512_extracti64x4_epi64(a2, 1));
      __m256i b3 = _mm256_add_epi32(_mm512_castsi512_si256(a3), _mm512_extracti64x4_epi64(a3, 1));
      __m256i d = _mm256_hadd_epi32(_mm256_hadd_epi32(b0, b1), _mm256_hadd_epi32(b2, b3));
      __m128i e = _mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1));
      int32_t raw[4];
      _mm_storeu_si128((__m128i *)raw, e);
      for (int j = 0; j < 4; j++) {
        float s = (float)(raw[j] - qcorr) * qscale * ks[KVSLOT(t + j)];
        sc[t + j - start] = s;
        if (s > mx) mx = s;
      }
    }
    for (; t < end; t++) {
      const int8_t *kp = kc + (size_t)KVSLOT(t) * kvd + kvh * head_dim;
      __m512i acc = _mm512_setzero_si512();
      for (int i = 0; i < head_dim; i += 64) {
        acc = _mm512_dpbusd_epi32(
            acc, _mm512_xor_si512(_mm512_loadu_si512((const void *)(kp + i)), kbias),
            _mm512_loadu_si512((const void *)(qq + i)));
      }
      float s = (float)(_mm512_reduce_add_epi32(acc) - qcorr) * qscale * ks[KVSLOT(t)];
      sc[t - start] = s;
      if (s > mx) mx = s;
    }
    // Vectorised softmax. This loop ran scalar libm expf while the blocked
    // kernel two functions down already used exp512_ps -- and it is the hot one:
    // over an 8192-token prefill the full-attention layers alone evaluate
    // sum_r r * n_heads * n_full = 1.9e9 exponentials.
    const int n = end - start;
    __m512 vmx = _mm512_set1_ps(mx), vsum = _mm512_setzero_ps();
    int u = 0;
    for (; u + 16 <= n; u += 16) {
      __m512 e = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(sc + u), vmx));
      _mm512_storeu_ps(sc + u, e);
      vsum = _mm512_add_ps(vsum, e);
    }
    float sum = _mm512_reduce_add_ps(vsum);
    for (; u < n; u++) { sc[u] = expf(sc[u] - mx); sum += sc[u]; }
    float inv = 1.0f / sum;

    // P.V in int8 with VNNI, against a transposed V cache.
    //
    // out[i] = sum_t w_t * v_t[i] is a scaled accumulate, not a dot product, so
    // it cannot use vpdpbusd while V is stored [position][dim]. Stored instead
    // as groups of four consecutive slots interleaved per dim --
    //
    //     vt[((slot/4) * kvd + j) * 4 + (slot%4)]
    //
    // -- the four bytes vpdpbusd multiplies and adds within each 32-bit lane
    // are exactly four consecutive positions of one dim, so the reduction over
    // positions falls out of the instruction. That is the VNNI/AMX B-tile
    // layout; storing V this way is what makes P.V a GEMM.
    //
    // Weights are split across two u8 passes, giving 16-bit resolution:
    //
    //     iw = round(w * 65535 / wmax),  hi = iw >> 8,  lo = iw & 255
    //     out = (wmax/65535) * (256 * sum hi*v + sum lo*v)
    //
    // One u8 pass was tried first and measured 2.90x the error the int8 cache
    // already contributes -- worst on *flat* attention, which is exactly the
    // long-context regime this work exists to speed up. The second pass reuses
    // the same loaded V vector, so it costs two extra dpbusd and one extra
    // accumulator per 16 dims, not a second pass over memory. Still ~6x fewer
    // ops per MAC than the f32 version it replaces, against ~9.6x for single-u8.
    //
    // Accumulator bound: 255 * 127 * n_positions, so int32 is safe past 65k
    // positions.
    //
    // Accumulators live in scratch rather than registers: head_dim 512 would
    // need 32 zmm for one accumulator set alone. Sixteen slots (four groups)
    // are processed per pass over the dims, so each accumulator is loaded and
    // stored once per four vpdpbusd rather than once per one.
    //
    // Scratch layout, all sized off the cache capacity (slots are < cap, and a
    // score range is never longer than the cache that holds it):
    //   [0, cap)                    scores, already written above
    //   [cap, ...)                  u8 weight highs, then lows, by slot
    //   after that                  two int32 accumulator sets, head_dim each
    const int wstride = (cap + 3) & ~3;
    uint8_t *wq_hi = (uint8_t *)(scratch + cap);
    uint8_t *wq_lo = wq_hi + wstride;
    int32_t *acc_hi = (int32_t *)(scratch + cap + wstride / 2 + 4);
    int32_t *acc_lo = acc_hi + head_dim;

    float wmax = 0.0f;
    for (int t = start; t < end; t++) {
      float w = sc[t - start] * inv * vs[KVSLOT(t)];
      if (w > wmax) wmax = w;
    }
    float *oh = out + (size_t)h * head_dim;
    if (!(wmax > 0.0f)) {
      for (int i = 0; i < head_dim; i += 16) _mm512_storeu_ps(oh + i, _mm512_setzero_ps());
      continue;
    }
    const float wq_scale = wmax / 65535.0f, wq_inv = 65535.0f / wmax;

    // Slot runs. Without a ring the slots are [start, end); with one they are
    // that range modulo the capacity, which is at most two contiguous runs.
    int run_lo[2], run_hi[2], nrun;
    if (!ring) {
      run_lo[0] = start; run_hi[0] = end; nrun = 1;
    } else if (end - start >= ring) {
      run_lo[0] = 0; run_hi[0] = ring; nrun = 1;
    } else {
      int s0 = start % ring, span = end - start;
      if (s0 + span <= ring) { run_lo[0] = s0; run_hi[0] = s0 + span; nrun = 1; }
      else {
        run_lo[0] = s0; run_hi[0] = ring;
        run_lo[1] = 0;  run_hi[1] = s0 + span - ring; nrun = 2;
      }
    }

    // Zero each run's group-aligned span, so lanes outside the run contribute
    // nothing and partial head/tail groups need no special case.
    for (int rn = 0; rn < nrun; rn++) {
      size_t lo = (size_t)(run_lo[rn] & ~3), len = (size_t)(((run_hi[rn] + 3) & ~3)) - lo;
      memset(wq_hi + lo, 0, len);
      memset(wq_lo + lo, 0, len);
    }
    // Weight fill, scalar and deliberately so.
    //
    // A vectorised version of this loop shipped briefly and was reverted. It
    // computed (sc*vs)*(inv*wq_inv) with round-half-to-even, while the ring
    // path computed ((sc*inv)*vs)*wq_inv with round-half-up -- different
    // association, different rounding rule. Since ring layers took the scalar
    // path and non-ring layers the vectorised one, the same position quantised
    // differently depending on whether its layer used a ring buffer, and the
    // ring regression diverged by 0.94 in the logits.
    //
    // It was also worth nothing: measured end to end it moved prefill 0.7% and
    // decode 0.2%, both inside noise, because 28 of 35 layers are sliding and
    // took the scalar path regardless. A change that buys nothing and breaks a
    // correctness guard has no defence.
    for (int t = start; t < end; t++) {
      int sl = KVSLOT(t);
      int iw = (int)(sc[t - start] * inv * vs[sl] * wq_inv + 0.5f);
      if (iw > 65535) iw = 65535;
      if (iw < 0) iw = 0;
      wq_hi[sl] = (uint8_t)(iw >> 8);
      wq_lo[sl] = (uint8_t)(iw & 255);
    }

    for (int i = 0; i < head_dim; i += 16) {
      _mm512_storeu_si512((void *)(acc_hi + i), _mm512_setzero_si512());
      _mm512_storeu_si512((void *)(acc_lo + i), _mm512_setzero_si512());
    }

    const size_t gstride = (size_t)kvd * 4;
    for (int rn = 0; rn < nrun; rn++) {
      const int glo = run_lo[rn] & ~3, ghi = (run_hi[rn] + 3) & ~3;
      for (int g = glo; g < ghi; g += 16) {
        const int ng = (ghi - g >= 16) ? 4 : (ghi - g) / 4;
        __m512i wh[4], wl[4];
        for (int j = 0; j < 4; j++) {
          wh[j] = j < ng ? _mm512_set1_epi32(*(const int32_t *)(wq_hi + g + 4 * j))
                         : _mm512_setzero_si512();
          wl[j] = j < ng ? _mm512_set1_epi32(*(const int32_t *)(wq_lo + g + 4 * j))
                         : _mm512_setzero_si512();
        }
        const int8_t *v0 = vc + ((size_t)(g / 4) * kvd + (size_t)kvh * head_dim) * 4;
        for (int i = 0; i < head_dim; i += 16) {
          __m512i ah = _mm512_loadu_si512((const void *)(acc_hi + i));
          __m512i al = _mm512_loadu_si512((const void *)(acc_lo + i));
          const int8_t *vp = v0 + (size_t)i * 4;
          for (int j = 0; j < ng; j++) {
            __m512i vv = _mm512_loadu_si512((const void *)(vp + (size_t)j * gstride));
            ah = _mm512_dpbusd_epi32(ah, wh[j], vv);
            al = _mm512_dpbusd_epi32(al, wl[j], vv);
          }
          _mm512_storeu_si512((void *)(acc_hi + i), ah);
          _mm512_storeu_si512((void *)(acc_lo + i), al);
        }
      }
    }
    {
      const __m512 vsc = _mm512_set1_ps(wq_scale);
      for (int i = 0; i < head_dim; i += 16) {
        __m512i a = _mm512_add_epi32(
            _mm512_slli_epi32(_mm512_loadu_si512((const void *)(acc_hi + i)), 8),
            _mm512_loadu_si512((const void *)(acc_lo + i)));
        _mm512_storeu_ps(oh + i, _mm512_mul_ps(_mm512_cvtepi32_ps(a), vsc));
      }
    }
  }
#undef KVSLOT
}


// Quantise one KV row (kv_heads*head_dim f32) to int8 with a single scale.
void fgm_quant_kv(const float *x, int n, int8_t *q, float *scale) {
  fgm_quant_act(x, 1, n, q, scale);
}

// Store one quantised V row into the transposed cache.
//
// Layout is groups of four consecutive slots interleaved per dim:
//     vt[((slot/4) * n + j) * 4 + (slot%4)] = src[j]
// which is what lets P.V reduce over positions with vpdpbusd. Writing is a
// stride-4 scatter of n bytes, paid once per position per layer; reading is
// what the whole prefill does repeatedly, so this is the right side to make
// awkward.
void fgm_store_v_t(int8_t *vt, const int8_t *src, int n, int slot) {
  int8_t *dst = vt + (size_t)(slot >> 2) * n * 4 + (slot & 3);
  for (int j = 0; j < n; j++) dst[(size_t)j * 4] = src[j];
}

// ------------------------------------------------- head-batched Q.K^T (MQA)
// Scores for ALL heads of one query row against one K range, loading each K
// line once instead of once per head.
//
// Why this and not AMX: measured, attention runs at 3-7% of the VNNI ceiling
// it already has, so the multiplier is not what is scarce. With one KV head
// every K byte feeds exactly one multiply-accumulate in the head-outer loop --
// arithmetic intensity 1 MAC/byte -- which caps attention at 33 G MAC/s from
// DRAM and 58 from L3 regardless of instruction set. AMX raises a ceiling that
// is not binding.
//
// MQA is the opening: `n_heads` query heads share ONE kv head, so a K line
// loaded once can serve all of them. That takes intensity from 1 to n_heads
// MACs per byte -- 8x here -- moving the L3 ceiling from 58 to ~464 G MAC/s,
// which is 80% of VNNI peak. Only then does the multiplier become the limit.
//
// Crucially this changes no stored value, only loop order, so it carries zero
// accuracy risk. After int4 KV destroyed retention (0/7) that property is
// worth as much as the speed.
//
// out_scores: [n_heads, end-start] f32, one row per head.
void fgm_qk_heads_batched(float *out_scores, const int8_t *qq, const int32_t *qsum,
                          const float *qscale, const int8_t *kc, const float *ks,
                          int n_heads, int head_dim, int kvd, int start, int end,
                          int ring, int score_stride) {
  const __m512i kbias = _mm512_set1_epi8((char)0x80);
  const int n = end - start;
  for (int t = start; t < end; t++) {
    const int slot = ring ? (t % ring) : t;
    const int8_t *kp = kc + (size_t)slot * kvd;
    const float kscale = ks[slot];
    // One pass over this K line serves every head.
    for (int h = 0; h < n_heads; h++) {
      const int8_t *qh = qq + (size_t)h * head_dim;
      __m512i acc = _mm512_setzero_si512();
      for (int i = 0; i < head_dim; i += 64) {
        acc = _mm512_dpbusd_epi32(
            acc, _mm512_xor_si512(_mm512_loadu_si512((const void *)(kp + i)), kbias),
            _mm512_loadu_si512((const void *)(qh + i)));
      }
      out_scores[(size_t)h * score_stride + (t - start)] =
          (float)(_mm512_reduce_add_epi32(acc) - 128 * qsum[h]) * qscale[h] * kscale;
    }
  }
  (void)n;
}

// ---------------------------------------------- head-batched attention (MQA)
// All heads of ONE query row, sharing each K and V line across every head.
//
// Same maths as fgm_attend_q8_heads, different loop order. With one KV head the
// head-outer kernel re-streams the whole K range once per head, so each byte
// feeds exactly one multiply-accumulate; here a K line loaded once serves all
// n_heads, taking arithmetic intensity from 1 to n_heads MACs per byte.
//
// Measured on Q.K^T alone (bench/kernels/qk_bench.c), head_dim 512:
//   ctx 2048  88.3 -> 73.3 G MAC/s   0.83x   head-outer wins
//   ctx 4096  48.1 -> 70.6 G MAC/s   1.47x
//   ctx 8192  28.2 -> 72.9 G MAC/s   2.59x
//
// The crossover is physical: head-outer batches four positions against four
// accumulators and pays ~1/4 of a horizontal reduction per score, while this
// pays one per (position, head). Below the L2 line the reductions dominate and
// head-outer is ~20% faster; above it, bandwidth dominates and this wins.
// head_dim 512 x 4096 = 2 MB and head_dim 256 x 8192 = 2 MB -- both crossovers
// sit exactly on the per-core L2 capacity.
//
// Scratch layout (floats), sized by the caller from fgm_rowbatch_scratch():
//   [0, n_heads*cap)   scores, one row of `cap` per head
//   then               u8 weight planes and int32 accumulators, as in the
//                      per-head kernel
void fgm_attend_q8_row(float *out, const float *q, const int8_t *kc, const float *ks,
                       const int8_t *vc, const float *vs, int n_heads, int kv_heads,
                       int head_dim, int start, int end, float *scratch,
                       int ring, int cap) {
  const int kvd = kv_heads * head_dim;
  const int n = end - start;
  if (n <= 0) {
    for (int i = 0; i < n_heads * head_dim; i++) out[i] = 0.0f;
    return;
  }
  // Only the MQA case shares a KV head across all query heads; anything else
  // falls back rather than silently reading the wrong head's cache.
  if (kv_heads != 1) {
    fgm_attend_q8_heads(out, q, kc, ks, vc, vs, n_heads, kv_heads, head_dim,
                        start, end, scratch, 0, n_heads, ring, cap);
    return;
  }

  _Alignas(64) int8_t qq[8 * FGM_MAX_HEAD_DIM];
  _Alignas(64) int32_t qsum[8];
  _Alignas(64) float qscale[8];
  const int nh = n_heads > 8 ? 8 : n_heads;
  for (int h = 0; h < nh; h++) {
    const float *qh = q + (size_t)h * head_dim;
    float amax = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      float a = fabsf(qh[i]);
      if (a > amax) amax = a;
    }
    qscale[h] = amax / 127.0f;
    const float qinv = amax > 0.0f ? 127.0f / amax : 0.0f;
    int32_t s = 0;
    for (int i = 0; i < head_dim; i++) {
      int v = (int)lrintf(qh[i] * qinv);
      v = v > 127 ? 127 : (v < -127 ? -127 : v);
      qq[(size_t)h * head_dim + i] = (int8_t)v;
      s += v;
    }
    qsum[h] = s;
  }

  fgm_qk_heads_batched(scratch, qq, qsum, qscale, kc, ks, nh, head_dim, kvd,
                       start, end, ring, cap);

  // Softmax and P.V per head, over the scores just produced. Reuses the
  // per-head kernel's tail by calling it with a precomputed score row would
  // require threading a flag through it; instead the small amount of work here
  // is done directly, which keeps the hot per-head path untouched.
  const int wstride = (cap + 3) & ~3;
  uint8_t *wq_hi = (uint8_t *)(scratch + (size_t)nh * cap);
  uint8_t *wq_lo = wq_hi + wstride;
  int32_t *acc_hi = (int32_t *)(scratch + (size_t)nh * cap + wstride / 2 + 4);
  int32_t *acc_lo = acc_hi + head_dim;

  for (int h = 0; h < nh; h++) {
    float *sc = scratch + (size_t)h * cap;
    float *oh = out + (size_t)h * head_dim;
    float mx = -INFINITY;
    for (int t = 0; t < n; t++) if (sc[t] > mx) mx = sc[t];

    __m512 vmx = _mm512_set1_ps(mx), vsum = _mm512_setzero_ps();
    int u = 0;
    for (; u + 16 <= n; u += 16) {
      __m512 e = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(sc + u), vmx));
      _mm512_storeu_ps(sc + u, e);
      vsum = _mm512_add_ps(vsum, e);
    }
    float sum = _mm512_reduce_add_ps(vsum);
    for (; u < n; u++) { sc[u] = expf(sc[u] - mx); sum += sc[u]; }
    const float inv = 1.0f / sum;

    float wmax = 0.0f;
    for (int t = start; t < end; t++) {
      const int sl = ring ? (t % ring) : t;
      float w = sc[t - start] * inv * vs[sl];
      if (w > wmax) wmax = w;
    }
    if (!(wmax > 0.0f)) {
      for (int i = 0; i < head_dim; i += 16) _mm512_storeu_ps(oh + i, _mm512_setzero_ps());
      continue;
    }
    const float wq_scale = wmax / 65535.0f, wq_inv = 65535.0f / wmax;

    int run_lo[2], run_hi[2], nrun;
    if (!ring) { run_lo[0] = start; run_hi[0] = end; nrun = 1; }
    else if (n >= ring) { run_lo[0] = 0; run_hi[0] = ring; nrun = 1; }
    else {
      int s0 = start % ring;
      if (s0 + n <= ring) { run_lo[0] = s0; run_hi[0] = s0 + n; nrun = 1; }
      else { run_lo[0] = s0; run_hi[0] = ring; run_lo[1] = 0; run_hi[1] = s0 + n - ring; nrun = 2; }
    }
    for (int rn = 0; rn < nrun; rn++) {
      size_t lo = (size_t)(run_lo[rn] & ~3), len = (size_t)((run_hi[rn] + 3) & ~3) - lo;
      memset(wq_hi + lo, 0, len);
      memset(wq_lo + lo, 0, len);
    }
    for (int t = start; t < end; t++) {
      const int sl = ring ? (t % ring) : t;
      int iw = (int)(sc[t - start] * inv * vs[sl] * wq_inv + 0.5f);
      if (iw > 65535) iw = 65535;
      if (iw < 0) iw = 0;
      wq_hi[sl] = (uint8_t)(iw >> 8);
      wq_lo[sl] = (uint8_t)(iw & 255);
    }
    for (int i = 0; i < head_dim; i += 16) {
      _mm512_storeu_si512((void *)(acc_hi + i), _mm512_setzero_si512());
      _mm512_storeu_si512((void *)(acc_lo + i), _mm512_setzero_si512());
    }
    const size_t gstride = (size_t)kvd * 4;
    for (int rn = 0; rn < nrun; rn++) {
      const int glo = run_lo[rn] & ~3, ghi = (run_hi[rn] + 3) & ~3;
      for (int g = glo; g < ghi; g += 16) {
        const int ng = (ghi - g >= 16) ? 4 : (ghi - g) / 4;
        __m512i wh[4], wl[4];
        for (int j = 0; j < 4; j++) {
          wh[j] = j < ng ? _mm512_set1_epi32(*(const int32_t *)(wq_hi + g + 4 * j))
                         : _mm512_setzero_si512();
          wl[j] = j < ng ? _mm512_set1_epi32(*(const int32_t *)(wq_lo + g + 4 * j))
                         : _mm512_setzero_si512();
        }
        const int8_t *v0 = vc + (size_t)(g / 4) * kvd * 4;
        for (int i = 0; i < head_dim; i += 16) {
          __m512i ah = _mm512_loadu_si512((const void *)(acc_hi + i));
          __m512i al = _mm512_loadu_si512((const void *)(acc_lo + i));
          const int8_t *vp = v0 + (size_t)i * 4;
          for (int j = 0; j < ng; j++) {
            __m512i vv = _mm512_loadu_si512((const void *)(vp + (size_t)j * gstride));
            ah = _mm512_dpbusd_epi32(ah, wh[j], vv);
            al = _mm512_dpbusd_epi32(al, wl[j], vv);
          }
          _mm512_storeu_si512((void *)(acc_hi + i), ah);
          _mm512_storeu_si512((void *)(acc_lo + i), al);
        }
      }
    }
    const __m512 vsc = _mm512_set1_ps(wq_scale);
    for (int i = 0; i < head_dim; i += 16) {
      __m512i a = _mm512_add_epi32(
          _mm512_slli_epi32(_mm512_loadu_si512((const void *)(acc_hi + i)), 8),
          _mm512_loadu_si512((const void *)(acc_lo + i)));
      _mm512_storeu_ps(oh + i, _mm512_mul_ps(_mm512_cvtepi32_ps(a), vsc));
    }
  }
}

// Floats of scratch fgm_attend_q8_row needs.
int fgm_rowbatch_scratch(int n_heads, int cap, int head_dim) {
  return n_heads * cap + (cap + 3) / 2 + 2 * head_dim + 64;
}
