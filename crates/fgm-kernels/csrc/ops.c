// Elementwise / reduction / attention primitives for the Gemma 4 forward pass.
// AVX-512 throughout; every routine works on f32 activations laid out row-major
// as [tokens, dim], which is what the AMX GEMM produces and consumes.

#define _GNU_SOURCE
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>


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
                         float *, int, int, int);
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
                      start, end, scratch, 0, n_heads, 0);
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
                         int h0, int h1, int ring) {
  const int kvd = kv_heads * head_dim;
  const int grp = n_heads / kv_heads;
#define KVSLOT(t) ((ring) ? ((t) % (ring)) : (t))
  for (int h = h0; h < h1; h++) {
    const float *qh = q + (size_t)h * head_dim;
    const int kvh = h / grp;
    float *sc = scratch;
    float mx = -INFINITY;
    for (int t = start; t < end; t++) {
      const int8_t *kp = kc + (size_t)KVSLOT(t) * kvd + kvh * head_dim;
      __m512 acc = _mm512_setzero_ps();
      for (int i = 0; i < head_dim; i += 16) {
        __m512i kv = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(kp + i)));
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(qh + i), _mm512_cvtepi32_ps(kv), acc);
      }
      float s = _mm512_reduce_add_ps(acc) * ks[KVSLOT(t)];
      sc[t - start] = s;
      if (s > mx) mx = s;
    }
    float sum = 0.0f;
    for (int t = 0; t < end - start; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
    float inv = 1.0f / sum;
    float *oh = out + (size_t)h * head_dim;
    for (int i = 0; i < head_dim; i += 16) _mm512_storeu_ps(oh + i, _mm512_setzero_ps());
    for (int t = start; t < end; t++) {
      float w = sc[t - start] * inv * vs[KVSLOT(t)];
      if (w == 0.0f) continue;
      const int8_t *vp = vc + (size_t)KVSLOT(t) * kvd + kvh * head_dim;
      __m512 wv = _mm512_set1_ps(w);
      for (int i = 0; i < head_dim; i += 16) {
        __m512i vv = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(vp + i)));
        _mm512_storeu_ps(oh + i, _mm512_fmadd_ps(_mm512_cvtepi32_ps(vv), wv,
                                                 _mm512_loadu_ps(oh + i)));
      }
    }
  }
#undef KVSLOT
}

// Quantise one KV row (kv_heads*head_dim f32) to int8 with a single scale.
void fgm_quant_kv(const float *x, int n, int8_t *q, float *scale) {
  fgm_quant_act(x, 1, n, q, scale);
}
