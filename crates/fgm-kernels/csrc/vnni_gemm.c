// AVX-512 VNNI INT8 GEMM -- the fallback for hosts without AMX.
//
// Same contract as the AMX kernels in amx_gemm.c, same weight bytes, same
// packed-A bytes, same C layout. Nothing about the file format, the converter,
// or the AMX path changes; this file only adds a second way to consume them.
//
// Why it can read the AMX layout unchanged
// ----------------------------------------
// A B tile is 16 rows x 64 bytes holding (K=64, N=16) as [k/4][n][k%4]. Read
// one of those 64-byte rows as a __m512i and you have, in 32-bit lane n, the
// four bytes B[4r+0..3][n] -- which is exactly the operand shape vpdpbusd
// wants. So one tile row is one VNNI vector, and the AMX "tile order" is just
// VNNI order chunked 16 rows at a time. No repacking, no second weight file.
//
// The activation side is the broadcast operand: for the same four k values,
// row m contributes the dword A[m][4r+0..3], identical across all 16 output
// columns. GCC folds that into the embedded broadcast, so the inner loop is
// literally `vpdpbusd zmm_acc, zmm_b, dword ptr [a]{1to16}` -- one instruction
// per 64 MACs, no A load in the loop body.
//
// The signedness problem, and why it is paid on B
// -----------------------------------------------
// vpdpbusd multiplies *unsigned* x *signed*, but both our operands are signed
// int8. Biasing one side into u8 costs a correction term equal to 128 times
// the sum of the other side over the same k range:
//
//   sum (B + 128) * A  =  sum B*A  +  128 * sum A
//
// Biasing B (one `vpxorq` with 0x80 per tile row) needs `sum A` -- a per-row
// quantity we can compute here in O(M*K). Biasing A instead would need
// `sum B` per output column, which is a property of the weights and would mean
// touching the converter. So B is biased: the xor is amortised across all 16
// rows of an m-block, ~6% of the inner loop, and the weight files stay exactly
// as the AMX path wrote them.
//
// Blocking mirrors amx_gemm.c exactly -- an n-panel sized to hold its B slice
// in L2, with all of M streamed through it -- because that decision was about
// DRAM traffic, not about which multiply instruction runs.
//
// M-blocks are always 16 rows wide because fgm_pack_a already zero-pads to a
// multiple of 16; padding rows contribute zero and are simply not stored. A
// second instantiation at 8 rows exists for decode, where the model is read
// once per token and doing 16 rows' worth of MACs for 8 useful ones is pure
// waste.

#define _GNU_SOURCE
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#define PK_TILE 512   // packed int4 tile bytes
#define Q8_TILE 1024  // int8 tile bytes

// ------------------------------------------------------------------- cpu id
// Leaf 7, subleaf 0: EDX bit 24 = AMX-TILE, bit 25 = AMX-INT8.
//
// This is deliberately not `fgm_amx_init`'s return value: that is a per-thread
// permission grant, and asking it on a CPU with no tile registers is a syscall
// away from a SIGILL the moment anything executes a tile instruction. The
// backend decision has to be answerable without running one.
int fgm_cpu_has_amx(void) {
  uint32_t a = 0, b = 0, c = 0, d = 0;
  __asm__ __volatile__("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                       : "a"(0), "c"(0));
  if (a < 7) return 0;
  __asm__ __volatile__("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                       : "a"(7), "c"(0));
  return (d >> 24 & 1) && (d >> 25 & 1);
}

int fgm_cpu_has_avx512_vnni(void) {
  uint32_t a = 0, b = 0, c = 0, d = 0;
  __asm__ __volatile__("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                       : "a"(0), "c"(0));
  if (a < 7) return 0;
  __asm__ __volatile__("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                       : "a"(7), "c"(0));
  return (b >> 16 & 1) && (c >> 11 & 1);  // AVX512F && AVX512_VNNI
}

// Same target as amx_gemm.c: ~256 KB of B resident per n-panel.
static inline int panel_nblocks(int K, int bytes_per_weight_x2) {
  int nb = (int)(262144L * 2 / ((long)bytes_per_weight_x2 * 16 * K));
  if (nb < 4) nb = 4;
  return nb / 4 * 4;
}

// Sum of a row's int8 values over `nb64` 64-byte chunks, chunks `stride` apart.
// vpdpbusd against an all-ones unsigned operand gives sixteen partial sums for
// the price of one instruction per 64 bytes, so this costs 1/16th of the main
// loop it corrects.
static inline int32_t row_sum(const int8_t *p, int nb64, size_t stride) {
  const __m512i ones = _mm512_set1_epi8(1);
  __m512i s = _mm512_setzero_si512();
  for (int i = 0; i < nb64; i++)
    s = _mm512_dpbusd_epi32(s, ones, _mm512_loadu_si512((const void *)(p + i * stride)));
  return _mm512_reduce_add_epi32(s);
}

// ------------------------------------------------------------- int4 unpack
// Canonical layout: within a 64-byte block, packed[i] (i<32) holds value i in
// the low nibble and value i+32 in the high nibble.
//
// The nibble is TWO'S-COMPLEMENT 4-bit, not offset-by-8: amx_gemm.c's unpack64
// sign-extends with (v ^ 8) - 8, so 0..7 stay positive and 8..15 become -8..-1.
// Writing it as the Q4_0 offset convention (v - 8) instead costs nothing on
// half the values and flips the other half by 16, which is exactly the bug
// that shipped here first -- and it survived the unit test because the test's
// own reference made the same assumption. Adding the +128 VNNI bias folds into
// the same expression: ((v ^ 8) - 8) + 128 == (v ^ 8) + 120.
//
// The result is returned in a register rather than written to a scratch tile
// the way the AMX path must: _tile_loadd only reads memory, but vpdpbusd takes
// a register, so the round trip -- and the store-to-load stall behind it --
// simply does not exist here.
static inline __m512i unpack64_u8(const uint8_t *src) {
  const __m256i m4 = _mm256_set1_epi8(0x0F), k8 = _mm256_set1_epi8(8),
                k120 = _mm256_set1_epi8(120);
  __m256i p = _mm256_loadu_si256((const __m256i *)src);
  __m256i lo = _mm256_add_epi8(
      _mm256_xor_si256(_mm256_and_si256(p, m4), k8), k120);
  __m256i hi = _mm256_add_epi8(
      _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(p, 4), m4), k8), k120);
  return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

// ------------------------------------------------------------- register block
// The inner block is 8 rows x 2 n-blocks (32 output columns), which is not the
// obvious shape and was not the first one tried.
//
// 16 rows x 1 n-block uses the same 16 accumulators and looks equivalent, but
// it is measurably worse for two reasons. GCC spills a 16-element accumulator
// array to the stack no matter how it is coaxed, and -- the real cost -- it
// needs one vpbroadcastd per dpbusd, because every row has a different
// activation dword. Two n-blocks let each broadcast feed two multiplies, so
// the loop issues 8 broadcasts and 2 B loads per 16 dpbusd instead of 16 and 1.
// The B side pays two loads and two xors per row-of-tile rather than one, which
// is the trade: fewer load-port uops per unit of VNNI work.
//
// Rows come in eights because packed A is a 16-row tile and half of one is
// still contiguous, so the 8-row block is addressable with a base offset and
// costs nothing at the edges. It also happens to be the decode batch.
#define ACC_ZERO()                                                             \
  __m512i c00 = _mm512_setzero_si512(), c01 = _mm512_setzero_si512(),          \
          c02 = _mm512_setzero_si512(), c03 = _mm512_setzero_si512(),          \
          c04 = _mm512_setzero_si512(), c05 = _mm512_setzero_si512(),          \
          c06 = _mm512_setzero_si512(), c07 = _mm512_setzero_si512(),          \
          c10 = _mm512_setzero_si512(), c11 = _mm512_setzero_si512(),          \
          c12 = _mm512_setzero_si512(), c13 = _mm512_setzero_si512(),          \
          c14 = _mm512_setzero_si512(), c15 = _mm512_setzero_si512(),          \
          c16 = _mm512_setzero_si512(), c17 = _mm512_setzero_si512()

// One row-of-tile: two B vectors, eight activation broadcasts, sixteen MACs.
#define FMA16(b0, b1, ap, r)                                                   \
  do {                                                                         \
    __m512i a;                                                                 \
    a = _mm512_set1_epi32(*(const int32_t *)((ap) + 0 * 64 + (r) * 4));         \
    c00 = _mm512_dpbusd_epi32(c00, b0, a); c10 = _mm512_dpbusd_epi32(c10, b1, a); \
    a = _mm512_set1_epi32(*(const int32_t *)((ap) + 1 * 64 + (r) * 4));         \
    c01 = _mm512_dpbusd_epi32(c01, b0, a); c11 = _mm512_dpbusd_epi32(c11, b1, a); \
    a = _mm512_set1_epi32(*(const int32_t *)((ap) + 2 * 64 + (r) * 4));         \
    c02 = _mm512_dpbusd_epi32(c02, b0, a); c12 = _mm512_dpbusd_epi32(c12, b1, a); \
    a = _mm512_set1_epi32(*(const int32_t *)((ap) + 3 * 64 + (r) * 4));         \
    c03 = _mm512_dpbusd_epi32(c03, b0, a); c13 = _mm512_dpbusd_epi32(c13, b1, a); \
    a = _mm512_set1_epi32(*(const int32_t *)((ap) + 4 * 64 + (r) * 4));         \
    c04 = _mm512_dpbusd_epi32(c04, b0, a); c14 = _mm512_dpbusd_epi32(c14, b1, a); \
    a = _mm512_set1_epi32(*(const int32_t *)((ap) + 5 * 64 + (r) * 4));         \
    c05 = _mm512_dpbusd_epi32(c05, b0, a); c15 = _mm512_dpbusd_epi32(c15, b1, a); \
    a = _mm512_set1_epi32(*(const int32_t *)((ap) + 6 * 64 + (r) * 4));         \
    c06 = _mm512_dpbusd_epi32(c06, b0, a); c16 = _mm512_dpbusd_epi32(c16, b1, a); \
    a = _mm512_set1_epi32(*(const int32_t *)((ap) + 7 * 64 + (r) * 4));         \
    c07 = _mm512_dpbusd_epi32(c07, b0, a); c17 = _mm512_dpbusd_epi32(c17, b1, a); \
  } while (0)

// Apply `body(t, m, accumulator)` to all sixteen, so the epilogues stay short.
#define ACC_EACH(body)                                                         \
  do {                                                                         \
    body(0, 0, c00); body(0, 1, c01); body(0, 2, c02); body(0, 3, c03);         \
    body(0, 4, c04); body(0, 5, c05); body(0, 6, c06); body(0, 7, c07);         \
    body(1, 0, c10); body(1, 1, c11); body(1, 2, c12); body(1, 3, c13);         \
    body(1, 4, c14); body(1, 5, c15); body(1, 6, c16); body(1, 7, c17);         \
  } while (0)

// ============================================================ int8 per-channel
// Accumulated across the whole K: int8 weights carry one scale per output
// channel, so there is nothing to drain for.
//
// Range: |acc| <= K * 255 * 127. The +128 bias makes the unsigned side twice
// the AMX path's, so the headroom is worth stating: it holds to K = 66048.
static void q8c_block(int M, int K, const int8_t *A, const float *a_scale,
                      const int8_t *B, const float *b_scale, float *C, int ldc,
                      int mb, int m0, int nb, const int32_t *asum) {
  const int KB = K / 64;
  const size_t nbs = (size_t)KB * Q8_TILE;
  const __m512i sign = _mm512_set1_epi8((char)0x80);
  ACC_ZERO();
  for (int kb = 0; kb < KB; kb++) {
    const int8_t *bp = B + (size_t)nb * nbs + (size_t)kb * Q8_TILE;
    const int8_t *ap = A + ((size_t)mb * KB + kb) * 1024 + (size_t)m0 * 64;
    for (int r = 0; r < 16; r++) {
      __m512i b0 = _mm512_xor_si512(_mm512_loadu_si512((const void *)(bp + r * 64)), sign);
      __m512i b1 = _mm512_xor_si512(_mm512_loadu_si512((const void *)(bp + nbs + r * 64)), sign);
      FMA16(b0, b1, ap, r);
    }
  }
  const __m512 bs0 = _mm512_loadu_ps(b_scale + nb * 16);
  const __m512 bs1 = _mm512_loadu_ps(b_scale + (nb + 1) * 16);
#define Q8_STORE(T, MM, ACC)                                                   \
  do {                                                                         \
    int row = mb * 16 + m0 + (MM);                                             \
    if (row < M) {                                                             \
      __m512i v = _mm512_sub_epi32(ACC, _mm512_set1_epi32(128 * asum[m0 + (MM)])); \
      _mm512_storeu_ps(                                                        \
          C + (size_t)row * ldc + (nb + (T)) * 16,                             \
          _mm512_mul_ps(_mm512_cvtepi32_ps(v),                                 \
                        _mm512_mul_ps((T) ? bs1 : bs0,                         \
                                      _mm512_set1_ps(a_scale[row]))));         \
    }                                                                          \
  } while (0)
  ACC_EACH(Q8_STORE);
#undef Q8_STORE
}

void fgm_gemm_q8c_vnni(int M, int N, int K, const int8_t *A, const float *a_scale,
                       const int8_t *B, const float *b_scale, float *C, int ldc,
                       int n0, int n1) {
  const int KB = K / 64;
  const int npanel = panel_nblocks(K, 2);
  const int MB = (M + 15) / 16;
  (void)N;

  for (int np = n0; np < n1; np += npanel) {
    const int ne = (np + npanel < n1) ? np + npanel : n1;
    for (int mb = 0; mb < MB; mb++) {
      int32_t asum[16];
      for (int m = 0; m < 16; m++)
        asum[m] = row_sum(A + (size_t)mb * KB * 1024 + m * 64, KB, 1024);
      for (int m0 = 0; m0 < 16; m0 += 8) {
        if (mb * 16 + m0 >= M) break;
        for (int nb = np; nb + 1 < ne; nb += 2)
          q8c_block(M, K, A, a_scale, B, b_scale, C, ldc, mb, m0, nb, asum);
      }
    }
  }
}

// ============================================================== int4 grouped
// The int32 accumulator has to be drained into f32 every `group` k-values,
// because that is how often the weight scale changes. Same structure as the
// AMX path; the difference is that the bias correction is per group too, so
// `asum` is indexed by group as well as by row.
static void q4g_block(int M, int N, int K, const int8_t *A, const uint8_t *Bq,
                      const uint16_t *b_scale, int gk, int NG, float *C, int ldc,
                      int mb, int m0, int nb, const int32_t (*asum)[16]) {
  const int KB = K / 64;
  const size_t nbs = (size_t)KB * PK_TILE;
  float *Cb = C + (size_t)(mb * 16 + m0) * ldc + nb * 16;
  for (int m = 0; m < 8 && mb * 16 + m0 + m < M; m++)
    memset(Cb + (size_t)m * ldc, 0, 32 * sizeof(float));

  for (int g = 0; g < NG; g++) {
    ACC_ZERO();
    for (int kb = g * gk; kb < (g + 1) * gk; kb++) {
      const uint8_t *bp = Bq + (size_t)nb * nbs + (size_t)kb * PK_TILE;
      const int8_t *ap = A + ((size_t)mb * KB + kb) * 1024 + (size_t)m0 * 64;
      for (int r = 0; r < 16; r++) {
        __m512i b0 = unpack64_u8(bp + r * 32);
        __m512i b1 = unpack64_u8(bp + nbs + r * 32);
        FMA16(b0, b1, ap, r);
      }
    }
    const __m512 bs0 = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i *)(b_scale + (size_t)g * N + nb * 16)));
    const __m512 bs1 = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i *)(b_scale + (size_t)g * N + (nb + 1) * 16)));
#define Q4_STORE(T, MM, ACC)                                                   \
    do {                                                                       \
      if (mb * 16 + m0 + (MM) < M) {                                           \
        __m512i v = _mm512_sub_epi32(ACC, _mm512_set1_epi32(128 * asum[g][m0 + (MM)])); \
        float *o = Cb + (size_t)(MM) * ldc + (T) * 16;                         \
        _mm512_storeu_ps(o, _mm512_fmadd_ps(_mm512_cvtepi32_ps(v),             \
                                            (T) ? bs1 : bs0,                   \
                                            _mm512_loadu_ps(o)));              \
      }                                                                        \
    } while (0)
    ACC_EACH(Q4_STORE);
#undef Q4_STORE
  }
}

void fgm_gemm_q4g_vnni(int M, int N, int K, const int8_t *A, const float *a_scale,
                       const uint8_t *Bq, const uint16_t *b_scale, int group,
                       float *C, int ldc, int n0, int n1) {
  const int KB = K / 64;
  const int gk = group / 64;
  const int NG = KB / gk;
  const int npanel = panel_nblocks(K, 1);
  const int MB = (M + 15) / 16;

  for (int np = n0; np < n1; np += npanel) {
    const int ne = (np + npanel < n1) ? np + npanel : n1;
    for (int mb = 0; mb < MB; mb++) {
      int32_t asum[NG][16];
      for (int g = 0; g < NG; g++)
        for (int m = 0; m < 16; m++)
          asum[g][m] = row_sum(
              A + ((size_t)mb * KB + (size_t)g * gk) * 1024 + m * 64, gk, 1024);
      for (int m0 = 0; m0 < 16; m0 += 8) {
        if (mb * 16 + m0 >= M) break;
        for (int nb = np; nb + 1 < ne; nb += 2)
          q4g_block(M, N, K, A, Bq, b_scale, gk, NG, C, ldc, mb, m0, nb, asum);
      }
    }
  }

  for (int m = 0; m < M; m++) {  // fold per-row activation scale
    __m512 s = _mm512_set1_ps(a_scale[m]);
    float *row = C + (size_t)m * ldc;
    for (int n = n0 * 16; n < n1 * 16; n += 16)
      _mm512_storeu_ps(row + n, _mm512_mul_ps(_mm512_loadu_ps(row + n), s));
  }
}
