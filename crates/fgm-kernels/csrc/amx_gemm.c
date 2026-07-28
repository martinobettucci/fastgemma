// AMX INT8 GEMM for fastgemma.
//
//   C[M,N] (f32) = (A[M,K] int8 * a_scale[M]) . (B[K,N] int4/int8 * b_scale)
//
// B arrives pre-packed in AMX tile order from the converter, so there is no
// runtime repacking: tiles are laid out [n_block][k_block], each tile 16 rows x
// 64 bytes holding (K=64, N=16) as [k/4][n][k%4] — the VNNI layout _tile_dpbssd
// consumes directly.
//
// Grouped int4 scales
// -------------------
// A scale group spans `group` K-values, so the int32 tile accumulator must be
// drained into an f32 accumulator every group/64 tile-steps. The drain costs
// O(MR*NR) regardless of group size while the AMX work it amortises is
// O(MR*NR*group) — so `group` is a direct speed/accuracy dial, benchmarked in
// bench/kernels. The drain is cheap by construction: a tile row is exactly one
// 16-lane int32 vector, and the 16 scales it needs are one contiguous f16 load
// (the converter stores scales transposed to [K/group, N]).
//
// int8 weights carry one scale per output channel, so they need no drain and
// run the accumulator across the whole K.
//
// Every N in this model is a multiple of 64 (4 tiles), so both paths use fixed
// unrolls — tile register indices must be compile-time constants anyway.

#define _GNU_SOURCE
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <immintrin.h>
#include <stdatomic.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

int fgm_amx_init(void) {
  return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
}

typedef struct {
  uint8_t palette_id, start_row, reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} tilecfg_t;

// large path: tmm0..3 acc (2M x 2N), tmm4..5 A, tmm6..7 B
static inline void cfg_2x2(int mr0, int mr1) {
  tilecfg_t c;
  memset(&c, 0, sizeof(c));
  c.palette_id = 1;
  c.rows[0] = mr0; c.colsb[0] = 64;
  c.rows[1] = mr0; c.colsb[1] = 64;
  c.rows[2] = mr1; c.colsb[2] = 64;
  c.rows[3] = mr1; c.colsb[3] = 64;
  c.rows[4] = mr0; c.colsb[4] = 64;
  c.rows[5] = mr1; c.colsb[5] = 64;
  c.rows[6] = 16;  c.colsb[6] = 64;
  c.rows[7] = 16;  c.colsb[7] = 64;
  _tile_loadconfig(&c);
}

// decode path: tmm0..3 acc (1M x 4N), tmm4 A, tmm5 B
static inline void cfg_1x4(int mr) {
  tilecfg_t c;
  memset(&c, 0, sizeof(c));
  c.palette_id = 1;
  for (int i = 0; i < 4; i++) { c.rows[i] = mr; c.colsb[i] = 64; }
  c.rows[4] = mr; c.colsb[4] = 64;
  c.rows[5] = 16; c.colsb[5] = 64;
  _tile_loadconfig(&c);
}

// ---------------------------------------------------------------- int4 unpack
// Canonical layout: within each 64-byte block, packed[i] (i<32) holds value i in
// the low nibble and value i+32 in the high nibble. One 32B load -> 64 int8.
static inline void unpack64(const uint8_t *src, int8_t *dst) {
  const __m256i m4 = _mm256_set1_epi8(0x0F), k8 = _mm256_set1_epi8(8);
  __m256i p = _mm256_loadu_si256((const __m256i *)src);
  __m256i lo = _mm256_and_si256(p, m4);
  __m256i hi = _mm256_and_si256(_mm256_srli_epi16(p, 4), m4);
  lo = _mm256_sub_epi8(_mm256_xor_si256(lo, k8), k8);  // 4-bit sign extend
  hi = _mm256_sub_epi8(_mm256_xor_si256(hi, k8), k8);
  _mm256_storeu_si256((__m256i *)dst, lo);
  _mm256_storeu_si256((__m256i *)(dst + 32), hi);
}

// The barrier is load-bearing: _tile_loadd is opaque to GCC's alias analysis,
// so at -O3 it will happily sink or eliminate the stores below it. Without this,
// the int4 path silently produces garbage at -O3 while passing at -O2.
static inline void unpack_tile(const uint8_t *src, int8_t *dst) {
  for (int r = 0; r < 16; r++) unpack64(src + r * 32, dst + r * 64);
  __asm__ __volatile__("" : : "r"(dst) : "memory");
}

// ---------------------------------------------------------------- A packing
// A tiles must be contiguous. Loading a 16x64 tile straight out of row-major
// A[M,K] means 16 cache lines K bytes apart (24 KB of stride for K=1536), and
// the resulting tile-load stalls cost more than the dpbssd they feed — measured
// at ~5x off peak. Packing A once per GEMM into [m16][kb][16][64] costs O(M*K)
// against the GEMM's O(M*N*K), and makes every tile load a flat 1 KB read.
void fgm_pack_a(int M, int K, const int8_t *A, int8_t *Ap) {
  const int KB = K / 64, MB = (M + 15) / 16;
  for (int mb = 0; mb < MB; mb++)
    for (int kb = 0; kb < KB; kb++) {
      int8_t *d = Ap + ((size_t)mb * KB + kb) * 1024;
      for (int r = 0; r < 16; r++) {
        int m = mb * 16 + r;
        if (m < M) memcpy(d + r * 64, A + (size_t)m * K + kb * 64, 64);
        else memset(d + r * 64, 0, 64);
      }
    }
}

// acc_f32[m][n] += (float)acc_i32[m][n] * bs[n], one 16-wide n block
static inline void drain16(const int32_t *ai, int mr, float *af, int ldaf,
                           const _Float16 *bs, int32_t bias) {
  __m512 s = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)bs));
  __m512i b = _mm512_set1_epi32(bias);
  for (int m = 0; m < mr; m++) {
    __m512i v = _mm512_sub_epi32(_mm512_loadu_si512(ai + m * 16), b);
    __m512 f = _mm512_cvtepi32_ps(v);
    _mm512_storeu_ps(af + m * ldaf,
                     _mm512_fmadd_ps(f, s, _mm512_loadu_ps(af + m * ldaf)));
  }
}


// ------------------------------------------------------- tile-state sentinel
// This VM does not preserve XTILEDATA across context switches: tiles come back
// ZEROED (see bench/platform/amx_tilestate.c -- 16% loss for state held 1 ms
// under load, 89% at 10 ms). An accumulator held across a k-loop therefore
// silently loses everything summed before the switch, and the result looks
// plausible rather than obviously broken.
//
// Detection costs no tile register and no blocking: seed each accumulator with
// a large constant instead of zeroing it. An intact tile reads back as
// SENTINEL + sum; a tile that was reset reads back as just the partial sum.
//
//   SENTINEL   = 2^30                    = 1073741824
//   |sum|     <= Kmax * 127 * 127        =  198193152  (K = 12288, worst case)
//
// so an intact accumulator is always > SENTINEL/2 and a reset one always well
// below it. One compare per stored tile detects the reset; the drain subtracts
// SENTINEL to recover the true sum. On detection the block is recomputed --
// per-block windows are sub-microsecond, so retries are rare and cheap.
//
// Set FGM_NO_TILE_GUARD=1 to disable, for measuring the guard's own cost.
#define SENTINEL 0x40000000
#define SENT_MIN 0x20000000  /* SENTINEL/2 */
#define MAX_RETRY 16

static const int32_t *acc_seed(void) {
  static _Alignas(64) int32_t seed[256];
  static int init = 0;
  if (!init) {
    for (int i = 0; i < 256; i++) seed[i] = SENTINEL;
    init = 1;
  }
  return seed;
}

static int guard_on = -1;
static inline int tile_guard(void) {
  if (guard_on < 0) {
    const char *e = getenv("FGM_NO_TILE_GUARD");
    guard_on = !(e && e[0] == '1');
  }
  return guard_on;
}

// A reset tile has lost the seed, so every element is small.
static inline int tile_intact(const int32_t *acc, int mr) {
  for (int m = 0; m < mr; m++)
    if (acc[m * 16] < SENT_MIN) return 0;
  return 1;
}

// Number of corruptions detected and recovered, for reporting.
static _Atomic long fgm_tile_retries = 0;
long fgm_tile_retry_count(void) { return atomic_load(&fgm_tile_retries); }
void fgm_tile_retry_reset(void) { atomic_store(&fgm_tile_retries, 0); }

// Seed an accumulator: SENTINEL everywhere when guarding, else plain zero.
#define ACC_SEED(T)                                                            \
  do {                                                                         \
    if (guarded) _tile_loadd(T, seedp, 64);                                    \
    else _tile_zero(T);                                                        \
  } while (0)

#define PK_TILE 512   // packed int4 tile bytes
#define Q8_TILE 1024  // int8 tile bytes

// ============================================================= int4 grouped
// A: [M,K] int8 row-major (K % 64 == 0). Bq: packed tiles [nb][kb].
// b_scale: [K/group][N] f16. C: [M,N] f32. n0/n1: n-block range (multiple of 4).
// Panel blocking: the B panel for one n-panel is sized to sit in L2, and all M
// rows are streamed through it before moving on. Without this, B is re-read from
// DRAM once per 32-row M block, which at 33 GB/s caps the kernel far below the
// 14.7 TOPS AMX ceiling. M is split into whole 32-row blocks plus a tail so the
// tile configuration is set once per panel rather than once per M block.
static inline int panel_nblocks(int K, int bytes_per_weight_x2) {
  // target ~256 KB of B per panel: nb * 16 cols * K * (0.5 or 1) bytes
  int nb = (int)(262144L * 2 / ((long)bytes_per_weight_x2 * 16 * K));
  if (nb < 4) nb = 4;
  return nb / 4 * 4;
}

#define INNER_Q4_2x2(MB, MR0, MR1)                                                     \
  do {                                                                                 \
    float *C0 = C + (size_t)(MB) * ldc + nb * 16;                                       \
    float *C1 = C + (size_t)((MB) + 16) * ldc + nb * 16;                                \
    for (int att = 0; att < MAX_RETRY; att++) {                                          \
      int bad = 0;                                                                       \
      for (int m = 0; m < (MR0); m++) memset(C0 + (size_t)m * ldc, 0, 2 * 64);            \
      for (int m = 0; m < (MR1); m++) memset(C1 + (size_t)m * ldc, 0, 2 * 64);            \
      for (int kb0 = 0; kb0 < KB && !bad; kb0 += gk) {                                    \
        ACC_SEED(0); ACC_SEED(1); ACC_SEED(2); ACC_SEED(3);                               \
        for (int kb = kb0; kb < kb0 + gk; kb++) {                                          \
          const uint8_t *bp = Bq + (size_t)nb * nbs + (size_t)kb * PK_TILE;                \
          _tile_loadd(4, A + (((size_t)(MB) / 16 * KB) + kb) * 1024, 64);                  \
          _tile_loadd(5, A + (((size_t)((MB) + 16) / 16 * KB) + kb) * 1024, 64);           \
          unpack_tile(bp, bt[0]);                                                          \
          _tile_loadd(6, bt[0], 64); _tile_dpbssd(0, 4, 6); _tile_dpbssd(2, 5, 6);         \
          unpack_tile(bp + nbs, bt[1]);                                                    \
          _tile_loadd(7, bt[1], 64); _tile_dpbssd(1, 4, 7); _tile_dpbssd(3, 5, 7);         \
        }                                                                                  \
        const _Float16 *bs = b_scale + (size_t)(kb0 / gk) * N + nb * 16;                   \
        _tile_stored(0, acc[0], 64); _tile_stored(1, acc[1], 64);                          \
        _tile_stored(2, acc[2], 64); _tile_stored(3, acc[3], 64);                          \
        if (guarded && !(tile_intact(acc[0], (MR0)) && tile_intact(acc[1], (MR0)) &&       \
                         tile_intact(acc[2], (MR1)) && tile_intact(acc[3], (MR1)))) {      \
          bad = 1; break;                                                                  \
        }                                                                                  \
        drain16(acc[0], (MR0), C0,      ldc, bs,      bias);                               \
        drain16(acc[1], (MR0), C0 + 16, ldc, bs + 16, bias);                               \
        drain16(acc[2], (MR1), C1,      ldc, bs,      bias);                               \
        drain16(acc[3], (MR1), C1 + 16, ldc, bs + 16, bias);                               \
      }                                                                                    \
      if (!bad) break;                                                                     \
      atomic_fetch_add(&fgm_tile_retries, 1);                                              \
    }                                                                                      \
  } while (0)

#define INNER_Q4_1x4(MB, MR)                                                            \
  do {                                                                                   \
    float *Cb = C + (size_t)(MB) * ldc + nb * 16;                                         \
    for (int att = 0; att < MAX_RETRY; att++) {                                           \
      int bad = 0;                                                                        \
      for (int m = 0; m < (MR); m++) memset(Cb + (size_t)m * ldc, 0, 4 * 64);             \
      for (int kb0 = 0; kb0 < KB && !bad; kb0 += gk) {                                    \
        ACC_SEED(0); ACC_SEED(1); ACC_SEED(2); ACC_SEED(3);                               \
        for (int kb = kb0; kb < kb0 + gk; kb++) {                                          \
          const uint8_t *bp = Bq + (size_t)nb * nbs + (size_t)kb * PK_TILE;                \
          _tile_loadd(4, A + (((size_t)(MB) / 16 * KB) + kb) * 1024, 64);                  \
          unpack_tile(bp, bt[0]);           _tile_loadd(5, bt[0], 64); _tile_dpbssd(0, 4, 5);\
          unpack_tile(bp + nbs, bt[1]);     _tile_loadd(5, bt[1], 64); _tile_dpbssd(1, 4, 5);\
          unpack_tile(bp + 2 * nbs, bt[2]); _tile_loadd(5, bt[2], 64); _tile_dpbssd(2, 4, 5);\
          unpack_tile(bp + 3 * nbs, bt[3]); _tile_loadd(5, bt[3], 64); _tile_dpbssd(3, 4, 5);\
        }                                                                                  \
        const _Float16 *bs = b_scale + (size_t)(kb0 / gk) * N + nb * 16;                   \
        _tile_stored(0, acc[0], 64); _tile_stored(1, acc[1], 64);                          \
        _tile_stored(2, acc[2], 64); _tile_stored(3, acc[3], 64);                          \
        if (guarded && !(tile_intact(acc[0], (MR)) && tile_intact(acc[1], (MR)) &&         \
                         tile_intact(acc[2], (MR)) && tile_intact(acc[3], (MR)))) {        \
          bad = 1; break;                                                                  \
        }                                                                                  \
        drain16(acc[0], (MR), Cb,      ldc, bs,      bias);                                \
        drain16(acc[1], (MR), Cb + 16, ldc, bs + 16, bias);                                \
        drain16(acc[2], (MR), Cb + 32, ldc, bs + 32, bias);                                \
        drain16(acc[3], (MR), Cb + 48, ldc, bs + 48, bias);                                \
      }                                                                                    \
      if (!bad) break;                                                                     \
      atomic_fetch_add(&fgm_tile_retries, 1);                                              \
    }                                                                                      \
  } while (0)

void fgm_gemm_q4g(int M, int N, int K, const int8_t *A, const float *a_scale,
                  const uint8_t *Bq, const _Float16 *b_scale, int group,
                  float *C, int ldc, int n0, int n1) {
  const int KB = K / 64;
  const int gk = group / 64;
  const size_t nbs = (size_t)KB * PK_TILE;
  _Alignas(64) int8_t bt[4][1024];
  _Alignas(64) int32_t acc[4][256];

  const int guarded = tile_guard();
  const int32_t *seedp = acc_seed();
  const int32_t bias = guarded ? SENTINEL : 0;
  const int npanel = panel_nblocks(K, 1);   // int4: 0.5 byte/weight
  const int M32 = M & ~31;
  const int rem = M - M32;

  for (int np = n0; np < n1; np += npanel) {
    const int ne = (np + npanel < n1) ? np + npanel : n1;
    if (M32) {
      cfg_2x2(16, 16);
      for (int mb = 0; mb < M32; mb += 32)
        for (int nb = np; nb < ne; nb += 2) INNER_Q4_2x2(mb, 16, 16);
    }
    if (rem > 16) {
      cfg_2x2(16, rem - 16);
      for (int nb = np; nb < ne; nb += 2) INNER_Q4_2x2(M32, 16, rem - 16);
    } else if (rem > 0) {
      cfg_1x4(rem);
      for (int nb = np; nb < ne; nb += 4) INNER_Q4_1x4(M32, rem);
    }
  }
  _tile_release();

  for (int m = 0; m < M; m++) {  // fold per-row activation scale
    __m512 s = _mm512_set1_ps(a_scale[m]);
    float *row = C + (size_t)m * ldc;
    for (int n = n0 * 16; n < n1 * 16; n += 16)
      _mm512_storeu_ps(row + n, _mm512_mul_ps(_mm512_loadu_ps(row + n), s));
  }
}

// ============================================================= int8 per-channel
#define INNER_Q8_2x2(MB, MR0, MR1)                                                      \
  do {                                                                                   \
    const int8_t *bp0 = B + (size_t)nb * nbs;                                            \
    for (int att = 0; att < MAX_RETRY; att++) {                                           \
    ACC_SEED(0); ACC_SEED(1); ACC_SEED(2); ACC_SEED(3);                                   \
    for (int kb = 0; kb < KB; kb++) {                                                     \
      const int8_t *bp = bp0 + (size_t)kb * Q8_TILE;                                      \
      _tile_loadd(4, A + (((size_t)(MB) / 16 * KB) + kb) * 1024, 64);                     \
      _tile_loadd(5, A + (((size_t)((MB) + 16) / 16 * KB) + kb) * 1024, 64);              \
      _tile_loadd(6, bp, 64);       _tile_dpbssd(0, 4, 6); _tile_dpbssd(2, 5, 6);         \
      _tile_loadd(7, bp + nbs, 64); _tile_dpbssd(1, 4, 7); _tile_dpbssd(3, 5, 7);         \
    }                                                                                     \
    _tile_stored(0, acc[0], 64); _tile_stored(1, acc[1], 64);                             \
    _tile_stored(2, acc[2], 64); _tile_stored(3, acc[3], 64);                             \
    if (guarded && !(tile_intact(acc[0], (MR0)) && tile_intact(acc[1], (MR0)) &&          \
                     tile_intact(acc[2], (MR1)) && tile_intact(acc[3], (MR1)))) {         \
      atomic_fetch_add(&fgm_tile_retries, 1); continue;                                   \
    }                                                                                     \
    for (int t = 0; t < 2; t++) {                                                          \
      __m512 bs = _mm512_loadu_ps(b_scale + (nb + t) * 16);                                \
      for (int m = 0; m < (MR0); m++) {                                                     \
        __m512i v = _mm512_sub_epi32(_mm512_loadu_si512(acc[t] + m * 16), vbias);           \
        __m512 f = _mm512_cvtepi32_ps(v);                                                   \
        _mm512_storeu_ps(C + (size_t)((MB) + m) * ldc + (nb + t) * 16,                      \
            _mm512_mul_ps(f, _mm512_mul_ps(bs, _mm512_set1_ps(a_scale[(MB) + m]))));        \
      }                                                                                     \
      for (int m = 0; m < (MR1); m++) {                                                     \
        __m512i v = _mm512_sub_epi32(_mm512_loadu_si512(acc[2 + t] + m * 16), vbias);       \
        __m512 f = _mm512_cvtepi32_ps(v);                                                   \
        _mm512_storeu_ps(C + (size_t)((MB) + 16 + m) * ldc + (nb + t) * 16,                 \
            _mm512_mul_ps(f, _mm512_mul_ps(bs, _mm512_set1_ps(a_scale[(MB) + 16 + m]))));   \
      }                                                                                     \
    }                                                                                       \
    break;                                                                                  \
    }                                                                                       \
  } while (0)

#define INNER_Q8_1x4(MB, MR)                                                              \
  do {                                                                                     \
    const int8_t *bp0 = B + (size_t)nb * nbs;                                              \
    for (int att = 0; att < MAX_RETRY; att++) {                                             \
    ACC_SEED(0); ACC_SEED(1); ACC_SEED(2); ACC_SEED(3);                                     \
    for (int kb = 0; kb < KB; kb++) {                                                       \
      const int8_t *bp = bp0 + (size_t)kb * Q8_TILE;                                        \
      _tile_loadd(4, A + (((size_t)(MB) / 16 * KB) + kb) * 1024, 64);                       \
      _tile_loadd(5, bp, 64);           _tile_dpbssd(0, 4, 5);                              \
      _tile_loadd(5, bp + nbs, 64);     _tile_dpbssd(1, 4, 5);                              \
      _tile_loadd(5, bp + 2 * nbs, 64); _tile_dpbssd(2, 4, 5);                              \
      _tile_loadd(5, bp + 3 * nbs, 64); _tile_dpbssd(3, 4, 5);                              \
    }                                                                                       \
    _tile_stored(0, acc[0], 64); _tile_stored(1, acc[1], 64);                               \
    _tile_stored(2, acc[2], 64); _tile_stored(3, acc[3], 64);                               \
    if (guarded && !(tile_intact(acc[0], (MR)) && tile_intact(acc[1], (MR)) &&              \
                     tile_intact(acc[2], (MR)) && tile_intact(acc[3], (MR)))) {             \
      atomic_fetch_add(&fgm_tile_retries, 1); continue;                                     \
    }                                                                                       \
    for (int t = 0; t < 4; t++) {                                                            \
      __m512 bs = _mm512_loadu_ps(b_scale + (nb + t) * 16);                                  \
      for (int m = 0; m < (MR); m++) {                                                        \
        __m512i v = _mm512_sub_epi32(_mm512_loadu_si512(acc[t] + m * 16), vbias);             \
        __m512 f = _mm512_cvtepi32_ps(v);                                                     \
        _mm512_storeu_ps(C + (size_t)((MB) + m) * ldc + (nb + t) * 16,                        \
            _mm512_mul_ps(f, _mm512_mul_ps(bs, _mm512_set1_ps(a_scale[(MB) + m]))));          \
      }                                                                                       \
    }                                                                                         \
    break;                                                                                    \
    }                                                                                         \
  } while (0)

void fgm_gemm_q8c(int M, int N, int K, const int8_t *A, const float *a_scale,
                  const int8_t *B, const float *b_scale, float *C, int ldc,
                  int n0, int n1) {
  const int KB = K / 64;
  const size_t nbs = (size_t)KB * Q8_TILE;
  _Alignas(64) int32_t acc[4][256];
  (void)N;

  const int guarded = tile_guard();
  const int32_t *seedp = acc_seed();
  const __m512i vbias = _mm512_set1_epi32(guarded ? SENTINEL : 0);
  const int npanel = panel_nblocks(K, 2);   // int8: 1 byte/weight
  const int M32 = M & ~31;
  const int rem = M - M32;

  for (int np = n0; np < n1; np += npanel) {
    const int ne = (np + npanel < n1) ? np + npanel : n1;
    if (M32) {
      cfg_2x2(16, 16);
      for (int mb = 0; mb < M32; mb += 32)
        for (int nb = np; nb < ne; nb += 2) INNER_Q8_2x2(mb, 16, 16);
    }
    if (rem > 16) {
      cfg_2x2(16, rem - 16);
      for (int nb = np; nb < ne; nb += 2) INNER_Q8_2x2(M32, 16, rem - 16);
    } else if (rem > 0) {
      cfg_1x4(rem);
      for (int nb = np; nb < ne; nb += 4) INNER_Q8_1x4(M32, rem);
    }
  }
  _tile_release();
}

// ============================================================ reference (test)
void fgm_gemm_ref_q4g(int M, int N, int K, const int8_t *A, const float *a_scale,
                      const uint8_t *Bq, const _Float16 *b_scale, int group,
                      float *C, int ldc) {
  const int KB = K / 64;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) C[(size_t)m * ldc + n] = 0.f;
  _Alignas(64) int8_t bt[1024];
  for (int nb = 0; nb < N / 16; nb++)
    for (int kb = 0; kb < KB; kb++) {
      unpack_tile(Bq + ((size_t)nb * KB + kb) * PK_TILE, bt);
      const int g = (kb * 64) / group;
      for (int m = 0; m < M; m++)
        for (int n = 0; n < 16; n++) {
          int32_t s = 0;
          for (int r = 0; r < 16; r++)
            for (int j = 0; j < 4; j++)
              s += (int32_t)A[(size_t)m * K + kb * 64 + r * 4 + j] * bt[r * 64 + n * 4 + j];
          C[(size_t)m * ldc + nb * 16 + n] +=
              (float)s * (float)b_scale[(size_t)g * N + nb * 16 + n];
        }
    }
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) C[(size_t)m * ldc + n] *= a_scale[m];
}

void fgm_gemm_ref_q8c(int M, int N, int K, const int8_t *A, const float *a_scale,
                      const int8_t *B, const float *b_scale, float *C, int ldc) {
  const int KB = K / 64;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) C[(size_t)m * ldc + n] = 0.f;
  for (int nb = 0; nb < N / 16; nb++)
    for (int kb = 0; kb < KB; kb++) {
      const int8_t *bt = B + ((size_t)nb * KB + kb) * Q8_TILE;
      for (int m = 0; m < M; m++)
        for (int n = 0; n < 16; n++) {
          int32_t s = 0;
          for (int r = 0; r < 16; r++)
            for (int j = 0; j < 4; j++)
              s += (int32_t)A[(size_t)m * K + kb * 64 + r * 4 + j] * bt[r * 64 + n * 4 + j];
          C[(size_t)m * ldc + nb * 16 + n] += (float)s;
        }
    }
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++)
      C[(size_t)m * ldc + n] *= a_scale[m] * b_scale[n];
}
