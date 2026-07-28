// Correctness + throughput for the AVX-512 VNNI GEMM fallback.
//
// This test carries its own scalar reference rather than calling
// fgm_gemm_ref_q4g from amx_gemm.c, on purpose: that file is compiled for
// Sapphire Rapids and will not run on the hosts this kernel exists to serve.
// The reference here is written straight from the layout definition, so it
// pins the VNNI kernel to the maths and not to the AMX kernel's opinion of it.
//
// Build:
//   gcc -O2 -march=cascadelake -fno-strict-aliasing vnni_test.c \
//       ../../crates/fgm-kernels/csrc/vnni_gemm.c \
//       ../../crates/fgm-kernels/csrc/ops.c -o vnni_test -lm -lpthread
//
// Run: ./vnni_test          (correctness)
//      ./vnni_test bench    (correctness, then throughput)

#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void fgm_pack_a(int M, int K, const int8_t *A, int8_t *Ap);
void fgm_gemm_q8c_vnni(int M, int N, int K, const int8_t *A, const float *a_scale,
                       const int8_t *B, const float *b_scale, float *C, int ldc,
                       int n0, int n1);
void fgm_gemm_q4g_vnni(int M, int N, int K, const int8_t *A, const float *a_scale,
                       const uint8_t *Bq, const uint16_t *b_scale, int group,
                       float *C, int ldc, int n0, int n1);
int fgm_cpu_has_amx(void);
int fgm_cpu_has_avx512_vnni(void);

#define PK_TILE 512
#define Q8_TILE 1024

static uint64_t rs = 0x243F6A8885A308D3ull;
static inline int rnd(int lo, int hi) {
  rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
  return lo + (int)(rs % (uint64_t)(hi - lo + 1));
}
static double now(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

// tile order: B[k][n] lives at tile(nb,kb)[ (k%64)/4 * 64 + (n%16)*4 + k%4 ]
static inline int bpos(int k, int n) { return (k % 64) / 4 * 64 + (n % 16) * 4 + k % 4; }

// ------------------------------------------------------------------ int8 ref
static void ref_q8c(int M, int N, int K, const int8_t *A, const float *as,
                    const int8_t *B, const float *bs, float *C, int ldc,
                    int n0, int n1) {
  const int KB = K / 64;
  for (int nb = n0; nb < n1; nb++)
    for (int n = nb * 16; n < nb * 16 + 16; n++)
      for (int m = 0; m < M; m++) {
        long acc = 0;
        for (int k = 0; k < K; k++)
          acc += (long)A[(size_t)m * K + k] *
                 B[(size_t)nb * KB * Q8_TILE + (size_t)(k / 64) * Q8_TILE + bpos(k, n)];
        C[(size_t)m * ldc + n] = (float)acc * as[m] * bs[n];
      }
}

// ------------------------------------------------------------------ int4 ref
static void ref_q4g(int M, int N, int K, const int8_t *A, const float *as,
                    const uint8_t *Bq, const uint16_t *bs, int group, float *C,
                    int ldc, int n0, int n1) {
  const int KB = K / 64;
  for (int nb = n0; nb < n1; nb++)
    for (int n = nb * 16; n < nb * 16 + 16; n++)
      for (int m = 0; m < M; m++) {
        // f32 accumulate with fma, in group order -- deliberately mirroring the
        // kernel's drain rather than summing in double. The drain is where the
        // int4 path loses precision (12 group partials of ~1e4 summing to a
        // small result cancel hard), and that loss belongs to the algorithm,
        // which the AMX path shares. Accumulating the reference in double would
        // measure that shared loss instead of measuring this kernel.
        float sum = 0;
        for (int g = 0; g * group < K; g++) {
          long acc = 0;
          for (int k = g * group; k < (g + 1) * group; k++) {
            int p = bpos(k, n), row = p / 64, o = p % 64;
            const uint8_t *tile = Bq + (size_t)nb * KB * PK_TILE + (size_t)(k / 64) * PK_TILE;
            // canonical half-split, per 64-byte block: unpacked value o of block
            // `row` is nibble (o<32 ? low : high) of packed byte row*32 + o%32
            int byte = tile[row * 32 + o % 32], v = (o < 32) ? (byte & 0xF) : (byte >> 4);
            acc += (long)A[(size_t)m * K + k] * (v - 8);
          }
          sum = fmaf((float)acc, _cvtsh_ss(bs[(size_t)g * N + n]), sum);
        }
        C[(size_t)m * ldc + n] = sum * as[m];
      }
}

static double maxrel(const float *a, const float *b, int M, int N, int ldc,
                     int n0, int n1) {
  double worst = 0;
  for (int m = 0; m < M; m++)
    for (int n = n0 * 16; n < n1 * 16; n++) {
      double x = a[(size_t)m * ldc + n], y = b[(size_t)m * ldc + n];
      double d = fabs(x - y), s = fabs(x) + fabs(y);
      if (s > 1e-6 && d / s > worst) worst = d / s;
    }
  (void)N;
  return worst;
}

static int fails = 0;

static void case_q8c(int M, int N, int K, int n0, int n1) {
  const int KB = K / 64, NB = N / 16;
  int8_t *A = malloc((size_t)M * K);
  int8_t *Ap = malloc((size_t)((M + 15) / 16) * 16 * K);
  int8_t *B = malloc((size_t)NB * KB * Q8_TILE);
  float *as = malloc((size_t)M * 4), *bs = malloc((size_t)N * 4);
  float *C = calloc((size_t)M * N, 4), *R = calloc((size_t)M * N, 4);
  for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int8_t)rnd(-127, 127);
  for (size_t i = 0; i < (size_t)NB * KB * Q8_TILE; i++) B[i] = (int8_t)rnd(-127, 127);
  for (int i = 0; i < M; i++) as[i] = 0.01f + 0.001f * (i % 7);
  for (int i = 0; i < N; i++) bs[i] = 0.02f + 0.0005f * (i % 11);
  fgm_pack_a(M, K, A, Ap);
  fgm_gemm_q8c_vnni(M, N, K, Ap, as, B, bs, C, N, n0, n1);
  ref_q8c(M, N, K, A, as, B, bs, R, N, n0, n1);
  double e = maxrel(C, R, M, N, N, n0, n1);
  int bad = !(e < 1e-5);
  fails += bad;
  printf("  q8c M=%-5d N=%-5d K=%-5d n[%d,%d)  max rel err %.2e  %s\n", M, N, K,
         n0, n1, e, bad ? "FAIL" : "ok");
  free(A); free(Ap); free(B); free(as); free(bs); free(C); free(R);
}

static void case_q4g(int M, int N, int K, int group, int n0, int n1) {
  const int KB = K / 64, NB = N / 16, NG = K / group;
  int8_t *A = malloc((size_t)M * K);
  int8_t *Ap = malloc((size_t)((M + 15) / 16) * 16 * K);
  uint8_t *B = malloc((size_t)NB * KB * PK_TILE);
  float *as = malloc((size_t)M * 4);
  uint16_t *bs = malloc((size_t)NG * N * 2);
  float *C = calloc((size_t)M * N, 4), *R = calloc((size_t)M * N, 4);
  for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int8_t)rnd(-127, 127);
  for (size_t i = 0; i < (size_t)NB * KB * PK_TILE; i++) B[i] = (uint8_t)rnd(0, 255);
  for (int i = 0; i < M; i++) as[i] = 0.01f + 0.001f * (i % 7);
  for (int i = 0; i < NG * N; i++) bs[i] = _cvtss_sh(0.002f + 0.0001f * (i % 13), 0);
  fgm_pack_a(M, K, A, Ap);
  fgm_gemm_q4g_vnni(M, N, K, Ap, as, B, bs, group, C, N, n0, n1);
  ref_q4g(M, N, K, A, as, B, bs, group, R, N, n0, n1);
  double e = maxrel(C, R, M, N, N, n0, n1);
  int bad = !(e < 1e-4);  // f32 accumulate order differs from the f64 reference
  fails += bad;
  printf("  q4g M=%-5d N=%-5d K=%-5d g=%-4d n[%d,%d)  max rel err %.2e  %s\n", M,
         N, K, group, n0, n1, e, bad ? "FAIL" : "ok");
  free(A); free(Ap); free(B); free(as); free(bs); free(C); free(R);
}

// Single-core vpdpbusd peak, measured rather than assumed.
//
// The 1.16 TOPS figure in the journal is an all-core number from the AMX host
// and means nothing here: this part issues about one vpdpbusd per cycle where
// that one issued two, so quoting it would make a good kernel look like a 6%
// one. Sixteen independent accumulators, no memory traffic, nothing to stall on.
static volatile int peak_sink;
static double vnni_peak(void) {
  __m512i a = _mm512_set1_epi8(3), b = _mm512_set1_epi8(2);
  // Hide the operands from the optimiser. vpdpbusd of two compile-time
  // constants folds to "add a constant vector", the loop collapses to a closed
  // form, and the peak comes out at 40 000 000 G MAC/s.
  __asm__ __volatile__("" : "+v"(a), "+v"(b));
  // Sixteen named accumulators, not an array: at -O2 GCC keeps an array of
  // sixteen zmm on the stack and the "peak" comes out at 40% of the real one,
  // which would then flatter every kernel measured against it.
  __m512i c0 = _mm512_setzero_si512(), c1 = c0, c2 = c0, c3 = c0, c4 = c0,
          c5 = c0, c6 = c0, c7 = c0, c8 = c0, c9 = c0, ca = c0, cb = c0,
          cc = c0, cd = c0, ce = c0, cf = c0;
  const long it = 8000000L;
  double t0 = now();
  for (long i = 0; i < it; i++) {
    c0 = _mm512_dpbusd_epi32(c0, a, b); c1 = _mm512_dpbusd_epi32(c1, a, b);
    c2 = _mm512_dpbusd_epi32(c2, a, b); c3 = _mm512_dpbusd_epi32(c3, a, b);
    c4 = _mm512_dpbusd_epi32(c4, a, b); c5 = _mm512_dpbusd_epi32(c5, a, b);
    c6 = _mm512_dpbusd_epi32(c6, a, b); c7 = _mm512_dpbusd_epi32(c7, a, b);
    c8 = _mm512_dpbusd_epi32(c8, a, b); c9 = _mm512_dpbusd_epi32(c9, a, b);
    ca = _mm512_dpbusd_epi32(ca, a, b); cb = _mm512_dpbusd_epi32(cb, a, b);
    cc = _mm512_dpbusd_epi32(cc, a, b); cd = _mm512_dpbusd_epi32(cd, a, b);
    ce = _mm512_dpbusd_epi32(ce, a, b); cf = _mm512_dpbusd_epi32(cf, a, b);
  }
  double d = now() - t0;
  __m512i s = _mm512_add_epi32(
      _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(c0, c1),
                                        _mm512_add_epi32(c2, c3)),
                       _mm512_add_epi32(_mm512_add_epi32(c4, c5),
                                        _mm512_add_epi32(c6, c7))),
      _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(c8, c9),
                                        _mm512_add_epi32(ca, cb)),
                       _mm512_add_epi32(_mm512_add_epi32(cc, cd),
                                        _mm512_add_epi32(ce, cf))));
  peak_sink = _mm512_reduce_add_epi32(s);  // keep the loop alive
  return it * 16.0 * 64.0 / d;
}

static void bench(void) {
  const int K = 1536, N = 2048, NB = N / 16, KB = K / 64;
  int8_t *B8 = malloc((size_t)NB * KB * Q8_TILE);
  uint8_t *B4 = malloc((size_t)NB * KB * PK_TILE);
  float *bs8 = malloc((size_t)N * 4);
  uint16_t *bs4 = malloc((size_t)(K / 128) * N * 2);
  for (size_t i = 0; i < (size_t)NB * KB * Q8_TILE; i++) B8[i] = (int8_t)rnd(-127, 127);
  for (size_t i = 0; i < (size_t)NB * KB * PK_TILE; i++) B4[i] = (uint8_t)rnd(0, 255);
  for (int i = 0; i < N; i++) bs8[i] = 0.02f;
  for (int i = 0; i < (K / 128) * N; i++) bs4[i] = _cvtss_sh(0.002f, 0);

  const double peak = vnni_peak();
  printf("\nsingle-core vpdpbusd peak: %.1f G MAC/s\n", peak / 1e9);
  printf("single-thread throughput (K=%d N=%d)\n", K, N);
  for (int mi = 0; mi < 4; mi++) {
    const int M = (int[]){8, 64, 256, 1024}[mi];
    int8_t *A = malloc((size_t)M * K), *Ap = malloc((size_t)((M + 15) / 16) * 16 * K);
    float *as = malloc((size_t)M * 4), *C = calloc((size_t)M * N, 4);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int8_t)rnd(-127, 127);
    for (int i = 0; i < M; i++) as[i] = 0.01f;
    fgm_pack_a(M, K, A, Ap);
    double macs = (double)M * N * K;
    for (int w = 0; w < 2; w++) {
      int reps = M >= 256 ? 9 : 40;
      double best = 1e30;
      for (int r = 0; r < reps; r++) {
        double t0 = now();
        if (w) fgm_gemm_q8c_vnni(M, N, K, Ap, as, B8, bs8, C, N, 0, NB);
        else fgm_gemm_q4g_vnni(M, N, K, Ap, as, B4, bs4, 128, C, N, 0, NB);
        double d = now() - t0;
        if (d < best) best = d;
      }
      printf("  M=%-5d %s  %6.1f G MAC/s  (%.0f%% of peak)\n", M,
             w ? "int8" : "int4", macs / best / 1e9, 100.0 * macs / best / peak);
    }
    free(A); free(Ap); free(as); free(C);
  }
  free(B8); free(B4); free(bs8); free(bs4);
}

int main(int argc, char **argv) {
  printf("cpu: amx=%d avx512_vnni=%d\n", fgm_cpu_has_amx(), fgm_cpu_has_avx512_vnni());
  if (!fgm_cpu_has_avx512_vnni()) { printf("no VNNI on this host\n"); return 77; }

  printf("\ncorrectness\n");
  // shapes that exercise every tail: M below/at/above a 16-row block, the
  // 8-row instantiation, partial n ranges, and both group sizes
  case_q8c(1, 256, 1536, 0, 16);
  case_q8c(8, 256, 1536, 0, 16);
  case_q8c(15, 256, 1536, 0, 16);
  case_q8c(16, 256, 1536, 0, 16);
  case_q8c(17, 256, 1536, 0, 16);
  case_q8c(33, 512, 1536, 4, 12);
  case_q8c(64, 256, 2048, 0, 16);
  case_q4g(1, 256, 1536, 128, 0, 16);
  case_q4g(8, 256, 1536, 128, 0, 16);
  case_q4g(15, 256, 1536, 64, 0, 16);
  case_q4g(16, 256, 1536, 256, 0, 16);
  case_q4g(17, 256, 1536, 128, 0, 16);
  case_q4g(33, 512, 1536, 128, 4, 12);
  case_q4g(64, 256, 2048, 128, 0, 16);

  if (argc > 1 && !strcmp(argv[1], "bench")) bench();
  printf("\n%s\n", fails ? "FAILURES" : "all ok");
  return fails ? 1 : 0;
}
