// Correctness + throughput harness for the AMX GEMM kernels.
// Verifies the tiled kernels against the scalar references, then sweeps
// (M, group) to find where the int4 drain overhead stops hurting.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>

int fgm_amx_init(void);
void fgm_pack_a(int,int,const int8_t*,int8_t*);
void fgm_pack_a(int,int,const int8_t*,int8_t*);
void fgm_gemm_q4g(int, int, int, const int8_t *, const float *, const uint8_t *,
                  const _Float16 *, int, float *, int, int, int);
void fgm_gemm_q8c(int, int, int, const int8_t *, const float *, const int8_t *,
                  const float *, float *, int, int, int);
void fgm_gemm_ref_q4g(int, int, int, const int8_t *, const float *, const uint8_t *,
                      const _Float16 *, int, float *, int);
void fgm_gemm_ref_q8c(int, int, int, const int8_t *, const float *, const int8_t *,
                      const float *, float *, int);

static double now(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static uint32_t rs = 12345;
static int rnd(int lo, int hi) { rs = rs * 1103515245u + 12345u; return lo + (int)((rs >> 16) % (unsigned)(hi - lo + 1)); }

static double relerr(const float *a, const float *b, int n) {
  double num = 0, den = 0;
  for (int i = 0; i < n; i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
  return sqrt(num / (den + 1e-30));
}

// ---- threading -------------------------------------------------------------
typedef struct {
  int M, N, K, group, q4;
  const int8_t *A; const float *as;
  const void *B; const void *bs;
  float *C; int ldc, n0, n1;
} job_t;

static void *worker(void *p) {
  job_t *j = (job_t *)p;
  fgm_amx_init();
  if (j->q4)
    fgm_gemm_q4g(j->M, j->N, j->K, j->A, j->as, (const uint8_t *)j->B,
                 (const _Float16 *)j->bs, j->group, j->C, j->ldc, j->n0, j->n1);
  else
    fgm_gemm_q8c(j->M, j->N, j->K, j->A, j->as, (const int8_t *)j->B,
                 (const float *)j->bs, j->C, j->ldc, j->n0, j->n1);
  return NULL;
}

static double run_mt(job_t base, int nthreads, int reps) {
  int nblocks = base.N / 16;
  int per = nblocks / nthreads;
  per = per / 4 * 4;                      // both paths step n in multiples of 4
  if (per == 0) { per = 4; nthreads = nblocks / 4 ? nblocks / 4 : 1; }
  pthread_t th[64]; job_t jobs[64];
  double t0 = now();
  for (int r = 0; r < reps; r++) {
    for (int t = 0; t < nthreads; t++) {
      jobs[t] = base;
      jobs[t].n0 = t * per;
      jobs[t].n1 = (t == nthreads - 1) ? nblocks : (t + 1) * per;
      pthread_create(&th[t], NULL, worker, &jobs[t]);
    }
    for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
  }
  return (now() - t0) / reps;
}

int main(int argc, char **argv) {
  int nthreads = argc > 1 ? atoi(argv[1]) : 4;
  if (!fgm_amx_init()) { printf("AMX permission FAILED\n"); return 1; }

  // ---------------------------------------------------------- correctness
  printf("== correctness (vs scalar reference) ==\n");
  {
    int M = 40, N = 64, K = 256, group = 64;
    int KB = K / 64, NB = N / 16;
    int8_t *A = aligned_alloc(64, (size_t)M * K);
    float *as = malloc(M * sizeof(float));
    uint8_t *Bq = aligned_alloc(64, (size_t)NB * KB * 512);
    int8_t *B8 = aligned_alloc(64, (size_t)NB * KB * 1024);
    _Float16 *bs4 = aligned_alloc(64, (size_t)(K / group) * N * sizeof(_Float16));
    float *bs8 = aligned_alloc(64, (size_t)N * sizeof(float));
    float *C = aligned_alloc(64, (size_t)M * N * sizeof(float));
    float *R = aligned_alloc(64, (size_t)M * N * sizeof(float));

    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int8_t)rnd(-127, 127);
    for (int i = 0; i < M; i++) as[i] = 0.002f + 0.001f * (i % 7);
    for (size_t i = 0; i < (size_t)NB * KB * 512; i++) {
      int lo = rnd(-8, 7) & 0xF, hi = rnd(-8, 7) & 0xF;
      Bq[i] = (uint8_t)(lo | (hi << 4));
    }
    for (size_t i = 0; i < (size_t)NB * KB * 1024; i++) B8[i] = (int8_t)rnd(-127, 127);
    for (int i = 0; i < (K / group) * N; i++) bs4[i] = (_Float16)(0.01f + 0.001f * (i % 5));
    for (int i = 0; i < N; i++) bs8[i] = 0.003f + 0.0005f * (i % 11);

    int8_t *Ap = aligned_alloc(64,(size_t)((M+15)/16)*16*K); fgm_pack_a(M,K,A,Ap);
    int8_t *Ap8 = aligned_alloc(64,(size_t)((8+15)/16)*16*K); fgm_pack_a(8,K,A,Ap8);
    { unsigned long ca=0,cb=0,cs=0;
      for(size_t i=0;i<(size_t)M*K;i++) ca=ca*31+(unsigned char)A[i];
      for(size_t i=0;i<(size_t)NB*KB*512;i++) cb=cb*31+Bq[i];
      for(int i=0;i<(K/group)*N;i++) cs=cs*31+(unsigned long)((float)bs4[i]*1e6f);
      printf("  [chk] A=%lu Bq=%lu bs4=%lu as0=%g\n",ca,cb,cs,as[0]); }
    fgm_gemm_ref_q4g(M, N, K, A, as, Bq, bs4, group, R, N);
    fgm_gemm_q4g(M, N, K, Ap, as, Bq, bs4, group, C, N, 0, NB);
    { int bad=0; for(int i=0;i<M*N && bad<4;i++) if(fabsf(C[i]-R[i])>1e-3f*(fabsf(R[i])+1e-6f)){
        printf("  [diff] idx=%d (m=%d,n=%d) ref=%.5f amx=%.5f\n",i,i/N,i%N,R[i],C[i]); bad++; } }
    printf("  q4g M=%d N=%d K=%d g=%d  rel_err = %.3e  %s\n", M, N, K, group,
           relerr(C, R, M * N), relerr(C, R, M * N) < 1e-5 ? "PASS" : "FAIL");

    fgm_gemm_ref_q8c(M, N, K, A, as, B8, bs8, R, N);
    fgm_gemm_q8c(M, N, K, Ap, as, B8, bs8, C, N, 0, NB);
    printf("  q8c M=%d N=%d K=%d      rel_err = %.3e  %s\n", M, N, K,
           relerr(C, R, M * N), relerr(C, R, M * N) < 1e-5 ? "PASS" : "FAIL");

    // decode-shaped case (M=8, single tile, 4-N path)
    int M2 = 8;
    fgm_gemm_ref_q4g(M2, N, K, A, as, Bq, bs4, group, R, N);
    fgm_gemm_q4g(M2, N, K, Ap8, as, Bq, bs4, group, C, N, 0, NB);
    printf("  q4g M=8  (decode shape)     rel_err = %.3e  %s\n",
           relerr(C, R, M2 * N), relerr(C, R, M2 * N) < 1e-5 ? "PASS" : "FAIL");
    free(A); free(as); free(Bq); free(B8); free(bs4); free(bs8); free(C); free(R);
  }

  // ---------------------------------------------------------- throughput
  printf("\n== throughput, %d threads (N=6144 K=1536, an E2B FFN shape) ==\n", nthreads);
  int N = 6144, K = 1536, NB = N / 16, KB = K / 64;
  int8_t *A = aligned_alloc(64, (size_t)512 * K);
  float *as = malloc(512 * sizeof(float));
  uint8_t *Bq = aligned_alloc(64, (size_t)NB * KB * 512);
  int8_t *B8 = aligned_alloc(64, (size_t)NB * KB * 1024);
  float *bs8 = aligned_alloc(64, (size_t)N * sizeof(float));
  float *C = aligned_alloc(64, (size_t)512 * N * sizeof(float));
  memset(A, 3, (size_t)512 * K);
  memset(Bq, 0x35, (size_t)NB * KB * 512);
  memset(B8, 5, (size_t)NB * KB * 1024);
  for (int i = 0; i < 512; i++) as[i] = 0.01f;
  for (int i = 0; i < N; i++) bs8[i] = 0.01f;

  int Ms[] = {8, 16, 32, 64, 128, 256, 512};
  int groups[] = {64, 128, 256, 512, 1536};
  printf("  %-6s %-10s", "M", "q8c");
  for (unsigned g = 0; g < sizeof(groups) / sizeof(int); g++) printf(" q4g/g=%-6d", groups[g]);
  printf("   (TOPS)\n");

  for (unsigned mi = 0; mi < sizeof(Ms) / sizeof(int); mi++) {
    int M = Ms[mi];
    double ops = 2.0 * M * N * K;
    int reps = M <= 32 ? 40 : (M <= 128 ? 15 : 6);
    printf("  %-6d", M);
    job_t j = {M, N, K, 0, 0, A, as, B8, bs8, C, N, 0, NB};
    double t = run_mt(j, nthreads, reps);
    printf(" %-10.2f", ops / t / 1e12);
    for (unsigned gi = 0; gi < sizeof(groups) / sizeof(int); gi++) {
      _Float16 *bs4 = aligned_alloc(64, (size_t)(K / groups[gi]) * N * sizeof(_Float16));
      for (int i = 0; i < (K / groups[gi]) * N; i++) bs4[i] = (_Float16)0.01f;
      job_t j4 = {M, N, K, groups[gi], 1, A, as, Bq, bs4, C, N, 0, NB};
      t = run_mt(j4, nthreads, reps);
      printf(" %-11.2f", ops / t / 1e12);
      free(bs4);
    }
    printf("\n");
  }
  return 0;
}
