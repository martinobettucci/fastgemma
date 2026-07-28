// Throughput sweep for the AMX GEMM kernels, with a persistent thread pool so
// the measurement reflects the kernel and not pthread_create.
//
// Sweeps M (decode batch 8 through prefill chunk 512) against the int4 group
// size, on the E2B FFN shape. The int4 drain overhead scales as 1/group, so the
// crossover this prints is what picks the shipped group size.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

int fgm_amx_init(void);
void fgm_pack_a(int,int,const int8_t*,int8_t*);
void fgm_gemm_q4g(int, int, int, const int8_t *, const float *, const uint8_t *,
                  const _Float16 *, int, float *, int, int, int);
void fgm_gemm_q8c(int, int, int, const int8_t *, const float *, const int8_t *,
                  const float *, float *, int, int, int);

static double now(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

// ------------------------------------------------------------- thread pool
typedef struct {
  int M, N, K, group, q4;
  const int8_t *A; const float *as;
  const void *B; const void *bs;
  float *C; int ldc;
} work_t;

static work_t W;
static int NT, n_per;
static pthread_barrier_t bar_start, bar_done;
static volatile int stop_flag;

static void *pool_worker(void *arg) {
  long id = (long)arg;
  fgm_amx_init();
  for (;;) {
    pthread_barrier_wait(&bar_start);
    if (stop_flag) return NULL;
    int nblocks = W.N / 16;
    int n0 = id * n_per;
    int n1 = (id == NT - 1) ? nblocks : (int)(id + 1) * n_per;
    if (n0 < n1) {
      if (W.q4)
        fgm_gemm_q4g(W.M, W.N, W.K, W.A, W.as, (const uint8_t *)W.B,
                     (const _Float16 *)W.bs, W.group, W.C, W.ldc, n0, n1);
      else
        fgm_gemm_q8c(W.M, W.N, W.K, W.A, W.as, (const int8_t *)W.B,
                     (const float *)W.bs, W.C, W.ldc, n0, n1);
    }
    pthread_barrier_wait(&bar_done);
  }
}

static double run(work_t w, double min_seconds) {
  W = w;
  int nblocks = w.N / 16;
  n_per = nblocks / NT / 4 * 4;
  if (n_per == 0) n_per = 4;
  // warm-up
  pthread_barrier_wait(&bar_start); pthread_barrier_wait(&bar_done);
  int reps = 0; double t0 = now(), el;
  do {
    pthread_barrier_wait(&bar_start);
    pthread_barrier_wait(&bar_done);
    reps++;
    el = now() - t0;
  } while (el < min_seconds);
  return el / reps;
}

int main(int argc, char **argv) {
  NT = argc > 1 ? atoi(argv[1]) : 4;
  int N = argc > 2 ? atoi(argv[2]) : 6144;
  int K = argc > 3 ? atoi(argv[3]) : 1536;
  if (!fgm_amx_init()) { printf("AMX permission FAILED\n"); return 1; }

  int NB = N / 16, KB = K / 64, MMAX = 512;
  int8_t *A = aligned_alloc(64, (size_t)MMAX * K);
  float *as = malloc(MMAX * sizeof(float));
  uint8_t *Bq = aligned_alloc(64, (size_t)NB * KB * 512);
  int8_t *B8 = aligned_alloc(64, (size_t)NB * KB * 1024);
  float *bs8 = aligned_alloc(64, (size_t)N * sizeof(float));
  float *C = aligned_alloc(64, (size_t)MMAX * N * sizeof(float));
  for (size_t i = 0; i < (size_t)MMAX * K; i++) A[i] = (int8_t)(i * 7 % 61 - 30);
  for (size_t i = 0; i < (size_t)NB * KB * 512; i++) Bq[i] = (uint8_t)(i * 13 % 251);
  for (size_t i = 0; i < (size_t)NB * KB * 1024; i++) B8[i] = (int8_t)(i * 11 % 61 - 30);
  for (int i = 0; i < MMAX; i++) as[i] = 0.01f;
  for (int i = 0; i < N; i++) bs8[i] = 0.01f;
  int8_t *Ap = aligned_alloc(64, (size_t)MMAX * K);
  fgm_pack_a(MMAX, K, A, Ap);

  pthread_barrier_init(&bar_start, NULL, NT + 1);
  pthread_barrier_init(&bar_done, NULL, NT + 1);
  pthread_t th[64];
  for (long t = 0; t < NT; t++) pthread_create(&th[t], NULL, pool_worker, (void *)t);

  int Ms[] = {8, 16, 32, 64, 128, 256, 512};
  int groups[] = {64, 128, 256, 512};
  printf("AMX GEMM, %d threads, N=%d K=%d  (TOPS; DRAM roofline for int4 = 2*M*33e9/1e12)\n",
         NT, N, K);
  printf("  %-5s %-9s", "M", "q8c");
  for (unsigned g = 0; g < 4; g++) printf(" q4g/g%-6d", groups[g]);
  printf(" %-9s\n", "int4 roof");

  for (unsigned mi = 0; mi < sizeof(Ms) / sizeof(int); mi++) {
    int M = Ms[mi];
    double ops = 2.0 * M * N * K;
    printf("  %-5d", M);
    work_t w = {M, N, K, 0, 0, Ap, as, B8, bs8, C, N};
    printf(" %-9.2f", ops / run(w, 0.5) / 1e12);
    for (unsigned gi = 0; gi < 4; gi++) {
      _Float16 *bs4 = aligned_alloc(64, (size_t)(K / groups[gi]) * N * sizeof(_Float16));
      for (int i = 0; i < (K / groups[gi]) * N; i++) bs4[i] = (_Float16)0.01f;
      work_t w4 = {M, N, K, groups[gi], 1, Ap, as, Bq, bs4, C, N};
      printf(" %-10.2f", ops / run(w4, 0.5) / 1e12);
      free(bs4);
    }
    printf(" %-9.2f\n", 2.0 * M * 33e9 / 1e12);
  }

  stop_flag = 1;
  pthread_barrier_wait(&bar_start);
  for (int t = 0; t < NT; t++) pthread_join(th[t], NULL);
  return 0;
}
