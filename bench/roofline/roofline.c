// Roofline probe: AMX-INT8 / AMX-BF16 / AVX512-VNNI peak + memory bandwidth.
// Establishes the hardware ceiling that every fastgemma kernel is measured against.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <immintrin.h>

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

static int amx_enable(void) {
  return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

typedef struct {
  uint8_t palette_id, start_row, reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} tilecfg_t;

static int NTHREADS = 4;

// ---------------------------------------------------------------- AMX INT8
static void *amx_i8_worker(void *arg) {
  long iters = *(long *)arg;
  if (amx_enable() != 0) return (void *)-1;

  tilecfg_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.palette_id = 1;
  // tmm0..tmm3 = accumulators (16x16 int32), tmm4/5 = A tiles (16x64 int8),
  // tmm6/7 = B tiles (16x64 int8, VNNI-packed as K/4 x N*4)
  for (int i = 0; i < 4; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
  for (int i = 4; i < 8; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
  _tile_loadconfig(&cfg);

  static _Alignas(64) int8_t A[2][16 * 64];
  static _Alignas(64) int8_t B[2][16 * 64];
  _Alignas(64) int32_t C[4][16 * 16];
  memset(A, 1, sizeof(A));
  memset(B, 2, sizeof(B));
  memset(C, 0, sizeof(C));

  _tile_loadd(0, C[0], 64); _tile_loadd(1, C[1], 64);
  _tile_loadd(2, C[2], 64); _tile_loadd(3, C[3], 64);
  _tile_loadd(4, A[0], 64); _tile_loadd(5, A[1], 64);
  _tile_loadd(6, B[0], 64); _tile_loadd(7, B[1], 64);

  for (long i = 0; i < iters; i++) {
    _tile_dpbssd(0, 4, 6);
    _tile_dpbssd(1, 4, 7);
    _tile_dpbssd(2, 5, 6);
    _tile_dpbssd(3, 5, 7);
  }
  _tile_stored(0, C[0], 64);
  _tile_release();
  volatile int sink = C[0][0]; (void)sink;
  return NULL;
}

// ---------------------------------------------------------------- AMX BF16
static void *amx_bf16_worker(void *arg) {
  long iters = *(long *)arg;
  if (amx_enable() != 0) return (void *)-1;
  tilecfg_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.palette_id = 1;
  for (int i = 0; i < 8; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
  _tile_loadconfig(&cfg);
  static _Alignas(64) uint16_t A[2][16 * 32];
  static _Alignas(64) uint16_t B[2][16 * 32];
  _Alignas(64) float C[4][16 * 16];
  memset(A, 0x3f, sizeof(A)); memset(B, 0x3f, sizeof(B)); memset(C, 0, sizeof(C));
  _tile_loadd(0, C[0], 64); _tile_loadd(1, C[1], 64);
  _tile_loadd(2, C[2], 64); _tile_loadd(3, C[3], 64);
  _tile_loadd(4, A[0], 64); _tile_loadd(5, A[1], 64);
  _tile_loadd(6, B[0], 64); _tile_loadd(7, B[1], 64);
  for (long i = 0; i < iters; i++) {
    _tile_dpbf16ps(0, 4, 6);
    _tile_dpbf16ps(1, 4, 7);
    _tile_dpbf16ps(2, 5, 6);
    _tile_dpbf16ps(3, 5, 7);
  }
  _tile_stored(0, C[0], 64);
  _tile_release();
  volatile float sink = C[0][0]; (void)sink;
  return NULL;
}

// ------------------------------------------------------------- AVX512 VNNI
static void *vnni_worker(void *arg) {
  long iters = *(long *)arg;
  __m512i a = _mm512_set1_epi8(1), b = _mm512_set1_epi8(2);
  __m512i c0 = _mm512_setzero_si512(), c1 = _mm512_setzero_si512();
  __m512i c2 = _mm512_setzero_si512(), c3 = _mm512_setzero_si512();
  __m512i c4 = _mm512_setzero_si512(), c5 = _mm512_setzero_si512();
  __m512i c6 = _mm512_setzero_si512(), c7 = _mm512_setzero_si512();
  for (long i = 0; i < iters; i++) {
    c0 = _mm512_dpbusd_epi32(c0, a, b); c1 = _mm512_dpbusd_epi32(c1, a, b);
    c2 = _mm512_dpbusd_epi32(c2, a, b); c3 = _mm512_dpbusd_epi32(c3, a, b);
    c4 = _mm512_dpbusd_epi32(c4, a, b); c5 = _mm512_dpbusd_epi32(c5, a, b);
    c6 = _mm512_dpbusd_epi32(c6, a, b); c7 = _mm512_dpbusd_epi32(c7, a, b);
  }
  c0 = _mm512_add_epi32(_mm512_add_epi32(c0, c1), _mm512_add_epi32(c2, c3));
  c4 = _mm512_add_epi32(_mm512_add_epi32(c4, c5), _mm512_add_epi32(c6, c7));
  volatile int sink = _mm512_reduce_add_epi32(_mm512_add_epi32(c0, c4)); (void)sink;
  return NULL;
}

// ------------------------------------------------------------- memory bandwidth
typedef struct { char *buf; size_t bytes; long reps; double gbs; } bwarg_t;

static void *bw_read_worker(void *arg) {
  bwarg_t *a = (bwarg_t *)arg;
  __m512i acc = _mm512_setzero_si512();
  for (long r = 0; r < a->reps; r++)
    for (size_t o = 0; o < a->bytes; o += 256) {
      acc = _mm512_add_epi32(acc, _mm512_load_si512(a->buf + o));
      acc = _mm512_add_epi32(acc, _mm512_load_si512(a->buf + o + 64));
      acc = _mm512_add_epi32(acc, _mm512_load_si512(a->buf + o + 128));
      acc = _mm512_add_epi32(acc, _mm512_load_si512(a->buf + o + 192));
    }
  volatile int sink = _mm512_reduce_add_epi32(acc); (void)sink;
  return NULL;
}

static double run_threads(void *(*fn)(void *), void *args, size_t argsz, int n) {
  pthread_t th[64];
  double t0 = now();
  for (int i = 0; i < n; i++)
    pthread_create(&th[i], NULL, fn, (char *)args + i * argsz);
  for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
  return now() - t0;
}

int main(int argc, char **argv) {
  if (argc > 1) NTHREADS = atoi(argv[1]);
  if (amx_enable() != 0) { printf("AMX permission request FAILED\n"); }
  else printf("AMX XTILEDATA permission: OK\n");
  printf("threads=%d\n\n", NTHREADS);

  long iters = 20000000L;
  long args[64];
  for (int i = 0; i < 64; i++) args[i] = iters;

  // AMX INT8: 4 dpbssd per iter, each 16x16x64 MAC = 16384 MAC = 32768 ops
  double t = run_threads(amx_i8_worker, args, sizeof(long), NTHREADS);
  double ops = (double)iters * 4 * 16.0 * 16.0 * 64.0 * 2.0 * NTHREADS;
  printf("AMX  INT8  : %8.2f TOPS   (%.3fs)\n", ops / t / 1e12, t);

  t = run_threads(amx_bf16_worker, args, sizeof(long), NTHREADS);
  ops = (double)iters * 4 * 16.0 * 16.0 * 32.0 * 2.0 * NTHREADS;
  printf("AMX  BF16  : %8.2f TFLOPS (%.3fs)\n", ops / t / 1e12, t);

  long viters = 40000000L;
  for (int i = 0; i < 64; i++) args[i] = viters;
  t = run_threads(vnni_worker, args, sizeof(long), NTHREADS);
  ops = (double)viters * 8 * 64.0 * 2.0 * NTHREADS;
  printf("AVX512 VNNI: %8.2f TOPS   (%.3fs)\n", ops / t / 1e12, t);

  printf("\n-- memory bandwidth (read, all threads) --\n");
  size_t sizes[] = {64ul << 10, 1ul << 20, 8ul << 20, 64ul << 20, 512ul << 20};
  const char *names[] = {"64KB (L1/L2)", "1MB (L2)", "8MB (L2/L3)", "64MB (L3)", "512MB (DRAM)"};
  for (int s = 0; s < 5; s++) {
    size_t per = sizes[s] / NTHREADS;
    per &= ~(size_t)255;
    bwarg_t ba[64];
    long reps = (long)(2.0e9 / (double)per);
    if (reps < 3) reps = 3;
    if (reps > 200000) reps = 200000;
    for (int i = 0; i < NTHREADS; i++) {
      if (posix_memalign((void **)&ba[i].buf, 64, per)) { perror("alloc"); return 1; }
      memset(ba[i].buf, 1, per);
      ba[i].bytes = per; ba[i].reps = reps;
    }
    t = run_threads(bw_read_worker, ba, sizeof(bwarg_t), NTHREADS);
    double gb = (double)per * reps * NTHREADS / 1e9;
    printf("  %-14s %8.1f GB/s\n", names[s], gb / t);
    for (int i = 0; i < NTHREADS; i++) free(ba[i].buf);
  }
  return 0;
}
