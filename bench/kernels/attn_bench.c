// Standalone throughput probe for the attention kernel.
//
// Written before starting an AMX rewrite, because the two previous times I
// picked an attention optimisation from a first-principles estimate I was wrong:
// once by assuming KV re-reads hit DRAM when they were L2-resident (blocked
// attention, -12%), and once by putting attention at 11% of prefill when it was
// 61.9%. So: measure what the kernel actually achieves, against both ceilings,
// and let the gap say which ceiling is binding.
//
// Reports, per shape:
//   G MAC/s   achieved multiply-accumulates per second
//   %VNNI     against this box's measured AVX512-VNNI peak
//   GB/s      K and V bytes touched per second, assuming each (row, head) pass
//             re-reads the range -- an upper bound on traffic, since with one KV
//             head the eight query heads of a row hit the same lines
//
// If %VNNI is high, the kernel is instruction-bound and AMX (14.69 TOPS here,
// 12.7x VNNI) is the answer. If GB/s is near DRAM (33) or L2 (209), it is
// memory-bound and AMX buys nothing -- the bytes still have to move. If neither
// is close, it is latency or overhead, and the fix is neither.
//
// Build:
//   gcc -O2 -march=native -o attn_bench attn_bench.c ../../crates/fgm-kernels/csrc/ops.c -lm

#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void fgm_attend_q8_heads(float *out, const float *q, const int8_t *kc, const float *ks,
                         const int8_t *vc, const float *vs, int n_heads, int kv_heads,
                         int head_dim, int start, int end, float *scratch,
                         int h0, int h1, int ring, int cap);
void fgm_store_v_t(int8_t *vt, const int8_t *src, int n, int slot);

// measured on this box by bench/roofline
#define VNNI_PEAK_MACS 580e9
#define DRAM_GBS 33.0
#define L2_GBS 209.0

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static uint64_t rs = 99;
static float rnd(void) {
  rs = rs * 6364136223846793005ULL + 1;
  return (float)((int64_t)(rs >> 33) % 2001 - 1000) / 1000.0f;
}

int main(int argc, char **argv) {
  const int NH = 8, KVH = 1;
  const int reps = argc > 1 ? atoi(argv[1]) : 3;
  const int hds[] = {256, 512};
  const int ctxs[] = {512, 1024, 2048, 4096, 8192};

  printf("  %4s %6s %5s %9s %9s %8s %8s %8s\n",
         "hd", "ctx", "rows", "ms", "G MAC/s", "%VNNI", "GB/s", "%DRAM");
  for (int hi = 0; hi < 2; hi++) {
    const int HD = hds[hi];
    for (int ci = 0; ci < 5; ci++) {
      const int T = ctxs[ci];
      // one prefill chunk's worth of query rows against a full context
      const int ROWS = 16;
      const size_t kvd = (size_t)KVH * HD;

      float *q = aligned_alloc(64, (size_t)NH * HD * 4);
      int8_t *kc = aligned_alloc(64, (size_t)T * kvd);
      int8_t *vt = aligned_alloc(64, (size_t)((T + 3) / 4) * 4 * kvd);
      int8_t *vrow = aligned_alloc(64, kvd);
      float *ks = aligned_alloc(64, (size_t)T * 4), *vs = aligned_alloc(64, (size_t)T * 4);
      float *out = aligned_alloc(64, (size_t)NH * HD * 4);
      // scratch: scores, two u8 weight planes, two int32 accumulator sets
      float *scratch = aligned_alloc(64, (size_t)(T + T / 2 + 2 * HD + 64) * 4);
      if (!q || !kc || !vt || !ks || !vs || !out || !scratch) { puts("alloc"); return 1; }

      for (size_t i = 0; i < (size_t)NH * HD; i++) q[i] = rnd();
      for (size_t i = 0; i < (size_t)T * kvd; i++) kc[i] = (int8_t)(int)(rnd() * 127);
      for (int t = 0; t < T; t++) {
        ks[t] = 0.01f; vs[t] = 0.02f;
        for (size_t i = 0; i < kvd; i++) vrow[i] = (int8_t)(int)(rnd() * 127);
        fgm_store_v_t(vt, vrow, (int)kvd, t);
      }

      // warm the caches so the first iteration is not a page-fault measurement
      fgm_attend_q8_heads(out, q, kc, ks, vt, vs, NH, KVH, HD, 0, T, scratch, 0, NH, 0, T);

      double best = 1e30;
      for (int r = 0; r < reps; r++) {
        double t0 = now();
        for (int row = 0; row < ROWS; row++)
          fgm_attend_q8_heads(out, q, kc, ks, vt, vs, NH, KVH, HD, 0, T, scratch, 0, NH, 0, T);
        double el = now() - t0;
        if (el < best) best = el;
      }
      // Q.K^T and P.V are both T*HD MACs per head
      double macs = (double)ROWS * NH * 2.0 * T * HD;
      double bytes = (double)ROWS * NH * 2.0 * T * kvd;
      printf("  %4d %6d %5d %9.2f %9.1f %7.1f%% %8.1f %7.1f%%\n",
             HD, T, ROWS, best * 1e3, macs / best / 1e9,
             100.0 * (macs / best) / VNNI_PEAK_MACS,
             bytes / best / 1e9, 100.0 * (bytes / best / 1e9) / DRAM_GBS);
      free(q); free(kc); free(vt); free(vrow); free(ks); free(vs); free(out); free(scratch);
    }
  }
  printf("\n  ceilings: VNNI %.0f G MAC/s, DRAM %.0f GB/s, L2 %.0f GB/s\n",
         VNNI_PEAK_MACS / 1e9, DRAM_GBS, L2_GBS);
  return 0;
}
