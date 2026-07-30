// Head-batched vs head-outer Q.K^T, isolating the loop-order effect.
//
// The claim under test: with MQA (one KV head shared by n_heads query heads),
// loading each K line once and serving all heads raises arithmetic intensity
// from 1 MAC/byte to n_heads MACs/byte, which is the only thing that can move a
// kernel measured at 3-7% of the VNNI ceiling but at or near the DRAM/L3
// bandwidth ceiling.
//
// The counter-pressure, which is why this needs measuring rather than
// reasoning: the head-outer kernel batches FOUR POSITIONS against four
// accumulators and combines them with an unpack/hadd tree, so it pays ~1/4 of a
// horizontal reduction per score. The head-batched version below does one
// _mm512_reduce_add_epi32 per (position, head) -- four times as many
// reductions. Better traffic, worse reduction amortisation. Which wins is an
// empirical question about this machine, and the last three times I answered
// that class of question by reasoning I was wrong.
//
// Build:
//   gcc -O2 -march=native -o qk_bench qk_bench.c ../../crates/fgm-kernels/csrc/ops.c -lm

#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

void fgm_qk_heads_batched(float *out_scores, const int8_t *qq, const int32_t *qsum,
                          const float *qscale, const int8_t *kc, const float *ks,
                          int n_heads, int head_dim, int kvd, int start, int end,
                          int ring, int score_stride);

#define VNNI_PEAK_MACS 580e9
#define NH 8

static double now(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}
static uint64_t rs = 7;
static int8_t rnd8(void) {
  rs = rs * 6364136223846793005ULL + 1;
  return (int8_t)((int64_t)(rs >> 33) % 255 - 127);
}

// Head-outer with 4-position batching -- the shape the shipped kernel uses.
static void qk_head_outer(float *out, const int8_t *qq, const int32_t *qsum,
                          const float *qscale, const int8_t *kc, const float *ks,
                          int head_dim, int start, int end, int stride) {
  const __m512i kbias = _mm512_set1_epi8((char)0x80);
  for (int h = 0; h < NH; h++) {
    const int8_t *qh = qq + (size_t)h * head_dim;
    const int32_t qc = 128 * qsum[h];
    int t = start;
    for (; t + 4 <= end; t += 4) {
      const int8_t *k0 = kc + (size_t)t * head_dim, *k1 = k0 + head_dim;
      const int8_t *k2 = k1 + head_dim, *k3 = k2 + head_dim;
      __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
      __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
      for (int i = 0; i < head_dim; i += 64) {
        const __m512i qv = _mm512_loadu_si512((const void *)(qh + i));
        a0 = _mm512_dpbusd_epi32(a0, _mm512_xor_si512(_mm512_loadu_si512((const void *)(k0+i)), kbias), qv);
        a1 = _mm512_dpbusd_epi32(a1, _mm512_xor_si512(_mm512_loadu_si512((const void *)(k1+i)), kbias), qv);
        a2 = _mm512_dpbusd_epi32(a2, _mm512_xor_si512(_mm512_loadu_si512((const void *)(k2+i)), kbias), qv);
        a3 = _mm512_dpbusd_epi32(a3, _mm512_xor_si512(_mm512_loadu_si512((const void *)(k3+i)), kbias), qv);
      }
      __m256i b0 = _mm256_add_epi32(_mm512_castsi512_si256(a0), _mm512_extracti64x4_epi64(a0,1));
      __m256i b1 = _mm256_add_epi32(_mm512_castsi512_si256(a1), _mm512_extracti64x4_epi64(a1,1));
      __m256i b2 = _mm256_add_epi32(_mm512_castsi512_si256(a2), _mm512_extracti64x4_epi64(a2,1));
      __m256i b3 = _mm256_add_epi32(_mm512_castsi512_si256(a3), _mm512_extracti64x4_epi64(a3,1));
      __m256i d = _mm256_hadd_epi32(_mm256_hadd_epi32(b0,b1), _mm256_hadd_epi32(b2,b3));
      __m128i e = _mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d,1));
      int32_t raw[4]; _mm_storeu_si128((__m128i*)raw, e);
      for (int j = 0; j < 4; j++)
        out[(size_t)h*stride + t + j - start] = (float)(raw[j] - qc) * qscale[h] * ks[t+j];
    }
    for (; t < end; t++) {
      const int8_t *kp = kc + (size_t)t * head_dim;
      __m512i acc = _mm512_setzero_si512();
      for (int i = 0; i < head_dim; i += 64)
        acc = _mm512_dpbusd_epi32(acc, _mm512_xor_si512(_mm512_loadu_si512((const void*)(kp+i)), kbias),
                                  _mm512_loadu_si512((const void*)(qh+i)));
      out[(size_t)h*stride + t - start] = (float)(_mm512_reduce_add_epi32(acc) - qc) * qscale[h] * ks[t];
    }
  }
}

int main(int argc, char **argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 5;
  const int hds[] = {256, 512};
  const int ctxs[] = {1024, 2048, 4096, 8192};
  printf("  %4s %6s %12s %12s %9s %9s %8s\n",
         "hd", "ctx", "head-outer", "head-batch", "outer", "batch", "speedup");
  printf("  %4s %6s %12s %12s %9s %9s %8s\n",
         "", "", "ms", "ms", "G MAC/s", "G MAC/s", "");
  for (int hi = 0; hi < 2; hi++) {
    const int HD = hds[hi];
    for (int ci = 0; ci < 4; ci++) {
      const int T = ctxs[ci];
      int8_t *qq = aligned_alloc(64, (size_t)NH*HD);
      int8_t *kc = aligned_alloc(64, (size_t)T*HD);
      float *ks = aligned_alloc(64, (size_t)T*4);
      float *qscale = aligned_alloc(64, NH*4);
      int32_t *qsum = aligned_alloc(64, NH*4);
      float *o1 = aligned_alloc(64, (size_t)NH*T*4), *o2 = aligned_alloc(64, (size_t)NH*T*4);
      for (size_t i = 0; i < (size_t)NH*HD; i++) qq[i] = rnd8();
      for (size_t i = 0; i < (size_t)T*HD; i++) kc[i] = rnd8();
      for (int t = 0; t < T; t++) ks[t] = 0.01f;
      for (int h = 0; h < NH; h++) { qscale[h] = 0.02f; qsum[h] = 13; }

      qk_head_outer(o1, qq, qsum, qscale, kc, ks, HD, 0, T, T);
      fgm_qk_heads_batched(o2, qq, qsum, qscale, kc, ks, NH, HD, HD, 0, T, 0, T);
      double worst = 0;
      for (size_t i = 0; i < (size_t)NH*T; i++) {
        double d = fabs((double)o1[i] - o2[i]);
        if (d > worst) worst = d;
      }

      double b1 = 1e30, b2 = 1e30;
      for (int r = 0; r < reps; r++) {
        double t0 = now(); qk_head_outer(o1, qq, qsum, qscale, kc, ks, HD, 0, T, T);
        double e = now()-t0; if (e < b1) b1 = e;
        t0 = now(); fgm_qk_heads_batched(o2, qq, qsum, qscale, kc, ks, NH, HD, HD, 0, T, 0, T);
        e = now()-t0; if (e < b2) b2 = e;
      }
      double macs = (double)NH * T * HD;
      printf("  %4d %6d %12.3f %12.3f %9.1f %9.1f %7.2fx%s\n",
             HD, T, b1*1e3, b2*1e3, macs/b1/1e9, macs/b2/1e9, b1/b2,
             worst > 1e-3 ? "  MISMATCH" : "");
      free(qq); free(kc); free(ks); free(qscale); free(qsum); free(o1); free(o2);
    }
  }
  printf("\n  VNNI ceiling %.0f G MAC/s; DRAM 33, L3 58, L2 209 GB/s\n", VNNI_PEAK_MACS/1e9);
  return 0;
}
