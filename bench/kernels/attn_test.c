// Correctness and quantisation-error harness for the per-(row, head) attention
// kernel.
//
// The interesting question is not "does the kernel match a float reference" --
// it cannot, the KV cache is int8 by design. It is *how much error the kernel
// adds on top of the error the cache already contributes*. So this measures
// against one exact double-precision reference:
//
//   int8 K/V only  what the engine already accepted before any of this work
//                  (exact query, quantised cache)
//   kernel         what it does now
//   ratio          the kernel's own approximation, in units of error the design
//                  already tolerates
//
// A kernel change landing near 1x is free in accuracy terms. One landing at 10x
// is a real degradation however small the absolute number looks.
//
// Build:
//   gcc -O2 -march=native -o attn_test attn_test.c ../../crates/fgm-kernels/csrc/ops.c -lm

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void fgm_attend_q8_heads(float *out, const float *q, const int8_t *kc, const float *ks,
                         const int8_t *vc, const float *vs, int n_heads, int kv_heads,
                         int head_dim, int start, int end, float *scratch,
                         int h0, int h1, int ring);

static uint64_t rs = 12345;
static float rnd(void) {
  rs = rs * 6364136223846793005ULL + 1;
  return (float)((int64_t)(rs >> 33) % 2001 - 1000) / 1000.0f;
}

#define NH 8
#define KVH 1
#define HD 256
#define N 1000

static float qf[NH * HD], kf[N * HD], vf[N * HD];
static int8_t kc[N * HD], vc[N * HD];
static float ks[N], vs[N];
static float out[NH * HD], scratch[N + 64];
static double exact[NH * HD], cached[NH * HD], got[NH * HD];

// Exact attention in double precision. `quantk`/`quantv` select which operands
// come from the quantised cache, so one routine produces every reference.
static void reference(double *o, int quantk, int quantv, int start, int end) {
  for (int h = 0; h < NH; h++) {
    double *sc = malloc((size_t)(end - start) * sizeof(double)), mx = -1e300, sum = 0;
    for (int t = start; t < end; t++) {
      double a = 0;
      for (int i = 0; i < HD; i++) {
        double kk = quantk ? (double)kc[(size_t)t * HD + i] * ks[t] : kf[(size_t)t * HD + i];
        a += (double)qf[h * HD + i] * kk;
      }
      sc[t - start] = a;
      if (a > mx) mx = a;
    }
    for (int t = 0; t < end - start; t++) { sc[t] = exp(sc[t] - mx); sum += sc[t]; }
    for (int i = 0; i < HD; i++) {
      double acc = 0;
      for (int t = start; t < end; t++) {
        double vv = quantv ? (double)vc[(size_t)t * HD + i] * vs[t] : vf[(size_t)t * HD + i];
        acc += sc[t - start] / sum * vv;
      }
      o[h * HD + i] = acc;
    }
    free(sc);
  }
}

static double rel(const double *a, const double *b) {
  double num = 0, den = 0;
  for (int i = 0; i < NH * HD; i++) { num += (a[i] - b[i]) * (a[i] - b[i]); den += b[i] * b[i]; }
  return sqrt(num / (den + 1e-30));
}

int main(void) {
  // qmul scales the query, which sets how peaked the softmax is -- and that
  // matters more here than anything else, because score error is amplified
  // through exp(). A quantisation error that is harmless on flat attention is
  // not harmless on sharp attention, so sweep it rather than pick one point.
  double worst = 0;
  printf("  %6s %13s %13s %8s\n", "qmul", "int8 KV only", "kernel", "ratio");
  for (double qmul = 0.25; qmul <= 4.001; qmul *= 2) {
    rs = 12345;
    for (int i = 0; i < NH * HD; i++) qf[i] = rnd() * (float)qmul;
    for (int i = 0; i < N * HD; i++) { kf[i] = rnd(); vf[i] = rnd(); }
    for (int t = 0; t < N; t++) {
      float ka = 0, va = 0;
      for (int i = 0; i < HD; i++) {
        float a = fabsf(kf[(size_t)t * HD + i]); if (a > ka) ka = a;
        float b = fabsf(vf[(size_t)t * HD + i]); if (b > va) va = b;
      }
      ks[t] = ka / 127.0f; vs[t] = va / 127.0f;
      for (int i = 0; i < HD; i++) {
        kc[(size_t)t * HD + i] = (int8_t)lrintf(kf[(size_t)t * HD + i] / ks[t]);
        vc[(size_t)t * HD + i] = (int8_t)lrintf(vf[(size_t)t * HD + i] / vs[t]);
      }
    }
    reference(exact, 0, 0, 0, N);
    reference(cached, 1, 1, 0, N);
    fgm_attend_q8_heads(out, qf, kc, ks, vc, vs, NH, KVH, HD, 0, N, scratch, 0, NH, 0);
    for (int i = 0; i < NH * HD; i++) got[i] = out[i];
    double e_cache = rel(cached, exact), e_kernel = rel(got, exact);
    double ratio = e_kernel / (e_cache + 1e-30);
    if (ratio > worst) worst = ratio;
    printf("  %6.2f %13.4e %13.4e %7.2fx\n", qmul, e_cache, e_kernel, ratio);
  }
  printf("  worst ratio %.2fx\n", worst);
  // 2x means the kernel's own approximation is comparable to the int8 cache the
  // design already accepts. Beyond that it is a new source of error, not a
  // rounding detail.
  return worst > 2.0;
}
