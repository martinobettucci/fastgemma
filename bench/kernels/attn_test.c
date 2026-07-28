// Correctness harness for the per-(row, head) attention kernel.
//
// The reference is an independent double-precision implementation of the same
// maths, not a second call into the kernel -- comparing a kernel against itself
// only proves it is consistent. Tolerance is relative and set at 1e-4, which is
// above int8 KV quantisation noise (~5e-5 here) and far below anything a real
// softmax or indexing bug would produce.
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

static uint64_t s = 12345;
static float rnd(void) { s = s * 6364136223846793005ULL + 1; return (float)((int64_t)(s >> 33) % 2001 - 1000) / 1000.0f; }

int main(void) {
  const int NH = 8, KVH = 1, HD = 256, N = 1000;
  float *q = aligned_alloc(64, NH * HD * 4);
  int8_t *kc = aligned_alloc(64, N * KVH * HD);
  int8_t *vc = aligned_alloc(64, N * KVH * HD);
  float *ks = aligned_alloc(64, N * 4), *vs = aligned_alloc(64, N * 4);
  float *out = aligned_alloc(64, NH * HD * 4);
  float *scratch = aligned_alloc(64, (N + 64) * 4);
  for (int i = 0; i < NH * HD; i++) q[i] = rnd() * 3.0f;
  for (int i = 0; i < N * KVH * HD; i++) { kc[i] = (int8_t)((int)(rnd() * 127)); vc[i] = (int8_t)((int)(rnd() * 127)); }
  for (int i = 0; i < N; i++) { ks[i] = 0.01f + 0.001f * i / N; vs[i] = 0.02f; }

  int worst_h = -1, worst_i = -1; double worst = 0, worstref = 0;
  for (int start = 0; start < 3; start++) {
    int end = N - start * 7;
    fgm_attend_q8_heads(out, q, kc, ks, vc, vs, NH, KVH, HD, start, end, scratch, 0, NH, 0);
    for (int h = 0; h < NH; h++) {
      double *sc = malloc((end - start) * sizeof(double)), mx = -1e300, sum = 0;
      for (int t = start; t < end; t++) {
        double a = 0;
        for (int i = 0; i < HD; i++) a += (double)q[h * HD + i] * kc[(size_t)t * HD + i];
        sc[t - start] = a * ks[t];
        if (sc[t - start] > mx) mx = sc[t - start];
      }
      for (int t = 0; t < end - start; t++) { sc[t] = exp(sc[t] - mx); sum += sc[t]; }
      for (int i = 0; i < HD; i++) {
        double o = 0;
        for (int t = start; t < end; t++) o += sc[t - start] / sum * vs[t] * vc[(size_t)t * HD + i];
        double d = fabs(o - out[h * HD + i]);
        if (d > worst) { worst = d; worstref = fabs(o); worst_h = h; worst_i = i; }
      }
      free(sc);
    }
  }
  printf("max abs diff %.3e (ref |%.4f|, rel %.3e) at h=%d i=%d\n",
         worst, worstref, worst / (worstref + 1e-12), worst_h, worst_i);
  return worst / (worstref + 1e-9) > 1e-4;
}
