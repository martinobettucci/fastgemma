// q4g repro with an INDEPENDENT scalar check (not sharing the kernel's unpack).
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

int fgm_amx_init(void);
void fgm_pack_a(int,int,const int8_t*,int8_t*);
void fgm_gemm_q4g(int, int, int, const int8_t *, const float *, const uint8_t *,
                  const _Float16 *, int, float *, int, int, int);
void fgm_gemm_ref_q4g(int, int, int, const int8_t *, const float *, const uint8_t *,
                      const _Float16 *, int, float *, int);

static int8_t nib(uint8_t byte, int high) {
  int v = high ? (byte >> 4) & 0xF : byte & 0xF;
  return (int8_t)((v ^ 8) - 8);
}

int main(void) {
  fgm_amx_init();
  const int M = 2, N = 64, K = 128, group = 64;
  const int NB = N / 16, KB = K / 64;
  int8_t *A = aligned_alloc(64, (size_t)M * K);
  float as[2] = {1.0f, 1.0f};
  uint8_t *Bq = aligned_alloc(64, (size_t)NB * KB * 512);
  _Float16 *bs = aligned_alloc(64, (size_t)(K / group) * N * sizeof(_Float16));
  float *C = aligned_alloc(64, (size_t)M * N * sizeof(float));
  float *R = aligned_alloc(64, (size_t)M * N * sizeof(float));
  float *S = calloc((size_t)M * N, sizeof(float));

  uint32_t s = 7;
#define RND ((s = s * 1103515245u + 12345u) >> 16)
  for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int8_t)((int)(RND % 9) - 4);
  for (size_t i = 0; i < (size_t)NB * KB * 512; i++) Bq[i] = (uint8_t)(RND & 0xFF);
  for (int i = 0; i < (K / group) * N; i++) bs[i] = (_Float16)(1.0f + 0.25f * (i % 3));

  // Independent scalar: decode the documented layouts directly.
  //   tile(nb,kb) byte at [r*64 + n*4 + j]  <-> B[k = kb*64 + r*4 + j][n = nb*16+n]
  //   nibble: packed[blk*32 + i] low = value i, high = value i+32, within each 64B block
  for (int nb = 0; nb < NB; nb++)
    for (int kb = 0; kb < KB; kb++) {
      const uint8_t *tp = Bq + ((size_t)nb * KB + kb) * 512;
      for (int r = 0; r < 16; r++)
        for (int n = 0; n < 16; n++)
          for (int j = 0; j < 4; j++) {
            int pos = r * 64 + n * 4 + j;         // position within the 1024B tile
            int blk = pos / 64, off = pos % 64;   // 64B block, offset in it
            int8_t bv = nib(tp[blk * 32 + (off % 32)], off >= 32);
            int k = kb * 64 + r * 4 + j;
            for (int m = 0; m < M; m++)
              S[(size_t)m * N + nb * 16 + n] +=
                  (float)A[(size_t)m * K + k] * bv *
                  (float)bs[(size_t)(k / group) * N + nb * 16 + n];
          }
    }

  fgm_gemm_ref_q4g(M, N, K, A, as, Bq, bs, group, R, N);
  fgm_gemm_q4g(M, N, K, A, as, Bq, bs, group, C, N, 0, NB);

  printf("      %8s %8s %8s\n", "scalar", "ref", "amx");
  for (int i = 0; i < 8; i++)
    printf("n=%-3d %8.1f %8.1f %8.1f\n", i, S[i], R[i], C[i]);
  double e1 = 0, e2 = 0, d = 0;
  for (int i = 0; i < M * N; i++) {
    e1 += (R[i] - S[i]) * (R[i] - S[i]);
    e2 += (C[i] - S[i]) * (C[i] - S[i]);
    d += S[i] * S[i];
  }
  printf("ref vs scalar: %.3e   amx vs scalar: %.3e\n", sqrt(e1 / d), sqrt(e2 / d));
  return 0;
}
