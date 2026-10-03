// SPDX-License-Identifier: Apache-2.0
#include "stages_fp32.cc"
// Return only the two final vectors consumed by output projection. All math
// and ordering match recurrence_finish_fp32.cc; no auxiliary DDR writeback.
extern "C" void rwkv7_pair_finish_projection(const float *aux, const float *y,
                                             float *out, int head) {
  alignas(32) float p[512], result[384];
  for (int token = 0; token < 2; ++token) {
    const float *a = aux + token * 1728;
    for (int j = 0; j < 64; ++j) {
      p[j] = y[token * 64 + j];
      p[64 + j] = a[9 * 64 + j];
      p[128 + j] = a[10 * 64 + j];
      p[192 + j] = a[13 * 64 + j];
      p[256 + j] = a[15 * 64 + j];
      p[320 + j] = a[11 * 64 + j];
      p[384 + j] = a[16 * 64 + j];
      p[448 + j] = a[12 * 64 + j];
    }
    rwkv7_finish_head(p, result);
    for (int j = 0; j < 64; ++j)
      out[head * 128 + token * 64 + j] = result[192 + j];
  }
}

extern "C" void rwkv7_pair_finish_projection512(const float *aux, const float *y,
                                                float *out, int head) {
  rwkv7_pair_finish_projection(aux, y, out, head);
}
