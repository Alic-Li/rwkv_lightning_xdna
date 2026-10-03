// SPDX-License-Identifier: Apache-2.0
#include "stages_fp32.cc"
extern "C" void rwkv7_stage_finish(const float *aux, float *updated) {
  alignas(32) float p[512], result[384];
  for (int j = 0; j < 64; ++j) {
    p[j] = aux[8 * 64 + j];
    p[64 + j] = aux[9 * 64 + j];
    p[128 + j] = aux[10 * 64 + j];
    p[192 + j] = aux[13 * 64 + j];
    p[256 + j] = aux[15 * 64 + j];
    p[320 + j] = aux[11 * 64 + j];
    p[384 + j] = aux[16 * 64 + j];
    p[448 + j] = aux[12 * 64 + j];
  }
  rwkv7_finish_head(p, result);
  for (int i = 0; i < 27 * 64; ++i)
    updated[i] = aux[i];
  for (int j = 0; j < 64; ++j)
    for (int v = 0; v < 4; ++v)
      updated[(21 + v) * 64 + j] = result[v * 64 + j];
}
