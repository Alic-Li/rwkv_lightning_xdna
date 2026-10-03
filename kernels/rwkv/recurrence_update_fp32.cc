// SPDX-License-Identifier: Apache-2.0
#include "wkv7_vector_fp32.cc"
extern "C" void rwkv7_stage_recurrent(const float *state, const float *aux,
                                      float *out) {
  rwkv7_wkv_fp32(state, aux + 13 * 64, out);
  alignas(32) float y[64];
  for (int j = 0; j < 64; ++j)
    y[j] = out[4096 + j];
  float *updated = out + 4096;
  for (int i = 0; i < 27 * 64; ++i)
    updated[i] = aux[i];
  for (int j = 0; j < 64; ++j)
    updated[8 * 64 + j] = y[j];
}
