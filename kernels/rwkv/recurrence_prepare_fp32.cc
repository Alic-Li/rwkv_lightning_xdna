// SPDX-License-Identifier: Apache-2.0
#include "stages_fp32.cc"
// One head of a shared 27-vector arena: prepare inputs [0,8), finish
// inputs [8,13), recurrent vectors [13,19), prepare outputs [19,21),
// finish outputs [21,27). Preserve every unchanged slot, including constants.
extern "C" void rwkv7_stage_prepare(const float *aux, float *updated) {
  alignas(32) float prepared[384];
  rwkv7_prepare_head(aux, prepared);
  for (int i = 0; i < 27 * 64; ++i)
    updated[i] = aux[i];
  for (int j = 0; j < 64; ++j) {
    updated[14 * 64 + j] = prepared[320 + j];
    updated[15 * 64 + j] = prepared[256 + j];
    updated[17 * 64 + j] = prepared[128 + j];
    updated[18 * 64 + j] = prepared[192 + j];
    updated[19 * 64 + j] = prepared[j];
    updated[20 * 64 + j] = prepared[64 + j];
  }
}
