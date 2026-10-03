// SPDX-License-Identifier: Apache-2.0
#include "recurrence_finish_fp32.cc"
extern "C" void rwkv7_pair_finish(const float *aux, const float *y, float *out) {
  for (int token = 0; token < 2; ++token) {
    float *updated = out + token * 1728;
    for (int i = 0; i < 1728; ++i) updated[i] = aux[token * 1728 + i];
    for (int i = 0; i < 64; ++i) updated[8 * 64 + i] = y[token * 64 + i];
    rwkv7_stage_finish(updated, updated);
  }
}
