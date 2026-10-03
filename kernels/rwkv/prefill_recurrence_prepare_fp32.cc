// SPDX-License-Identifier: Apache-2.0
#include "value_recurrence_fp32.cc"
extern "C" void rwkv7_pair_prepare(const float *aux, float *out) {
  for (int token = 0; token < 2; ++token)
    rwkv7_stage_prepare(aux + token * 1728, out + token * 1728);
}
extern "C" void rwkv7_pair_value_prepare(const float *aux, const float *first,
                                          float *out, int head) {
  rwkv7_value_stage_prepare(aux, first, out, head);
  rwkv7_value_stage_prepare(aux + 1920, first + 2048, out + 1728, head);
}
