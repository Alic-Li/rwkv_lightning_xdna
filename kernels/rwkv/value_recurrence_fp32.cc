// SPDX-License-Identifier: Apache-2.0
#include "recurrence_prepare_fp32.cc"
// The first 27 vectors retain the original arena ABI. The last three hold
// raw value, value gate projection and bias, preserving diagnostic inputs.
extern "C" void rwkv7_value_stage_prepare(const float *aux, const float *first,
                                          float *out, int head) {
  rwkv7_stage_prepare(aux, out);
#if RWKV_PREPARE_TRACE_REGION == 4
  event0();
#endif
  alignas(32) float gate[64];
  sigmoid64(aux + 28 * 64, aux + 29 * 64, gate);
  for (int j = 0; j < 64; ++j) {
    float value = aux[27 * 64 + j];
    out[16 * 64 + j] = value + (first[head * 64 + j] - value) *
        gate[j];
  }
#if RWKV_PREPARE_TRACE_REGION == 4
  event1();
#endif
}
