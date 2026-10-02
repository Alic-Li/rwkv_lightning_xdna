// SPDX-License-Identifier: Apache-2.0
#ifndef RWKV_EXACT_FP32
#define RWKV_FAST_EXP
#endif
#include "ops_fp32.cc"
extern "C" void rwkv7_value_residual(const float *p, const float *first,
                                     float *out) {
  for (int i = 0; i < 256; ++i)
    out[i] = p[i] + (first[i] - p[i]) * sigmoid(p[256 + i] + p[512 + i]);
}
