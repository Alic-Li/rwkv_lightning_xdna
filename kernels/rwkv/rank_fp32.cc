// SPDX-License-Identifier: Apache-2.0
#include "gemv_fp32.cc"
#ifndef RWKV_EXACT_FP32
#define RWKV_FAST_EXP
#endif
#include "ops_fp32.cc"
extern "C" void rwkv7_rank_zero64(float *out) {
  for (int i = 0; i < 64; ++i)
    out[i] = 0;
}
extern "C" void rwkv7_rank_tile64(const float *x, const float *w, float *out,
                                  int row) {
  rwkv7_gemv_tile(x, w, out + row * 16);
}
extern "C" void rwkv7_rank_activate64(const float *raw, float *out,
                                      int activation) {
  for (int i = 0; i < 64; ++i)
    out[i] = activation == 1   ? tanhf(raw[i])
             : activation == 2 ? sigmoid(raw[i])
                               : raw[i];
}
