// SPDX-License-Identifier: Apache-2.0
#include "gemv_bf16.cc"
#include "ffn_common.hpp"
extern "C" void rwkv7_ffn_key_tile(const float *x, const bfloat16 *w,
                                   float *out, int row, int col) {
  rwkv7_gemv_tile(x + col * 256, w, out + row * 16);
}
extern "C" void rwkv7_ffn_value_tile(const float *x, const bfloat16 *w,
                                     float *out, int row, int col) {
  rwkv7_gemv_tile(x + col * 256, w, out + row * 16);
}
