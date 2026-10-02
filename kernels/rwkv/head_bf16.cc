// SPDX-License-Identifier: Apache-2.0
#include "gemv_bf16.cc"
extern "C" void rwkv7_head_tile(const float *x, const bfloat16 *w, float *out,
                                int col) {
  rwkv7_gemv_tile(x + 256 * col, w, out);
}
