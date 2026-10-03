// SPDX-License-Identifier: Apache-2.0
#include "chunk4_channel_common.hpp"
extern "C" void rwkv7_chunk4_key(const bfloat16 *x, const bfloat16 *w,
                                float *out, int col) {
  for (int t = 0; t < 4; ++t)
    gemv_preconverted(x + t * 2048 + col * 256, w, out + t * 16);
}
extern "C" void rwkv7_chunk4_value(const bfloat16 *block, const bfloat16 *w,
                                  float *out, int row) {
  for (int t = 0; t < 4; ++t)
    gemv_preconverted(block + t * 256, w, out + t * 512 + row * 16);
}
