// SPDX-License-Identifier: Apache-2.0
#include "gemv_preconverted_bf16.hpp"
#include "prefill_channel_common.hpp"
extern "C" void rwkv7_prefill_key(const bfloat16 *x, const bfloat16 *w,
                                  float *out, int row, int col) {
  for (int token = 0; token < 2; ++token)
    gemv_preconverted(x + token * 2048 + col * 256, w,
                      out + token * 2048 + row * 16);
}
extern "C" void rwkv7_prefill_value(const bfloat16 *x, const bfloat16 *w,
                                    float *out, int row, int col) {
  // Gathered activations are [key core, token, 2048 channels].
  for (int token = 0; token < 2; ++token)
    gemv_preconverted(x + (col / 8) * 4096 + token * 2048 + (col % 8) * 256, w,
                      out + token * 512 + row * 16);
}
