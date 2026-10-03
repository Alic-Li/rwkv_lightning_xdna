// SPDX-License-Identifier: Apache-2.0
#include "gemv_preconverted_bf16.hpp"

// One streamed 16x256 weight tile serves every token before it is released.
// Keep decode's per-token MAC reduction and K-tile accumulation order.
extern "C" void rwkv7_prefill_projection_tile(const bfloat16 *x,
                                              const bfloat16 *w, float *out,
                                              int col) {
  for (int token = 0; token < PREFILL_BATCH; ++token)
    gemv_preconverted(x + token * PREFILL_K + col * 256, w, out + token * 16);
}
extern "C" void rwkv7_prefill_projection_zero(float *out) {
  for (int i = 0; i < PREFILL_BATCH * 16; i += 16)
    aie::store_v(out + i, aie::zeros<float, 16>());
}
