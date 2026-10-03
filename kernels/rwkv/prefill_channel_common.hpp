// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <aie_api/aie.hpp>
#include "relu_squared_fp32.hpp"
extern "C" void rwkv7_prefill_zero4096(float *out) {
  for (int i = 0; i < 4096; i += 32) aie::store_v(out + i, aie::zeros<float, 32>());
}
extern "C" void rwkv7_prefill_zero1024(float *out) {
  for (int i = 0; i < 1024; i += 32) aie::store_v(out + i, aie::zeros<float, 32>());
}
extern "C" void rwkv7_prefill_activate4096(const float *raw, bfloat16 *out) {
  // A vector-sized scratch preserves raw diagnostic values without a large
  // second FP32 activation allocation on the key core.
  alignas(64) float activated[32];
  const auto rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int i = 0; i < 4096; i += 32) {
    relu_squared_fp32<32>(raw + i, activated);
    aie::store_v(out + i, aie::accum<accfloat, 32>(aie::load_v<32>(activated))
                              .to_vector<bfloat16>());
  }
  aie::set_rounding(rounding);
}
extern "C" void rwkv7_prefill_add4096(const float *x, const float *projection,
                                      float *out) {
  // The value gather is [value core, token, 512 channels]; x/out are token-major.
  for (int i = 0; i < 4096; i += 32)
    aie::store_v(out + i, aie::add(
        aie::load_v<32>(projection + ((i % 2048) / 512) * 1024 +
                        (i / 2048) * 512 + i % 512), aie::load_v<32>(x + i)));
}
