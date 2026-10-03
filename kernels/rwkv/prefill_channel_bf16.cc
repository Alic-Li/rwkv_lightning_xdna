// SPDX-License-Identifier: Apache-2.0
#include "gemv_preconverted_bf16.hpp"
#include "relu_squared_fp32.hpp"
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
