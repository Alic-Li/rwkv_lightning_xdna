// SPDX-License-Identifier: Apache-2.0
// Experimental chunk4 FFN: each GEMV retains the production K reduction order.
#include "gemv_preconverted_bf16.hpp"
#include "relu_squared_fp32.hpp"
extern "C" void rwkv7_chunk4_zero64(float *out) {
  for (int i = 0; i < 64; i += 32) aie::store_v(out + i, aie::zeros<float, 32>());
}
extern "C" void rwkv7_chunk4_zero2048(float *out) {
  for (int i = 0; i < 2048; i += 32) aie::store_v(out + i, aie::zeros<float, 32>());
}
extern "C" void rwkv7_chunk4_key(const bfloat16 *x, const bfloat16 *w,
                                float *out, int col) {
  for (int t = 0; t < 4; ++t)
    gemv_preconverted(x + t * 2048 + col * 256, w, out + t * 16);
}
extern "C" void rwkv7_chunk4_collect(const float *packet, float *raw,
                                    bfloat16 *active, int row) {
  alignas(64) float squared[64];
  relu_squared_fp32<64>(packet, squared);
  const auto rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int t = 0; t < 4; ++t) {
    const int offset = t * 2048 + row * 16;
    aie::store_v(raw + offset, aie::load_v<16>(packet + t * 16));
    aie::store_v(active + offset,
      aie::accum<accfloat, 16>(aie::load_v<16>(squared + t * 16)).to_vector<bfloat16>());
  }
  aie::set_rounding(rounding);
}
extern "C" void rwkv7_chunk4_value(const bfloat16 *block, const bfloat16 *w,
                                  float *out, int row) {
  for (int t = 0; t < 4; ++t)
    gemv_preconverted(block + t * 256, w, out + t * 512 + row * 16);
}
extern "C" void rwkv7_chunk4_add(const float *x, const float *projection,
                                float *out, int token) {
  for (int i = 0; i < 2048; i += 32)
    aie::store_v(out + i, aie::add(aie::load_v<32>(x + i),
      aie::load_v<32>(projection + (i / 512) * 2048 + token * 512 + i % 512)));
}
