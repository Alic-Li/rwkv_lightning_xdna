// SPDX-License-Identifier: Apache-2.0
#include "chunk4_channel_common.hpp"
// W8A16: one unpack per weight tile, four BF16 token MACs, FP32 accumulation.
// Scale is applied only after the last K tile, exactly as in resident decode.
static void unpack(const uint8_t *packed, bfloat16 *w) {
  for (int i = 0; i < 4096; i += 64) {
    auto q = aie::load_v<64>(reinterpret_cast<const int8_t *>(packed + i));
    aie::accum<accfloat, 64> f(aie::to_float(q, 0));
    aie::store_v(w + i, f.to_vector<bfloat16>());
  }
}
static void scale_row(float *out, const uint8_t *packed) {
  auto scale = aie::load_v<16>(reinterpret_cast<const float *>(packed + 4096));
  aie::store_v(out, aie::mul(aie::load_v<16>(out), scale).to_vector<float>());
}
extern "C" void rwkv7_chunk4_key_int8(const bfloat16 *x, const uint8_t *packed,
                                     float *out, int col) {
  alignas(64) bfloat16 w[4096];
  unpack(packed, w);
  for (int t = 0; t < 4; ++t) {
    gemv_preconverted(x + t * 2048 + col * 256, w, out + t * 16);
    if (col == 7) scale_row(out + t * 16, packed);
  }
}
extern "C" void rwkv7_chunk4_value_int8(const bfloat16 *block, const uint8_t *packed,
                                       float *out, int row, int col) {
  alignas(64) bfloat16 w[4096];
  unpack(packed, w);
  for (int t = 0; t < 4; ++t) {
    auto sum = out + t * 512 + row * 16;
    gemv_preconverted(block + t * 256, w, sum);
    if (col == 31) scale_row(sum, packed);
  }
}
