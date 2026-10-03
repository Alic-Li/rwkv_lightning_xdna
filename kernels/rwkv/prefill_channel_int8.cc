// SPDX-License-Identifier: Apache-2.0
#include "gemv_preconverted_bf16.hpp"
#include "prefill_channel_common.hpp"

// Expand each streamed weight tile once, then reuse it for both tokens.
// Codes are exactly representable in BF16; the row scale is applied only after
// the final K tile, preserving the resident W8A16 accumulation contract.
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
extern "C" void rwkv7_prefill_key_int8(const bfloat16 *x, const uint8_t *packed,
                                      float *out, int row, int col) {
  alignas(64) bfloat16 w[4096];
  unpack(packed, w);
  for (int token = 0; token < 2; ++token) {
    auto sum = out + token * 2048 + row * 16;
    gemv_preconverted(x + token * 2048 + col * 256, w, sum);
    if (col == 7) scale_row(sum, packed);
  }
}
extern "C" void rwkv7_prefill_value_int8(const bfloat16 *x, const uint8_t *packed,
                                        float *out, int row, int col) {
  alignas(64) bfloat16 w[4096];
  unpack(packed, w);
  for (int token = 0; token < 2; ++token) {
    auto sum = out + token * 512 + row * 16;
    gemv_preconverted(x + (col / 8) * 4096 + token * 2048 + (col % 8) * 256, w, sum);
    if (col == 31) scale_row(sum, packed);
  }
}
