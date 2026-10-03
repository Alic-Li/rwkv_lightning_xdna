// SPDX-License-Identifier: Apache-2.0
#include "gemv_preconverted_bf16.hpp"
#include "ffn_common.hpp"

// INT8 values are exactly representable in BF16. Dequantize only the current
// streamed tile on the core; retain FP32 partial sums and apply the per-output
// FP16-derived scale once, after the complete dot product (CUDA W8A16 order).
template <int KTiles>
static void w8a16_tile(const bfloat16 *x, const uint8_t *packed, float *out,
                      int row, int col) {
  alignas(64) bfloat16 w[4096];
  for (int i = 0; i < 4096; i += 64) {
    auto q = aie::load_v<64>(reinterpret_cast<const int8_t *>(packed + i));
    aie::accum<accfloat, 64> f(aie::to_float(q, 0));
    aie::store_v(w + i, f.to_vector<bfloat16>());
  }
  float *sum = out + row * 16;
  gemv_preconverted(x + col * 256, w, sum);
  if (col == KTiles - 1) {
    auto scale = aie::load_v<16>(reinterpret_cast<const float *>(packed + 4096));
    aie::store_v(sum, aie::mul(aie::load_v<16>(sum), scale).to_vector<float>());
  }
}
extern "C" void rwkv7_ffn_key_tile(const bfloat16 *x, const uint8_t *w, float *out,
                                   int row, int col) {
  w8a16_tile<8>(x, w, out, row, col);
}
extern "C" void rwkv7_ffn_value_tile(const bfloat16 *x, const uint8_t *w, float *out,
                                     int row, int col) {
  w8a16_tile<32>(x, w, out, row, col);
}
