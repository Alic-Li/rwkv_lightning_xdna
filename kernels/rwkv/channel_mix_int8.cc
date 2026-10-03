// SPDX-License-Identifier: Apache-2.0
#include "ffn_common.hpp"
// Activation packet: 256 signed codes, FP32 scale, 60 padding bytes.
// Per-block dot bound is 256*127*127 = 4,129,024, exactly convertible to FP32.
static void tile(const uint8_t *x, const uint8_t *w, float *out, int row,
                 int col) {
  const auto *packet = x + col * 320;
  const auto *a = reinterpret_cast<const int8_t *>(packet);
  const float scale = *reinterpret_cast<const float *>(packet + 256);
  const auto *weight = reinterpret_cast<const int8_t *>(w);
  const auto *ws = reinterpret_cast<const float *>(w + 4096);
  auto a0 = aie::load_v<64>(a), a1 = aie::load_v<64>(a + 64);
  auto a2 = aie::load_v<64>(a + 128), a3 = aie::load_v<64>(a + 192);
  alignas(64) int32_t dots[16];
  for (int r = 0; r < 16; ++r) {
    aie::accum<acc32, 64> acc = aie::mul(a0, aie::load_v<64>(weight + r * 256));
    acc = aie::mac(acc, a1, aie::load_v<64>(weight + r * 256 + 64));
    acc = aie::mac(acc, a2, aie::load_v<64>(weight + r * 256 + 128));
    acc = aie::mac(acc, a3, aie::load_v<64>(weight + r * 256 + 192));
    dots[r] = aie::reduce_add(acc.to_vector<int32_t>());
  }
  auto scaled = aie::mul(aie::to_float(aie::load_v<16>(dots), 0), scale)
                    .to_vector<float>();
  auto value = aie::mul(scaled, aie::load_v<16>(ws)).to_vector<float>();
  aie::store_v(out + row * 16,
               aie::add(aie::load_v<16>(out + row * 16), value));
}
extern "C" void rwkv7_ffn_key_tile(const uint8_t *x, const uint8_t *w,
                                   float *out, int row, int col) {
  tile(x, w, out, row, col);
}
extern "C" void rwkv7_ffn_value_tile(const uint8_t *x, const uint8_t *w,
                                     float *out, int row, int col) {
  tile(x, w, out, row, col);
}

#include "activation_int8.hpp"
extern "C" void rwkv7_ffn_activate_quant256(float *diagnostic, uint8_t *out) {
  relu_squared_fp32<256>(diagnostic, diagnostic + 256);
  convert<256>(diagnostic + 256, out);
}
