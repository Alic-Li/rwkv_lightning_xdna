// SPDX-License-Identifier: Apache-2.0
#pragma once
// 320-byte packet: 256 signed codes, FP32 scale, 60 zero padding bytes.
#include <aie_api/aie.hpp>
template <int N> static void convert(const float *x, uint8_t *out) {
  const auto rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  const auto saturation = aie::swap_saturation(aie::saturation_mode::saturate);
  for (int block = 0; block < N / 256; ++block) {
    auto maximum = aie::zeros<float, 32>();
    for (int i = 0; i < 256; i += 32)
      maximum =
          aie::max(maximum, aie::abs(aie::load_v<32>(x + block * 256 + i)));
    float peak = aie::reduce_max(maximum);
    float scale = peak == 0.f ? 1.f : peak / 127.f;
    float inv = 1.f / scale;
    auto *packet = out + block * 320;
    for (int i = 0; i < 320; i += 64)
      aie::store_v(packet + i, aie::zeros<uint8_t, 64>());
    *reinterpret_cast<float *>(packet + 256) = scale;
    for (int i = 0; i < 256; i += 32) {
      auto v = aie::mul(aie::load_v<32>(x + block * 256 + i), inv)
                   .to_vector<float>();
      v = aie::min(aie::max(v, -127.f), 127.f);
      auto q = aie::to_fixed<int16_t>(v, 0).pack<int8_t>();
      aie::store_v(reinterpret_cast<int8_t *>(packet) + i, q);
    }
  }
  aie::set_saturation(saturation);
  aie::set_rounding(rounding);
}
