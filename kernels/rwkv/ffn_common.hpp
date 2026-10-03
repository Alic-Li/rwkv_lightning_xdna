// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <aie_api/aie.hpp>
extern "C" void rwkv7_ffn_zero2048(float *out) {
  for (int i = 0; i < 2048; i += 32)
    aie::store_v(out + i, aie::zeros<float, 32>());
}
extern "C" void rwkv7_ffn_zero256(float *out) {
  for (int i = 0; i < 256; i += 32)
    aie::store_v(out + i, aie::zeros<float, 32>());
}
// AIE FP32 multiply decomposes operands into BF16 products and can differ
// from the scalar IEEE square by one ULP. Square the 24-bit significand with
// an exact 64-bit integer product, then round once to nearest-even.
extern "C" void rwkv7_ffn_activate2048(const float *raw, float *out) {
  const auto rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  const auto zero = aie::zeros<float, 32>();
  for (int i = 0; i < 2048; i += 32) {
    const auto x = aie::load_v<32>(raw + i);
    const auto v = aie::select(zero, x, aie::gt(x, zero));
    const auto bits = v.cast_to<uint32_t>();
    const auto exponent = aie::logical_downshift(bits, 23);
    const auto mantissa = aie::bit_or(
        aie::bit_and(bits, aie::broadcast<uint32_t, 32>(0x7fffff)),
        aie::broadcast<uint32_t, 32>(0x800000));
    const auto product = aie::mul(mantissa, mantissa);
    // ceil(sqrt(2) * 2^23): whether the exact square needs an extra shift.
    const auto high = aie::ge(mantissa, uint32_t(11863284));
    const auto rounded = aie::select(product.to_vector<uint32_t>(23),
                                    product.to_vector<uint32_t>(24), high);
    const auto carry = aie::select(aie::zeros<uint32_t, 32>(),
                                  aie::broadcast<uint32_t, 32>(1), high);
    const auto exp_out = aie::add(
        aie::sub(aie::add(exponent, exponent), uint32_t(127)), carry);
    const auto result = aie::add(aie::upshift(exp_out, 23),
        aie::sub(rounded, uint32_t(0x800000))).cast_to<float>();
    aie::store_v(out + i, aie::select(zero, result, aie::gt(v, zero)));
    // Rare underflow/overflow and infinities retain scalar IEEE behavior.
    const auto exceptional = aie::gt(v, zero) &
        (aie::lt(exponent, uint32_t(64)) | aie::gt(exponent, uint32_t(190)));
    if (!exceptional.empty())
      for (int j = 0; j < 32; ++j)
        if (exceptional.test(j)) {
          float s = raw[i + j];
          out[i + j] = s * s;
        }
  }
  aie::set_rounding(rounding);
}
extern "C" void rwkv7_ffn_residual(const float *pair, float *out) {
  for (int i = 0; i < 2048; i += 32) {
    auto v = aie::load_v<32>(pair + i);
    aie::store_v(out + i, v);
    aie::store_v(out + 2048 + i, aie::add(v, aie::load_v<32>(pair + 2048 + i)));
  }
}

extern "C" void rwkv7_ffn_copy1024(const float *x, float *out) {
  for (int i = 0; i < 1024; i += 32)
    aie::store_v(out + i, aie::load_v<32>(x + i));
}

extern "C" void rwkv7_channel_zero512(float *out) {
  for (int i = 0; i < 512; i += 32)
    aie::store_v(out + i, aie::zeros<float, 32>());
}
