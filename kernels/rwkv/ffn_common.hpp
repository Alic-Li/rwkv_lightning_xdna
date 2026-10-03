// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <aie_api/aie.hpp>
#include "relu_squared_fp32.hpp"
extern "C" void rwkv7_ffn_zero2048(float *out) {
  for (int i = 0; i < 2048; i += 32)
    aie::store_v(out + i, aie::zeros<float, 32>());
}
extern "C" void rwkv7_ffn_zero256(float *out) {
  for (int i = 0; i < 256; i += 32)
    aie::store_v(out + i, aie::zeros<float, 32>());
}
extern "C" void rwkv7_ffn_activate2048(const float *raw, float *out) {
  relu_squared_fp32<2048>(raw, out);
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
