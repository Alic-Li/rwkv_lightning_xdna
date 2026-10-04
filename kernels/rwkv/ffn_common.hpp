// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "model_shape.hpp"
#include <aie_api/aie.hpp>
#include "relu_squared_fp32.hpp"
extern "C" void rwkv7_ffn_zero2048(float *out) {
  for (int i = 0; i < RWKV_HIDDEN / 4; i += 32)
    aie::store_v(out + i, aie::zeros<float, 32>());
}
extern "C" void rwkv7_ffn_zero256(float *out) {
  for (int i = 0; i < RWKV_CHANNELS / 8; i += 32)
    aie::store_v(out + i, aie::zeros<float, 32>());
}
extern "C" void rwkv7_ffn_activate2048(const float *raw, float *out) {
  relu_squared_fp32<RWKV_HIDDEN / 4>(raw, out);
}
extern "C" void rwkv7_ffn_residual(const float *pair, float *out) {
  for (int i = 0; i < RWKV_CHANNELS; i += 32) {
    auto v = aie::load_v<32>(pair + i);
    aie::store_v(out + i, v);
    aie::store_v(out + RWKV_CHANNELS + i, aie::add(v, aie::load_v<32>(pair + RWKV_CHANNELS + i)));
  }
}

extern "C" void rwkv7_ffn_copy1024(const float *x, float *out) {
  for (int i = 0; i < RWKV_CHANNELS / 2; i += 32)
    aie::store_v(out + i, aie::load_v<32>(x + i));
}

extern "C" void rwkv7_channel_zero512(float *out) {
  for (int i = 0; i < RWKV_CHANNELS / 4; i += 32)
    aie::store_v(out + i, aie::zeros<float, 32>());
}
