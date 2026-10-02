// SPDX-License-Identifier: Apache-2.0
#include "gemv_bf16.cc"
extern "C" void rwkv7_ffn_zero2048(float *out) {
  for (int i = 0; i < 2048; i += 32)
    aie::store_v(out + i, aie::zeros<float, 32>());
}
extern "C" void rwkv7_ffn_zero256(float *out) {
  for (int i = 0; i < 256; i += 32)
    aie::store_v(out + i, aie::zeros<float, 32>());
}
extern "C" void rwkv7_ffn_key_tile(const float *x, const bfloat16 *w,
                                   float *out, int row, int col) {
  rwkv7_gemv_tile(x + col * 256, w, out + row * 16);
}
extern "C" void rwkv7_ffn_activate2048(const float *raw, float *out) {
  for (int i = 0; i < 2048; ++i) {
    float v = raw[i] > 0 ? raw[i] : 0;
    out[i] = v * v;
  }
}
extern "C" void rwkv7_ffn_residual(const float *pair, float *out) {
  for (int i = 0; i < 2048; i += 32) {
    auto v = aie::load_v<32>(pair + i);
    aie::store_v(out + i, v);
    aie::store_v(out + 2048 + i, aie::add(v, aie::load_v<32>(pair + 2048 + i)));
  }
}

extern "C" void rwkv7_ffn_value_tile(const float *x, const bfloat16 *w,
                                     float *out, int row, int col) {
  rwkv7_gemv_tile(x + col * 256, w, out + row * 16);
}

extern "C" void rwkv7_ffn_copy1024(const float *x, float *out) {
  for (int i = 0; i < 1024; i += 32)
    aie::store_v(out + i, aie::load_v<32>(x + i));
}
