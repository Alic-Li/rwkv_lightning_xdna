// SPDX-License-Identifier: Apache-2.0
// Structural experiment: same row-local GEMV and residual arithmetic.
#include "ffn_pipeline_bf16.cc"
extern "C" void rwkv7_ffn_residual1024(const float *pair, float *out) {
  for (int i = 0; i < 1024; i += 32) {
    auto v = aie::load_v<32>(pair + i);
    aie::store_v(out + i, v);
    aie::store_v(out + 1024 + i,
                 aie::add(v, aie::load_v<32>(pair + 1024 + i)));
  }
}
extern "C" void rwkv7_ffn_copy4096(const float *x, float *out) {
  for (int i = 0; i < 4096; i += 32)
    aie::store_v(out + i, aie::load_v<32>(x + i));
}
