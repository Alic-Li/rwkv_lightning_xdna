// SPDX-License-Identifier: Apache-2.0
#include "attention_projections_bf16.cc"
#include "norm_fp32.cc"
extern "C" void rwkv7_mode_norm(const float *x, const float *gamma,
                                const float *beta, float *out) {
  layer_norm_f32_impl<float, float, 16, true>(x, out, gamma, beta, 2048);
}
extern "C" void rwkv7_mode_copy16(const float *x, float *out, int offset) {
  aie::store_v(out, aie::load_v<16>(x + offset));
}
extern "C" void rwkv7_mode_mix16(const float *x, const float *old,
                                 const float *coeff, float *out, int offset) {
  auto a = aie::load_v<16>(x + offset);
  auto b = aie::load_v<16>(old + offset);
  auto c = aie::load_v<16>(coeff + offset);
  auto delta = aie::sub(b, a);
  auto product = aie::mul(delta, c).to_vector<float>();
  aie::store_v(out, aie::add(a, product));
}
