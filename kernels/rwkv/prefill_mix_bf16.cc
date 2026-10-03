// SPDX-License-Identifier: Apache-2.0
#include <aie_api/aie.hpp>
extern "C" void rwkv7_prefill_shift_init(const float *parameters, float *shift) {
  for (int i = 0; i < 2048; i += 32)
    aie::store_v(shift + i, aie::load_v<32>(parameters + 2048 + i));
}
extern "C" void rwkv7_prefill_mix_tokens(const float *normalized,
                                  const float *parameters, float *shift,
                                  bfloat16 *mixed, int token) {
  mixed += token * 2048;
  const auto rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int i = 0; i < 2048; i += 32) {
    const auto x = aie::load_v<32>(normalized + i);
    const auto old = aie::load_v<32>(shift + i);
    const auto coefficient = aie::load_v<32>(parameters + i);
    const auto product = aie::mul(aie::sub(old, x), coefficient).to_vector<float>();
    const auto value = aie::add(x, product);
    aie::store_v(mixed + i, aie::accum<accfloat, 32>(value).to_vector<bfloat16>());
    aie::store_v(shift + i, x);
  }
  aie::set_rounding(rounding);
}
