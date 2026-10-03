// SPDX-License-Identifier: Apache-2.0
#include <aie_api/aie.hpp>
template<int N> static void convert(const float *x, bfloat16 *out) {
  const auto rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int i = 0; i < N; i += 32)
    aie::store_v(out + i, aie::accum<accfloat, 32>(
        aie::load_v<32>(x + i)).to_vector<bfloat16>());
  aie::set_rounding(rounding);
}
extern "C" void rwkv7_channel_convert2048(const float *x, bfloat16 *out) { convert<2048>(x, out); }
extern "C" void rwkv7_channel_convert8192(const float *x, bfloat16 *out) { convert<8192>(x, out); }
