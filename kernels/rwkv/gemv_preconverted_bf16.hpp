// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gemv_bf16.cc"
// Activations have already been rounded once with conv_even on a producer core.
// Keep the existing row accumulation and reduction tree unchanged.
static void gemv_preconverted(const bfloat16 *x, const bfloat16 *w, float *sum) {
  const auto rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int row = 0; row < 16; row += 4) {
    const auto sums = fold_rows<16>(mac_rows4<64, 256>(w + row * 256, x).to_vector<float>());
    for (int i = 0; i < 4; ++i) sum[row + i] += sums[i];
  }
  aie::set_rounding(rounding);
}
