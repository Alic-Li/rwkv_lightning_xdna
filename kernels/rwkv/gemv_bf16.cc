// SPDX-License-Identifier: Apache-2.0
// Reuse the upstream BF16 GEMV MAC and reduction, retaining FP32 partial sums
// between K tiles. Weights are BF16; activation rounding is nearest-even.
#define DIM_K 256
#include "../../third_party/mlir-aie/aie_kernels/linalg/mv_bf16.cc"

extern "C" void rwkv7_gemv_tile(const float *x, const bfloat16 *w, float *sum) {
  const auto rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  alignas(64) bfloat16 xb[256];
  for (int i = 0; i < 256; i += 32)
    aie::store_v(
        xb + i,
        aie::accum<accfloat, 32>(aie::load_v<32>(x + i)).to_vector<bfloat16>());
  for (int row = 0; row < 16; row += 4) {
    const auto sums =
        fold_rows<16>(mac_rows4<64, 256>(w + row * 256, xb).to_vector<float>());
    for (int i = 0; i < 4; ++i)
      sum[row + i] += sums[i];
  }
  aie::set_rounding(rounding);
}
extern "C" void rwkv7_zero(float *sum) {
  for (int i = 0; i < 16; ++i)
    sum[i] = 0;
}
extern "C" void rwkv7_relu_squared(float *out) {
  for (int i = 0; i < 16; ++i) {
    float a = out[i] > 0 ? out[i] : 0;
    out[16 + i] = a * a;
  }
}
