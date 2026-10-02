// SPDX-License-Identifier: Apache-2.0
#include <aie_api/aie.hpp>
// Sixteen output rows and a 256-wide input slice. Accumulate all K slices
// on device. FP32 vector multiply/accumulate; never round input to BF16.
extern "C" void rwkv7_gemv_tile(const float *x, const float *w, float *sum) {
  for (int row = 0; row < 16; ++row) {
    auto acc = aie::mul(aie::load_v<32>(x), aie::load_v<32>(w + row * 256));
    for (int col = 32; col < 256; col += 32)
      acc = aie::mac(acc, aie::load_v<32>(x + col),
                     aie::load_v<32>(w + row * 256 + col));
    sum[row] += aie::reduce_add(acc.to_vector<float>());
  }
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
