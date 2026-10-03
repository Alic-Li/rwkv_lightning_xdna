// SPDX-License-Identifier: Apache-2.0
#include "gemv_bf16.cc"
#include "ffn_common.hpp"

// Same per-row CUDA W8A16 scale convention as ChannelMix. Keep FP32
// accumulation across all eight K tiles and apply the row scale only once.
extern "C" void rwkv7_output_int8_tile(const float *x, const uint8_t *packed,
                                      float *out, int row, int col) {
  alignas(64) bfloat16 w[4096];
  for (int i = 0; i < 4096; i += 64) {
    auto q = aie::load_v<64>(reinterpret_cast<const int8_t *>(packed + i));
    aie::accum<accfloat, 64> f(aie::to_float(q, 0));
    aie::store_v(w + i, f.to_vector<bfloat16>());
  }
  float *sum = out + row * 16;
  rwkv7_gemv_tile(x + col * 256, w, sum);
  if (col == 7) {
    auto scale = aie::load_v<16>(reinterpret_cast<const float *>(packed + 4096));
    aie::store_v(sum, aie::mul(aie::load_v<16>(sum), scale).to_vector<float>());
  }
}
