// SPDX-License-Identifier: Apache-2.0
#include "ffn_pipeline_bf16.cc"
extern "C" void rwkv7_prefill_output_tile(const float *x, const bfloat16 *w,
                                          float *out, int row, int col) {
  for (int token = 0; token < 2; ++token)
    rwkv7_ffn_value_tile(x + token * 2048, w, out + token * 256, row, col);
}
extern "C" void rwkv7_prefill_output_zero(float *out) {
  for (int i = 0; i < 512; i += 32)
    aie::store_v(out + i, aie::zeros<float, 32>());
}
extern "C" void rwkv7_prefill_output_residual(const float *projection,
                                              const float *r0, const float *r1,
                                              float *out, int core) {
  for (int token = 0; token < 2; ++token)
    for (int i = 0; i < 256; i += 32) {
      auto v = aie::load_v<32>(projection + token * 256 + i);
      aie::store_v(out + token * 512 + i, v);
      aie::store_v(out + token * 512 + 256 + i,
                   aie::add(v, aie::load_v<32>((token == 0 ? r0 : r1) + core * 256 + i)));
    }
}
