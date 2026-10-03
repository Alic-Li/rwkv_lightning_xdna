// SPDX-License-Identifier: Apache-2.0
#include "ffn_pipeline_bf16.cc"
extern "C" void rwkv7_ffn_activate_block256(const float *raw, float *out, int block) {
  relu_squared_fp32<256>(raw + block * 256, out);
}
extern "C" void rwkv7_ffn_copy256(const float *x, float *out) {
  for (int i = 0; i < 256; i += 32)
    aie::store_v(out + i, aie::load_v<32>(x + i));
}
