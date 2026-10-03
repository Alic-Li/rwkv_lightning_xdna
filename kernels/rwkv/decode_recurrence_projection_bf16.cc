// SPDX-License-Identifier: Apache-2.0
#include "ffn_pipeline_bf16.cc"
extern "C" void rwkv7_decode_projection_tile(const float *input, const bfloat16 *w,
                                             float *out, int row, int col) {
  alignas(64) float x[256];
  for (int lane = 0; lane < 4; ++lane) {
    const int head = col * 4 + lane;
    const int offset = head < 20 ? (head % 4) * 320 + (head / 4) * 64
        : 1280 + ((head - 20) % 3) * 256 + ((head - 20) / 3) * 64;
    for (int j = 0; j < 64; j += 32)
      aie::store_v(x + lane * 64 + j, aie::load_v<32>(input + offset + j));
  }
  rwkv7_gemv_tile(x, w, out + row * 16);
}
extern "C" void rwkv7_decode_projection_zero(float *out) {
  for (int i = 0; i < 256; i += 32) aie::store_v(out + i, aie::zeros<float, 32>());
}
extern "C" void rwkv7_decode_projection_residual(float *out, const float *input, int core) {
  for (int i = 0; i < 256; i += 32)
    aie::store_v(out + 256 + i,
        aie::add(aie::load_v<32>(out + i), aie::load_v<32>(input + 2048 + core * 256 + i)));
}
extern "C" void rwkv7_copy1280(const float *in, float *out) {
  for (int i = 0; i < 1280; i += 32) aie::store_v(out + i, aie::load_v<32>(in + i));
}
extern "C" void rwkv7_copy768(const float *in, float *out) {
  for (int i = 0; i < 768; i += 32) aie::store_v(out + i, aie::load_v<32>(in + i));
}
