// SPDX-License-Identifier: Apache-2.0
#include "prefill_output_bf16.cc"
// Projection occupies the first 512 floats of the 1024-float output. Process
// token 1 first so writing token 0's residual cannot overwrite its projection.
extern "C" void rwkv7_pair_projection_residual(float *out, const float *input,
                                               int core) {
  const float *r0 = input + 4096, *r1 = input + 6144;
  for (int token = 1; token >= 0; --token)
    for (int i = 0; i < 256; i += 32) {
      auto v = aie::load_v<32>(out + token * 256 + i);
      aie::store_v(out + token * 512 + i, v);
      aie::store_v(out + token * 512 + 256 + i,
                   aie::add(v, aie::load_v<32>((token == 0 ? r0 : r1) +
                                               core * 256 + i)));
    }
}

extern "C" void rwkv7_recurrence_projection_tile(const float *input,
                                                 const bfloat16 *w, float *out,
                                                 int row, int col) {
  alignas(64) float x[256];
  // Gather from [head group, lane, head iteration, token, channel].
  for (int token = 0; token < 2; ++token) {
    for (int lane = 0; lane < 4; ++lane) {
      const int head = col * 4 + lane;
      const int offset =
          head < 20 ? (head % 4) * 640 + (head / 4) * 128
                    : 2560 + ((head - 20) % 3) * 512 + ((head - 20) / 3) * 128;
      for (int j = 0; j < 64; j += 32)
        aie::store_v(x + lane * 64 + j,
                     aie::load_v<32>(input + offset + token * 64 + j));
    }
    rwkv7_gemv_tile(x, w, out + token * 256 + row * 16);
  }
}

extern "C" void rwkv7_copy2560(const float *in, float *out) {
  for (int i = 0; i < 2560; i += 32)
    aie::store_v(out + i, aie::load_v<32>(in + i));
}
extern "C" void rwkv7_copy1536(const float *in, float *out) {
  for (int i = 0; i < 1536; i += 32)
    aie::store_v(out + i, aie::load_v<32>(in + i));
}
