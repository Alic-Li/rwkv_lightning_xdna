// SPDX-License-Identifier: Apache-2.0
#include "prefill_output_bf16.cc"
// Decode each weight tile once, then reuse its exact BF16 integer values for
// both tokens. Apply the FP16-expanded row scale after all eight K tiles.
extern "C" void rwkv7_prefill_output_int8_tile(const float *x, const uint8_t *packed,
                                               float *out, int row, int col) {
  alignas(64) bfloat16 w[4096];
  for (int i = 0; i < 4096; i += 64) {
    auto q = aie::load_v<64>(reinterpret_cast<const int8_t *>(packed + i));
    aie::accum<accfloat, 64> f(aie::to_float(q, 0));
    aie::store_v(w + i, f.to_vector<bfloat16>());
  }
  for (int token = 0; token < 2; ++token) {
    float *sum = out + token * 256 + row * 16;
    rwkv7_gemv_tile(x + token * 2048 + col * 256, w, sum);
    if (col == 7) {
      auto scale = aie::load_v<16>(reinterpret_cast<const float *>(packed + 4096));
      aie::store_v(sum, aie::mul(aie::load_v<16>(sum), scale).to_vector<float>());
    }
  }
}

// Projection occupies the first 512 floats of the 1024-float output. Process
// token 1 first so writing token 0's residual cannot overwrite its projection.
extern "C" void rwkv7_prefill_output_int8_finish(float *out, const float *input, int core) {
  const float *r0 = input + 4096, *r1 = input + 6144;
  for (int token = 1; token >= 0; --token)
    for (int i = 0; i < 256; i += 32) {
      auto v = aie::load_v<32>(out + token * 256 + i);
      aie::store_v(out + token * 512 + i, v);
      aie::store_v(out + token * 512 + 256 + i,
                   aie::add(v, aie::load_v<32>((token == 0 ? r0 : r1) + core * 256 + i)));
    }
}
