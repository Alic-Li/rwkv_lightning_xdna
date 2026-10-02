// SPDX-License-Identifier: Apache-2.0
#include <aie_api/aie.hpp>
// Experimental parallel value lanes; preserve key accumulation order.
// AIE FP32 vector primitives can round differently from the scalar reference.
// FP32 state is retained; no reassociation across recurrence steps.
extern "C" void rwkv7_wkv_fp32(const float *state, const float *p, float *out) {
  const float *r = p, *w = p + 64, *k = p + 128, *v = p + 192, *a = p + 256,
              *b = p + 320;
  for (int j = 0; j < 64; j += 32) {
    auto sa = aie::zeros<float, 32>();
    for (int i = 0; i < 64; ++i)
      sa = aie::add(sa, aie::mul(aie::load_v<32>(state + i * 64 + j), a[i])
                            .to_vector<float>());
    auto y = aie::zeros<float, 32>();
    auto vv = aie::load_v<32>(v + j);
    for (int i = 0; i < 64; ++i) {
      auto next = aie::add(aie::mul(aie::load_v<32>(state + i * 64 + j), w[i])
                               .to_vector<float>(),
                           aie::mul(sa, b[i]).to_vector<float>());
      next = aie::add(next, aie::mul(vv, k[i]).to_vector<float>());
      aie::store_v(out + i * 64 + j, next);
      y = aie::add(y, aie::mul(next, r[i]).to_vector<float>());
    }
    aie::store_v(out + 4096 + j, y);
  }
}
