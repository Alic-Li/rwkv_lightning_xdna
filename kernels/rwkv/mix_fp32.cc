// SPDX-License-Identifier: Apache-2.0
#include <aie_api/aie.hpp>
extern "C" void rwkv7_mix(const float *x, const float *old, const float *coeff,
                          float *out) {
#ifdef RWKV_EXACT_FP32
  for(int i=0;i<2048;++i) out[i]=x[i]+(old[i]-x[i])*coeff[i];
#else
  for (int i = 0; i < 2048; i += 32) {
    auto a = aie::load_v<32>(x + i), b = aie::load_v<32>(old + i),
         c = aie::load_v<32>(coeff + i);
    auto delta = aie::sub(b, a);
    auto product = aie::mul(delta, c).to_vector<float>();
    aie::store_v(out + i, aie::add(a, product));
  }
#endif
}
extern "C" void rwkv7_mix_shift(const float *x, const float *old,
                                const float *coeff, float *out) {
  rwkv7_mix(x, old, coeff, out);
  for (int i = 0; i < 2048; i += 32)
    aie::store_v(out + 2048 + i, aie::load_v<32>(x + i));
}
extern "C" void rwkv7_mix_pair_shift(const float *pair, const float *coeff,
                                     float *out) {
  rwkv7_mix_shift(pair, pair + 2048, coeff, out);
}
