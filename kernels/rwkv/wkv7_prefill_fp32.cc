// SPDX-License-Identifier: Apache-2.0
// Separate 16-token prefill transition. State lives in the output object
// throughout the sequence; padding steps do not change state.
extern "C" void rwkv7_prefill_init(const float *state, float *out) {
  for (int i = 0; i < 4096; ++i)
    out[i] = state[i];
  for (int i = 4096; i < 5120; ++i)
    out[i] = 0;
}
extern "C" void rwkv7_prefill_step(const float *input, float *out, int time) {
  if (input[0] == 0)
    return;
  const float *r = input + 16, *w = r + 64, *k = w + 64, *v = k + 64,
              *a = v + 64, *b = a + 64;
  for (int j = 0; j < 64; ++j) {
    float sa = 0;
    for (int i = 0; i < 64; ++i)
      sa += a[i] * out[i * 64 + j];
    float y = 0;
    for (int i = 0; i < 64; ++i) {
      float next = out[i * 64 + j] * w[i] + b[i] * sa + k[i] * v[j];
      out[i * 64 + j] = next;
      y += r[i] * next;
    }
    out[4096 + time * 64 + j] = y;
  }
}
