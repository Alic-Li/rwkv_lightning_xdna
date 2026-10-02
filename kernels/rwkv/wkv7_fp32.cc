// SPDX-License-Identifier: Apache-2.0
// One RWKV-7 head, explicit [key,value] state, no reduced-precision state.
// packed = [r, decay, k, v, a=-kk, b=kk*alpha], each 64 float32 values.
// output = [updated_state (4096), y (64)]. No persistent device globals.
extern "C" void rwkv7_wkv_fp32(const float *state, const float *packed,
                               float *output) {
  constexpr int N = 64;
  const float *r = packed, *w = packed + N, *k = packed + 2 * N;
  const float *v = packed + 3 * N, *a = packed + 4 * N, *b = packed + 5 * N;
  for (int j = 0; j < N; ++j) {
    float sa = 0;
    for (int i = 0; i < N; ++i)
      sa += a[i] * state[i * N + j];
    float y = 0;
    for (int i = 0; i < N; ++i) {
      const float next = state[i * N + j] * w[i] + b[i] * sa + k[i] * v[j];
      output[i * N + j] = next;
      y += r[i] * next;
    }
    output[N * N + j] = y;
  }
}
