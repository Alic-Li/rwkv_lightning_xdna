// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/model.hpp"
#include <stdexcept>

namespace rwkv::inference {
namespace {
void validate(const Vector &s, const Vector &r, const Vector &w,
              const Vector &k, const Vector &v, const Vector &a,
              const Vector &b, size_t n) {
  const size_t c = r.size();
  if (!n || !c || c % n || s.size() != c * n || w.size() != c ||
      k.size() != c || v.size() != c || a.size() != c || b.size() != c)
    throw std::runtime_error("WKV dimensions mismatch");
}
class Cpu final : public RecurrentBackend {
  Vector step(Vector &s, const Vector &r, const Vector &w, const Vector &k,
              const Vector &v, const Vector &a, const Vector &b,
              size_t n) override {
    validate(s, r, w, k, v, a, b, n);
    Vector y(r.size());
    for (size_t h = 0; h < r.size() / n; ++h) {
      for (size_t j = 0; j < n; ++j) {
        float sa = 0;
        for (size_t i = 0; i < n; ++i)
          sa += a[h * n + i] * s[(h * n + i) * n + j];
        float out = 0;
        for (size_t i = 0; i < n; ++i) {
          float &z = s[(h * n + i) * n + j];
          z = z * w[h * n + i] + b[h * n + i] * sa +
              k[h * n + i] * v[h * n + j];
          out += r[h * n + i] * z;
        }
        y[h * n + j] = out;
      }
    }
    return y;
  }
};
} // namespace
std::unique_ptr<RecurrentBackend> cpu_backend() {
  return std::make_unique<Cpu>();
}
} // namespace rwkv::inference
