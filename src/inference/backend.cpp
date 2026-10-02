// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/model.hpp"
#include "rwkv/xdna/session.hpp"
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
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
// Reference NPU design processes one 64x64 head per dispatch. State is explicit
// host-owned input/output, so reset, branching and interleaving are
// deterministic.
class Npu final : public RecurrentBackend {
  xdna::Session session_;
  std::vector<xdna::Buffer> buffers_;

public:
  explicit Npu(const std::filesystem::path &p)
      : session_(p / "design.xclbin", p / "instructions.bin") {
    std::ifstream in(p / "config.json");
    nlohmann::json j;
    in >> j;
    if (j.at("schema_version") != 1 || j.at("head_size") != 64 ||
        j.at("dtype") != "float32" || j.at("layout") != "key_value")
      throw std::runtime_error("Incompatible WKV NPU artifact");
    buffers_ = {{std::vector<uint8_t>(4096 * 4), false},
                {std::vector<uint8_t>(384 * 4), false},
                {std::vector<uint8_t>(4160 * 4), true}};
  }
  Vector step(Vector &s, const Vector &r, const Vector &w, const Vector &k,
              const Vector &v, const Vector &a, const Vector &b,
              size_t n) override {
    validate(s, r, w, k, v, a, b, n);
    if (n != 64)
      throw std::runtime_error("NPU WKV requires head_size=64");
    Vector next(s.size()), y(r.size());
    for (size_t h = 0; h < r.size() / n; ++h) {
      std::memcpy(buffers_[0].bytes.data(), s.data() + h * n * n, n * n * 4);
      const Vector *inputs[] = {&r, &w, &k, &v, &a, &b};
      for (size_t i = 0; i < 6; ++i)
        std::memcpy(buffers_[1].bytes.data() + i * n * 4,
                    inputs[i]->data() + h * n, n * 4);
      session_.execute(buffers_);
      std::memcpy(next.data() + h * n * n, buffers_[2].bytes.data(), n * n * 4);
      std::memcpy(y.data() + h * n, buffers_[2].bytes.data() + n * n * 4,
                  n * 4);
    }
    s.swap(next);
    return y;
  }
};
} // namespace
std::unique_ptr<RecurrentBackend> cpu_backend() {
  return std::make_unique<Cpu>();
}
std::unique_ptr<RecurrentBackend> npu_backend(const std::filesystem::path &p) {
  return std::make_unique<Npu>(p);
}
} // namespace rwkv::inference
