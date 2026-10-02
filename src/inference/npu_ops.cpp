// SPDX-License-Identifier: Apache-2.0
#include "ops_internal.hpp"
#include "rwkv/inference/model.hpp"
#include "rwkv/xdna/session.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
namespace rwkv::inference {
namespace {
std::filesystem::path checked(const std::filesystem::path &p) {
  std::ifstream file(p / "config.json");
  if (!file)
    throw std::runtime_error("Missing full NPU config: " + p.string());
  nlohmann::json j;
  file >> j;
  if (j.at("schema_version") != 1 || j.at("dtype") != "float32" ||
      j.at("vector_size") != 2048 || j.at("gemv_rows") != 256)
    throw std::runtime_error("Incompatible full NPU ABI");
  return p;
}
class FullNpu final : public RecurrentBackend {
  std::filesystem::path root_;
  std::unique_ptr<RecurrentBackend> recurrent_;
  std::unique_ptr<xdna::Session> ops_, prefill_;
  xdna::Session &device_session(std::unique_ptr<xdna::Session> &s,
                                const std::string &name) {
    if (!s)
      s = std::make_unique<xdna::Session>(root_ / name / "design.xclbin",
                                          root_ / name / "instructions.bin");
    return *s;
  }
  struct Gemv {
    xdna::Session session;
    std::vector<xdna::Buffer> buffers;
    Gemv(const std::filesystem::path &p, size_t k)
        : session(p / "design.xclbin", p / "instructions.bin"),
          buffers{{std::vector<uint8_t>(k * 4), false},
                  {std::vector<uint8_t>(256 * k * 4), false},
                  {std::vector<uint8_t>(256 * 4), true}} {}
  };
  std::map<size_t, std::unique_ptr<Gemv>> gemv_;
  std::vector<xdna::Buffer> op_buffers_;

public:
  explicit FullNpu(const std::filesystem::path &p)
      : root_(checked(p)), op_buffers_{{std::vector<uint8_t>(8208 * 4), false},
                                       {std::vector<uint8_t>(2048 * 4), true}} {
  }
  void release_device_cache() override {
    recurrent_.reset();
    ops_.reset();
    prefill_.reset();
    gemv_.clear();
  }
  Vector step(Vector &s, const Vector &r, const Vector &d, const Vector &k,
              const Vector &v, const Vector &a, const Vector &b,
              size_t n) override {
    if (!recurrent_)
      recurrent_ = npu_backend(root_ / "decode");
    return recurrent_->step(s, r, d, k, v, a, b, n);
  }
  std::vector<Vector> prefill(Vector &state, const std::vector<Vector> &r,
                              const std::vector<Vector> &d,
                              const std::vector<Vector> &k,
                              const std::vector<Vector> &v,
                              const std::vector<Vector> &a,
                              const std::vector<Vector> &b, size_t n) override {
    size_t t = r.size();
    if (!t)
      return {};
    const size_t c = r[0].size();
    if (n != 64 || !c || c % n || state.size() != c * n)
      throw std::runtime_error("NPU prefill shape mismatch");
    for (const auto *seq : {&r, &d, &k, &v, &a, &b}) {
      if (seq->size() != t)
        throw std::runtime_error("Prefill length mismatch");
      for (const auto &row : *seq)
        if (row.size() != c)
          throw std::runtime_error("Prefill channel mismatch");
    }
    std::vector<Vector> output(t, Vector(c));
    std::vector<xdna::Buffer> buffers{{std::vector<uint8_t>(4096 * 4), false},
                                      {std::vector<uint8_t>(6400 * 4), false},
                                      {std::vector<uint8_t>(5120 * 4), true}};
    Vector packed(6400);
    Vector next = state;
    for (size_t start = 0; start < t; start += 16)
      for (size_t h = 0; h < c / n; ++h) {
        std::fill(packed.begin(), packed.end(), 0);
        size_t count = std::min(t - start, size_t(16));
        for (size_t j = 0; j < count; ++j) {
          packed[j * 400] = 1;
          size_t slot = 0;
          for (const auto *seq : {&r, &d, &k, &v, &a, &b}) {
            std::copy_n((*seq)[start + j].data() + h * n, n,
                        packed.data() + j * 400 + 16 + slot * n);
            ++slot;
          }
        }
        std::memcpy(buffers[0].bytes.data(), next.data() + h * n * n, 4096 * 4);
        std::memcpy(buffers[1].bytes.data(), packed.data(), 6400 * 4);
        device_session(prefill_, "prefill").execute(buffers);
        std::memcpy(next.data() + h * n * n, buffers[2].bytes.data(), 4096 * 4);
        for (size_t j = 0; j < count; ++j)
          std::memcpy(output[start + j].data() + h * n,
                      buffers[2].bytes.data() + (4096 + j * n) * 4, n * 4);
      }
    state.swap(next);
    return output;
  }
  Vector element(Op op, const Vector &x, const Vector &y, const Vector &z,
                 const Vector &w, size_t group, float eps) override {
    validate_element(op, x, y, z, w, group, eps);
    const size_t size = x.size();
    if (!size || !group || size % group || group > 2048 || 2048 % group)
      throw std::runtime_error("Unsupported NPU element group");
    for (const auto *v : {&y, &z, &w})
      if (!v->empty() && v->size() != size)
        throw std::runtime_error("NPU element shape mismatch");
    Vector out(size), packed(8208);
    for (size_t start = 0; start < size; start += 2048) {
      const size_t count = std::min(size - start, size_t(2048));
      std::fill(packed.begin(), packed.end(), 0);
      packed[0] = int(op);
      packed[1] = count;
      packed[2] = group;
      packed[3] = eps;
      size_t slot = 0;
      for (const auto *v : {&x, &y, &z, &w}) {
        if (!v->empty())
          std::copy_n(v->data() + start, count,
                      packed.data() + 16 + slot * 2048);
        ++slot;
      }
      std::memcpy(op_buffers_[0].bytes.data(), packed.data(),
                  packed.size() * 4);
      device_session(ops_, "ops").execute(op_buffers_);
      std::memcpy(out.data() + start, op_buffers_[1].bytes.data(), count * 4);
    }
    return out;
  }
  Vector linear(const Vector &x, const Tensor &w, bool transpose) override {
    if (w.shape.size() != 2)
      throw std::runtime_error("NPU linear requires matrix");
    size_t inputs = transpose ? w.shape[0] : w.shape[1],
           outputs = transpose ? w.shape[1] : w.shape[0];
    if (x.size() != inputs)
      throw std::runtime_error("NPU linear shape mismatch");
    size_t k = ((inputs + 255) / 256) * 256;
    auto &g = gemv_[k];
    if (!g)
      g = std::make_unique<Gemv>(root_ / ("gemv-" + std::to_string(k)), k);
    Vector px(k);
    std::copy(x.begin(), x.end(), px.begin());
    std::memcpy(g->buffers[0].bytes.data(), px.data(), k * 4);
    Vector packed(256 * k), out(outputs);
    for (size_t start = 0; start < outputs; start += 256) {
      std::fill(packed.begin(), packed.end(), 0);
      size_t count = std::min(outputs - start, size_t(256));
      for (size_t r = 0; r < count; ++r)
        for (size_t col = 0; col < inputs; ++col) {
          size_t pos = ((r / 16) * (k / 256) + col / 256) * 4096 +
                       (r % 16) * 256 + col % 256;
          packed[pos] = w.data[transpose ? col * outputs + start + r
                                         : (start + r) * inputs + col];
        }
      std::memcpy(g->buffers[1].bytes.data(), packed.data(), packed.size() * 4);
      g->session.execute(g->buffers);
      std::memcpy(out.data() + start, g->buffers[2].bytes.data(), count * 4);
    }
    return out;
  }
};
} // namespace
std::unique_ptr<RecurrentBackend>
full_npu_backend(const std::filesystem::path &p) {
  return std::make_unique<FullNpu>(p);
}
} // namespace rwkv::inference
