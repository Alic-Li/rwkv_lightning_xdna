// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rwkv::inference {
using Vector = std::vector<float>;
struct Tensor {
  std::vector<size_t> shape;
  Vector data;
};
// Unmodified RWKV-7 checkpoint names/layout; all tensors decoded to float32.
class Weights {
public:
  explicit Weights(const std::filesystem::path &path);
  const Tensor &at(const std::string &name) const;
  size_t layers() const { return layers_; }
  size_t channels() const { return channels_; }
  size_t heads() const { return heads_; }
  size_t head_size() const { return channels_ / heads_; }
  size_t vocabulary() const { return vocabulary_; }

private:
  std::map<std::string, Tensor> tensors_;
  size_t layers_ = 0, channels_ = 0, heads_ = 0, vocabulary_ = 0;
};
struct LayerState {
  Vector attention_shift, ffn_shift, matrix;
};
struct State {
  std::vector<LayerState> layers;
};
// Matrix layout: [head, key, value]. decay is exp(-exp(-0.5)*sigmoid(w)).
enum class Op {
  Add,
  Mix,
  Tanh,
  Sigmoid,
  Decay,
  Multiply,
  ValueResidual,
  ReluSquared,
  KeyScale,
  NormalizeKey,
  Rkv,
  Norm,
  Negate
};
class RecurrentBackend {
public:
  virtual ~RecurrentBackend() = default;
  // Release idle device contexts before switching from prefill to resident
  // decode.
  virtual void release_device_cache() {}
  virtual std::vector<Vector>
  prefill(Vector &state, const std::vector<Vector> &r,
          const std::vector<Vector> &decay, const std::vector<Vector> &k,
          const std::vector<Vector> &v, const std::vector<Vector> &a,
          const std::vector<Vector> &b, size_t head_size);
  virtual Vector linear(const Vector &, const Tensor &,
                        bool transposed = false);
  virtual Vector element(Op, const Vector &, const Vector &y = {},
                         const Vector &z = {}, const Vector &w = {},
                         size_t group = 1, float epsilon = 0);
  virtual Vector norm(const Vector &x, const Tensor &w, const Tensor &b,
                      size_t group, float epsilon);

  virtual Vector step(Vector &state, const Vector &r, const Vector &decay,
                      const Vector &k, const Vector &v, const Vector &a,
                      const Vector &b, size_t head_size) = 0;
};
std::unique_ptr<RecurrentBackend> cpu_backend();
std::unique_ptr<RecurrentBackend>
npu_backend(const std::filesystem::path &artifacts);
std::unique_ptr<RecurrentBackend>
full_npu_backend(const std::filesystem::path &artifacts);
class Model {
public:
  Model(const Weights &weights, RecurrentBackend &backend)
      : w_(weights), backend_(backend) {}
  State initial_state() const;
  // Sequential prefill and decode use exactly the same token transition.
  Vector forward(int token, State &state) const;
  std::vector<Vector> prefill(const std::vector<int> &tokens,
                              State &state) const;

private:
  std::vector<Vector> evaluate(const std::vector<int> &tokens, State &state,
                               bool prefill) const;
  const Weights &w_;
  RecurrentBackend &backend_;
};
} // namespace rwkv::inference
