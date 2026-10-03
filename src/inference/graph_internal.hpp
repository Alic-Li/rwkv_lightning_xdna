// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rwkv/inference/graph.hpp"
#include "rwkv/xdna/session.hpp"
#include <array>
#include <limits>

namespace rwkv::inference {
struct DecodeGraph::Impl {
  using Id = size_t;
  static constexpr Id none = std::numeric_limits<Id>::max();
  enum class Kind { Element, Linear, Recurrent };
  struct Buffer {
    size_t size;
    const Vector *constant = nullptr;
    Vector value;
  };
  struct Node {
    Kind kind;
    Op op = Op::Add;
    std::array<Id, 7> inputs{none, none, none, none, none, none, none};
    Id output = none;
    const Tensor *weight = nullptr;
    bool transpose = false;
    size_t group = 1;
    float epsilon = 0;
    int layer = -1;
  };
  struct StateBinding {
    Id old_attention, old_ffn, matrix, new_attention, new_ffn;
  };
  const Weights &weights;
  RecurrentBackend *backend;
  WeightMode weight_mode;
  std::vector<Buffer> buffers;
  std::vector<Node> nodes;
  int recording_layer = -1;
  std::vector<StateBinding> states;
  std::map<const Vector *, Id> constants;
  Id embedding, logits;
  size_t replays = 0;
  DecodeGraph::Trace trace;
  DecodeGraph::ProjectionTrace projection_trace;
  std::vector<size_t> node_run_ends;
  bool device_failed = false, resident_state_valid = false;
  const Vector empty;
  std::map<std::string, std::unique_ptr<xdna::Session>> sessions;
  std::vector<xdna::DeviceBuffer> device_buffers;
  std::vector<xdna::DeviceRun> runs;
  size_t resident_bytes = 0, upload_bytes = 0, download_bytes = 0, root_bos = 0;
  xdna::Session &session(const std::filesystem::path &root,
                         const std::string &name);
  xdna::DeviceBuffer initialized(xdna::Session &s, const Vector &v);
  xdna::DeviceBuffer initialized_bf16(xdna::Session &s, const Vector &v);
  struct ResidentLayout {
    std::map<size_t, std::vector<xdna::DeviceBuffer>> value_args;
    std::map<size_t, xdna::DeviceBuffer> recurrence_stages;
  };
  void prepare_resident_arenas(const std::filesystem::path &, ResidentLayout &);
  void prepare_resident_runs(const std::filesystem::path &, ResidentLayout &);
  void prepare_resident(const std::filesystem::path &root);
  Id allocate(size_t size);
  Id constant(const Vector &v);
  const Vector &read(Id id) const;
  Id element(Op op, Id x, Id y = none, Id z = none, Id w = none,
             size_t group = 1, float eps = 0);
  Id norm(Id x, const Tensor &w, const Tensor &b, size_t group, float eps);
  Id linear(Id x, const Tensor &w, bool transpose = false);
  Id recurrent(Id state, Id r, Id d, Id k, Id v, Id a, Id b);
  Impl(const Weights &w, RecurrentBackend *b,
       const std::filesystem::path &resident, WeightMode = WeightMode::BFloat16);
  void validate_state(const State &state) const;
  void load_state(const State &state);
  State export_state() const;
  Vector replay(int token, State &state, bool persistent = false);
};
} // namespace rwkv::inference
