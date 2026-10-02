// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rwkv/inference/model.hpp"
namespace rwkv::inference {
struct GraphStats {
  size_t nodes = 0, buffers = 0, replays = 0;
};
// In-process, fixed-shape decode graph. Capture records typed operations and
// buffer dependencies without recording example tensor contents. Replay binds
// the new token/state and executes the same topological plan. Currently each
// node submits ordinary reusable XRT runs; this is not a runlist submission.
// Weights/backend must outlive the graph. One graph instance is serial-only.
class DecodeGraph {
public:
  DecodeGraph(const Weights &, RecurrentBackend &);
  ~DecodeGraph();
  DecodeGraph(const DecodeGraph &) = delete;
  DecodeGraph &operator=(const DecodeGraph &) = delete;
  Vector replay(int token, State &state);
  GraphStats stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace rwkv::inference
