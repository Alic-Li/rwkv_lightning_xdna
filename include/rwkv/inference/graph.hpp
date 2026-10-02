// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rwkv/inference/model.hpp"
namespace rwkv::inference {
struct GraphStats {
  size_t nodes = 0, buffers = 0, replays = 0;
  size_t device_runs = 0, resident_bytes = 0;
  size_t replay_upload_bytes = 0, replay_download_bytes = 0;
};
// In-process, fixed-shape decode graph. Capture records typed operations and
// buffer dependencies without recording example tensor contents. Replay binds
// the new token/state and executes the same topological plan. Currently each
// node submits ordinary reusable XRT runs; this is not a runlist submission.
// With resident_artifacts, all intermediate activations and packed weights stay
// in NPU-accessible BOs. Only embedding/state inputs and logits/state outputs
// cross the host boundary on replay. The backend is used only in host mode.
// Weights/backend must outlive the graph. One graph instance is serial-only.
class DecodeGraph {
public:
  DecodeGraph(const Weights &, RecurrentBackend &,
              const std::filesystem::path &resident_artifacts = {});
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
