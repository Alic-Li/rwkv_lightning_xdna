// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rwkv/inference/model.hpp"
#include <functional>
namespace rwkv::inference {
struct GraphStats {
  size_t nodes = 0, buffers = 0, replays = 0;
  size_t device_runs = 0, resident_bytes = 0;
  size_t replay_upload_bytes = 0, replay_download_bytes = 0;
  size_t root_bos = 0, persistent_runs = 0;
  size_t persistent_upload_bytes = 0, persistent_download_bytes = 0;
};
// In-process, fixed-shape decode graph. Capture records typed operations and
// buffer dependencies without recording example tensor contents. Replay binds
// the new token/state and executes the same topological plan. Currently each
// execution stage submits reusable XRT runs; this is not a runlist submission.
// With resident_artifacts, all intermediate activations and packed weights stay
// in NPU-accessible BOs. Only embedding/state inputs and logits/state outputs
// cross the host boundary on replay(token, State&). replay_resident transfers
// only embedding/logits. The backend is used only in host mode.
// Weights/backend must outlive the graph. One graph instance is serial-only.
class DecodeGraph {
public:
  DecodeGraph(const Weights &, RecurrentBackend &,
              const std::filesystem::path &resident_artifacts = {});
  ~DecodeGraph();
  DecodeGraph(const DecodeGraph &) = delete;
  DecodeGraph &operator=(const DecodeGraph &) = delete;
  Vector replay(int token, State &state);
  // Diagnostic only: called after each node; recurrent nodes also expose FP32
  // state.
  using Trace = std::function<void(size_t, const Vector &, const Vector *)>;
  void set_trace(Trace trace);
  // Diagnostic projection oracle hook. Reads device inputs and outputs;
  // never enable it for performance measurements. Includes the preserved
  // intermediate tensors of fused low-rank projections.
  using ProjectionTrace = std::function<void(size_t, const Vector &,
                                             const Tensor &, bool, const Vector &)>;
  void set_projection_trace(ProjectionTrace trace);
  // Explicit device-owned request state. load once after prefill/reset, export
  // only for checkpoint/branch. No implicit pointer-identity state cache.
  void load_state(const State &);
  State export_state() const;
  Vector replay_resident(int token);
  GraphStats stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace rwkv::inference
