// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/graph.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace rwkv::inference;
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("weights and kernel directory required");
    Weights w(argv[1]);
    for (int mode : {0, 1, 2}) {
      bool hardware = mode != 0;
      auto b = hardware ? full_npu_backend(argv[2]) : cpu_backend();
      Model eager(w, *b);
      DecodeGraph graph(w, *b,
                        mode == 2 ? std::filesystem::path(argv[2])
                                  : std::filesystem::path{});
      const auto captured = graph.stats();
      double worst = 0;
      auto check = [&](const Vector &a, const Vector &b) {
        if (a.size() != b.size())
          throw std::runtime_error("shape mismatch");
        for (size_t i = 0; i < a.size(); ++i) {
          double error = std::abs(a[i] - b[i]);
          worst = std::max(worst, error);
          if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
              error > 2e-6 + 2e-5 * std::abs(b[i]))
            throw std::runtime_error("Graph/eager mismatch");
        }
      };
      auto statecheck = [&](const State &a, const State &b) {
        for (size_t i = 0; i < a.layers.size(); ++i) {
          check(a.layers[i].matrix, b.layers[i].matrix);
          check(a.layers[i].attention_shift, b.layers[i].attention_shift);
          check(a.layers[i].ffn_shift, b.layers[i].ffn_shift);
        }
      };
      auto a = eager.initial_state(), c = eager.initial_state();
      for (int id : {1, 2, 7, 9, 3, 11, 4}) {
        check(graph.replay(id, a), eager.forward(id, c));
        statecheck(a, c);
      }
      // New request with unrelated state must not reuse the last request's
      // buffers.
      a = eager.initial_state();
      c = eager.initial_state();
      eager.prefill({7, 5, 8, 2, 1}, a);
      eager.prefill({7, 5, 8, 2, 1}, c);
      auto branch = a;
      for (int id : {9, 4, 6}) {
        check(graph.replay(id, a), eager.forward(id, c));
        statecheck(a, c);
      }
      auto branch_ref = branch;
      check(graph.replay(17, branch), eager.forward(17, branch_ref));
      statecheck(branch, branch_ref);
      auto saved = a;
      bool rejected = false;
      try {
        graph.replay(-1, a);
      } catch (const std::exception &) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error("Invalid graph token accepted");
      statecheck(a, saved);
      rejected = false;
      try {
        graph.replay(static_cast<int>(w.vocabulary()), a);
      } catch (const std::exception &) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error("Out-of-range token accepted");
      statecheck(a, saved);
      auto malformed = a;
      malformed.layers.back().matrix.pop_back();
      auto malformed_saved = malformed;
      rejected = false;
      try {
        graph.replay(1, malformed);
      } catch (const std::exception &) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error("Malformed state accepted");
      statecheck(malformed, malformed_saved);
      if (mode == 2 &&
          (!captured.device_runs || !captured.resident_bytes ||
           !captured.replay_upload_bytes || !captured.replay_download_bytes))
        throw std::runtime_error("Resident graph statistics missing");
      if (graph.stats().nodes != captured.nodes ||
          graph.stats().buffers != captured.buffers ||
          graph.stats().replays != 11)
        throw std::runtime_error("Graph was rebuilt or replay count wrong");
      std::cout << (mode == 2  ? "Resident NPU"
                    : hardware ? "NPU"
                               : "CPU")
                << " graph: " << captured.nodes << " nodes, "
                << captured.buffers
                << " buffers; token changes, reset, branch, prefill/decode "
                   "passed; max_abs_error="
                << worst << '\n';
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
