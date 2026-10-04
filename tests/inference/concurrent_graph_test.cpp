// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/graph.hpp"
#include "rwkv/xdna/session.hpp"
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace rwkv::inference;
// Compare simultaneous, different requests against isolated device executions.
// Exact logits and final state detect shared BOs, lost updates and state leakage.
int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: concurrent-graph-test MODEL KERNEL_ROOT");
    Weights w(argv[1]);
    auto backend = cpu_backend();
    const auto zero = Model(w, *backend).initial_state();
    const std::vector<std::vector<int>> inputs{{1, 2, 7, 9}, {42, 17, 3, 8}};
    std::vector<std::unique_ptr<DecodeGraph>> graphs;
    std::vector<std::vector<Vector>> golden(2), actual(2);
    std::vector<State> expected_state(2), actual_state(2);
    for (size_t i = 0; i < 2; ++i) {
      graphs.push_back(std::make_unique<DecodeGraph>(w, argv[2]));
      graphs[i]->load_state(zero);
      for (int token : inputs[i]) golden[i].push_back(graphs[i]->replay_resident(token));
      expected_state[i] = graphs[i]->export_state();
      graphs[i]->load_state(zero);
    }
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::vector<std::future<void>> tasks;
    try {
      for (size_t i = 0; i < 2; ++i)
        tasks.push_back(std::async(std::launch::async, [&, i] {
          gate.wait();
          for (int token : inputs[i]) actual[i].push_back(graphs[i]->replay_resident(token));
          actual_state[i] = graphs[i]->export_state();
        }));
    } catch (...) { start.set_value(); throw; }
    start.set_value();
    for (auto &task : tasks) task.get();
    for (size_t i = 0; i < 2; ++i) {
      if (actual[i] != golden[i]) throw std::runtime_error("Concurrent logits differ from isolated execution");
      for (size_t l = 0; l < w.layers(); ++l) {
        const auto &a = actual_state[i].layers[l], &b = expected_state[i].layers[l];
        if (a.attention_shift != b.attention_shift || a.ffn_shift != b.ffn_shift || a.matrix != b.matrix)
          throw std::runtime_error("Concurrent recurrent state differs from isolated execution");
      }
    }
    const auto peak = rwkv::xdna::dispatch_concurrency_peak();
    if (peak < 2) throw std::runtime_error("No simultaneous NPU submissions observed");
    std::cout << "Independent concurrent requests passed: exact logits/state, peak pending NPU commands=" << peak << '\n';
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
