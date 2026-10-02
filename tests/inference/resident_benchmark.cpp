// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/graph.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace rwkv::inference;
using Clock = std::chrono::steady_clock;
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("weights and kernel directory required");
    Weights weights(argv[1]);
    auto backend = full_npu_backend(argv[2]);
    Model model(weights, *backend);
    DecodeGraph baseline(weights, *backend);
    const std::vector<int> tokens{1, 2, 7, 9};
    std::vector<Vector> references;
    std::vector<State> states;
    std::vector<double> baseline_seconds;
    auto a = model.initial_state();
    for (int token : tokens) {
      auto begin = Clock::now();
      references.push_back(baseline.replay(token, a));
      baseline_seconds.push_back(
          std::chrono::duration<double>(Clock::now() - begin).count());
      states.push_back(a);
    }
    // Run the reference phase first, then release its idle hardware contexts.
    backend->release_device_cache();
    auto start = Clock::now();
    DecodeGraph resident(weights, *backend, argv[2]);
    std::cout << "resident_prepare_seconds="
              << std::chrono::duration<double>(Clock::now() - start).count()
              << std::endl;
    const auto stats = resident.stats();
    std::cout << "runs=" << stats.persistent_runs
              << " resident_bytes=" << stats.resident_bytes
              << " root_bos=" << stats.root_bos
              << " upload_bytes=" << stats.persistent_upload_bytes
              << " download_bytes=" << stats.persistent_download_bytes
              << std::endl;
    resident.load_state(model.initial_state());
    double worst = 0, baseline_total = 0, resident_total = 0;
    auto check = [&](const Vector &x, const Vector &y,
                     const std::string &label) {
      if (x.size() != y.size())
        throw std::runtime_error("shape mismatch");
      for (size_t i = 0; i < x.size(); ++i) {
        double error = std::abs(double(x[i]) - y[i]);
        worst = std::max(worst, error);
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]) ||
            error > 2e-6 + 2e-5 * std::abs(y[i])) {
          std::cerr << label << " index=" << i << " actual=" << x[i]
                    << " expected=" << y[i] << " abs=" << error << std::endl;
          throw std::runtime_error("Resident logits/state mismatch");
        }
      }
    };
    int iteration = 0;
    for (int token : tokens) {
      const auto &expected = references[iteration];
      const auto &a = states[iteration];
      double ordinary = baseline_seconds[iteration];
      start = Clock::now();
      auto actual = resident.replay_resident(token);
      double optimized =
          std::chrono::duration<double>(Clock::now() - start).count();
      auto b = resident.export_state(); // diagnostic readback excluded from
                                        // decode timing
      check(actual, expected, "logits");
      for (size_t i = 0; i < a.layers.size(); ++i) {
        check(a.layers[i].matrix, b.layers[i].matrix,
              "matrix layer " + std::to_string(i));
        check(a.layers[i].attention_shift, b.layers[i].attention_shift,
              "attention shift layer " + std::to_string(i));
        check(a.layers[i].ffn_shift, b.layers[i].ffn_shift,
              "ffn shift layer " + std::to_string(i));
      }
      std::cout << "token=" << token << " baseline_seconds=" << ordinary
                << " resident_seconds=" << optimized
                << " max_abs_error=" << worst << std::endl;
      if (iteration++) {
        baseline_total += ordinary;
        resident_total += optimized;
      }
    }
    std::cout << "measured_tokens=3 baseline_seconds=" << baseline_total
              << " resident_seconds=" << resident_total
              << " speedup=" << baseline_total / resident_total
              << " resident_tokens_per_second=" << 3 / resident_total
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
