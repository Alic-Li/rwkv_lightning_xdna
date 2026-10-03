// SPDX-License-Identifier: Apache-2.0
// A golden snapshot is recorded by the unchanged baseline binary, then verified
// by the candidate. Full first-token node traces cover every individual layer.
#include "rwkv/inference/graph.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
using namespace rwkv::inference;
using Clock = std::chrono::steady_clock;
struct Snapshot {
  bool record;
  std::fstream file;
  size_t checked = 0;
  Snapshot(const char *path, bool write)
      : record(write),
        file(path, std::ios::binary | (write ? std::ios::out | std::ios::trunc
                                             : std::ios::in)) {
    if (!file)
      throw std::runtime_error("Cannot open snapshot");
  }
  uint64_t number(uint64_t value) {
    if (record)
      file.write(reinterpret_cast<const char *>(&value), 8);
    else
      file.read(reinterpret_cast<char *>(&value), 8);
    if (!file)
      throw std::runtime_error("Truncated snapshot");
    return value;
  }
  nlohmann::json vector(const Vector &actual) {
    const auto size = number(actual.size());
    if (size != actual.size())
      throw std::runtime_error("Snapshot shape mismatch");
    Vector expected;
    if (record)
      file.write(reinterpret_cast<const char *>(actual.data()), size * 4);
    else {
      expected.resize(size);
      file.read(reinterpret_cast<char *>(expected.data()), size * 4);
    }
    if (!file)
      throw std::runtime_error("Truncated snapshot vector");
    double max_abs = 0, sum = 0, dot = 0, aa = 0, bb = 0;
    size_t worst_index = 0;
    for (size_t i = 0; i < size; ++i) {
      double a = actual[i], b = record ? a : expected[i];
      if (!std::isfinite(a) || !std::isfinite(b))
        throw std::runtime_error("Nonfinite snapshot");
      double e = std::abs(a - b);
      if (e > max_abs) { max_abs = e; worst_index = i; }
      sum += e;
      dot += a * b;
      aa += a * a;
      bb += b * b;
    }
    ++checked;
    // Cleanup is required to preserve the baseline arithmetic exactly. Existing
    // CPU oracle tolerances are independently checked by alignment/array tests.
    if (max_abs != 0)
      throw std::runtime_error("Cleanup changed output vector " +
                               std::to_string(checked) +
                               " " + nlohmann::json({{"max_abs", max_abs},
                                   {"index", worst_index},
                                   {"actual", actual[worst_index]},
                                   {"expected", expected[worst_index]}}).dump());
    return {
        {"max_abs", max_abs},
        {"mean_abs", size ? sum / size : 0},
        {"cosine", aa && bb ? dot / std::sqrt(aa * bb) : (aa == bb ? 1 : 0)}};
  }
};
int main(int argc, char **argv) {
  try {
    if (argc < 5 || argc > 6 ||
        (argc == 6 && std::string(argv[5]) != "--int8-ffn" &&
         std::string(argv[5]) != "--int8-ffn-output"))
      throw std::runtime_error("Usage: rwkv-cleanup-regression MODEL KERNELS "
                               "record|verify SNAPSHOT [--int8-ffn|--int8-ffn-output]");
    const bool int8 = argc == 6;
    const bool output_int8 = int8 && std::string(argv[5]) == "--int8-ffn-output";
    const std::string mode = argv[3];
    if (mode != "record" && mode != "verify")
      throw std::runtime_error("Invalid mode");
    Snapshot snapshot(argv[4], mode == "record");
    const uint64_t format = output_int8 ? 0x33564b5752474443ULL :
        int8 ? 0x32564b5752474443ULL : 0x31564b5752474443ULL;
    if (snapshot.number(format) != format)
      throw std::runtime_error("Snapshot version/precision mismatch");
    Weights weights(argv[1]);
    auto backend = cpu_backend();
    Model model(weights, *backend);
    DecodeGraph graph(weights, argv[2], output_int8 ? WeightMode::Int8FFNOutput :
        int8 ? WeightMode::Int8FFN : WeightMode::BFloat16);
    graph.load_state(model.initial_state());
    auto stats = graph.stats();
    if (stats.persistent_runs != (output_int8 ? 123 : 100))
      throw std::runtime_error("Unexpected resident run count");
    size_t traced = 0;
    graph.set_trace(
        [&](size_t node, const Vector &output, const Vector *state) {
          if (snapshot.number(node) != node)
            throw std::runtime_error("Node schedule changed");
          snapshot.vector(output);
          if (snapshot.number(state != nullptr) != (state != nullptr))
            throw std::runtime_error("State trace changed");
          if (state)
            snapshot.vector(*state);
          ++traced;
        });
    int next = 1;
    const int initial[] = {1, 2, 7, 9};
    double seconds = 0;
    for (int step = 1; step <= 128; ++step) {
      int token = int(snapshot.number(next));
      auto start = Clock::now();
      auto logits = graph.replay_resident(token);
      const double elapsed =
          std::chrono::duration<double>(Clock::now() - start).count();
      if (step > 1)
        seconds += elapsed;
      auto metrics = snapshot.vector(logits);
      if (step == 1) {
        graph.set_trace({});
        if (traced != stats.nodes)
          throw std::runtime_error("Incomplete node trace");
      }
      if (step == 1 || step == 8 || step == 32 || step == 128) {
        Vector flat;
        auto state = graph.export_state();
        for (const auto &layer : state.layers)
          for (const auto *v :
               {&layer.attention_shift, &layer.ffn_shift, &layer.matrix})
            flat.insert(flat.end(), v->begin(), v->end());
        std::cout << nlohmann::json({{"step", step},
                                     {"logits", metrics},
                                     {"state", snapshot.vector(flat)}})
                         .dump()
                  << std::endl;
      }
      next = step < 4 ? initial[step]
                      : int(std::max_element(logits.begin(), logits.end()) -
                            logits.begin());
    }
    if (!snapshot.record &&
        snapshot.file.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("Snapshot trailing data");
    // Branch, reset, and rejected calls must preserve explicit device state.
    auto equal_state = [](const State &a, const State &b) {
      if (a.layers.size() != b.layers.size())
        throw std::runtime_error("State layer mismatch");
      for (size_t i = 0; i < a.layers.size(); ++i)
        if (a.layers[i].matrix != b.layers[i].matrix ||
            a.layers[i].attention_shift != b.layers[i].attention_shift ||
            a.layers[i].ffn_shift != b.layers[i].ffn_shift)
          throw std::runtime_error("State branch mismatch");
    };
    auto branch = graph.export_state();
    auto expected = graph.replay_resident(7);
    auto advanced = graph.export_state();
    graph.load_state(branch);
    bool rejected = false;
    try {
      graph.replay_resident(-1);
    } catch (const std::exception &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("Invalid token accepted");
    auto malformed = branch;
    malformed.layers.back().matrix.pop_back();
    rejected = false;
    try {
      graph.load_state(malformed);
    } catch (const std::exception &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("Invalid state accepted");
    equal_state(branch, graph.export_state());
    if (graph.replay_resident(7) != expected)
      throw std::runtime_error("Branch logits mismatch");
    equal_state(advanced, graph.export_state());
    // Switch back to each diagnostic plan after optimized runs have overwritten
    // scratch arenas. Explicit state restoration must reproduce the same step.
    for (bool projection_only : {false, true}) {
      graph.load_state(branch);
      size_t observed = 0;
      if (projection_only)
        graph.set_projection_trace([&](size_t, const Vector &, const Tensor &, bool, const Vector &) { ++observed; });
      else graph.set_trace([&](size_t, const Vector &, const Vector *) { ++observed; });
      if (graph.stats().persistent_runs != 123 || graph.replay_resident(7) != expected)
        throw std::runtime_error("Diagnostic replay mismatch");
      equal_state(advanced, graph.export_state());
      if (!observed || (!projection_only && observed != stats.nodes))
        throw std::runtime_error("Diagnostic trace missing nodes");
      graph.set_trace({});
      graph.set_projection_trace({});
      if (graph.stats().persistent_runs != stats.persistent_runs)
        throw std::runtime_error("Optimized replay plan not restored");
    }
    auto host_state = branch;
    if (graph.replay(7, host_state) != expected)
      throw std::runtime_error("Host replay mismatch");
    equal_state(advanced, host_state);
    auto zero_state = model.initial_state();
    graph.load_state(zero_state);
    auto reset = graph.replay_resident(1);
    graph.load_state(zero_state);
    if (graph.replay_resident(1) != reset)
      throw std::runtime_error("Reset mismatch");
    std::cout << nlohmann::json(
                     {{"mode", mode},
                      {"weights", output_int8 ? "int8-ffn-output" : int8 ? "int8-ffn" : "bf16"},
                      {"status", "passed"},
                      {"first_token_nodes", traced},
                      {"vectors_checked", snapshot.checked},
                      {"runs_per_token", stats.persistent_runs},
                      {"resident_bytes", stats.resident_bytes},
                      {"upload_bytes", stats.persistent_upload_bytes},
                      {"download_bytes", stats.persistent_download_bytes},
                      {"measured_steps", 127},
                      {"mean_ms", seconds * 1000 / 127},
                      {"tokens_per_second", 127 / seconds}})
                     .dump()
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
