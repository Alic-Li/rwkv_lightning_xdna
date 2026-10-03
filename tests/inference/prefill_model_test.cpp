// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/graph.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
using namespace rwkv::inference;
using Json = nlohmann::json;
static void equal(const Vector &a, const Vector &b) {
  if (a.size() != b.size() || (!a.empty() && std::memcmp(a.data(), b.data(), a.size() * 4)))
    throw std::runtime_error("Bitwise vector mismatch");
}
static void equal(const State &a, const State &b) {
  if (a.layers.size() != b.layers.size()) throw std::runtime_error("State layer mismatch");
  for (size_t l = 0; l < a.layers.size(); ++l) {
    equal(a.layers[l].matrix, b.layers[l].matrix);
    equal(a.layers[l].attention_shift, b.layers[l].attention_shift);
    equal(a.layers[l].ffn_shift, b.layers[l].ffn_shift);
  }
}
struct Snapshot {
  bool record;
  std::fstream file;
  size_t vectors = 0;
  Snapshot(const char *path, bool write) : record(write),
      file(path, std::ios::binary | (write ? std::ios::out | std::ios::trunc : std::ios::in)) {
    if (!file) throw std::runtime_error("Cannot open snapshot");
    number(0x31504652474b5752ULL);
  }
  void number(uint64_t n) {
    if (record) file.write(reinterpret_cast<const char *>(&n), sizeof(n));
    else {
      uint64_t expected = 0;
      file.read(reinterpret_cast<char *>(&expected), sizeof(expected));
      if (expected != n) throw std::runtime_error("Snapshot shape/version mismatch");
    }
    if (!file) throw std::runtime_error("Snapshot I/O failure");
  }
  void vector(const Vector &v) {
    number(v.size());
    for (float f : v) if (!std::isfinite(f)) throw std::runtime_error("Nonfinite snapshot");
    if (record) file.write(reinterpret_cast<const char *>(v.data()), v.size() * 4);
    else {
      Vector expected(v.size());
      file.read(reinterpret_cast<char *>(expected.data()), expected.size() * 4);
      if (!file) throw std::runtime_error("Truncated snapshot");
      try { equal(v, expected); }
      catch (...) { throw std::runtime_error("Snapshot vector mismatch " + std::to_string(vectors)); }
    }
    if (!file) throw std::runtime_error("Snapshot write failure");
    ++vectors;
  }
  void state(const State &s) {
    number(s.layers.size());
    for (const auto &l : s.layers) {
      vector(l.attention_shift); vector(l.ffn_shift); vector(l.matrix);
    }
  }
};
int main(int argc, char **argv) {
  try {
    const bool chunk4 = argc > 1 && std::string(argv[argc - 1]) == "--chunk4";
    if (chunk4) --argc;
    const bool output_int8 = argc > 1 && std::string(argv[argc - 1]) == "--int8-ffn-output";
    const bool int8 = output_int8 || (argc > 1 && std::string(argv[argc - 1]) == "--int8-ffn");
    if (int8) --argc;
    if (argc != 5 || (std::string(argv[3]) != "record" && std::string(argv[3]) != "verify"))
      throw std::runtime_error("Usage: rwkv-prefill-model-test MODEL KERNELS record|verify SNAPSHOT [--int8-ffn|--int8-ffn-output]");
    const bool record = std::string(argv[3]) == "record";
    Snapshot snapshot(argv[4], record);
    Weights weights(argv[1]);
    auto backend = cpu_backend();
    Model model(weights, *backend);
    DecodeGraph graph(weights, argv[2], output_int8 ? WeightMode::Int8FFNOutput :
                      int8 ? WeightMode::Int8FFN : WeightMode::BFloat16,
                      record ? PrefillMode::Sequential : chunk4 ? PrefillMode::Chunked4 : PrefillMode::Batched2);
    auto rejected = [&](auto action) {
      bool failed = false;
      try { action(); } catch (const std::exception &) { failed = true; }
      if (!failed) throw std::runtime_error("Expected rejected call");
    };
    if (!record) rejected([&] { graph.prefill_resident({1, 2}); });
    auto zero = model.initial_state();
    if (!record && chunk4) {
      // A full chunk from zero state must agree, not only the warm-state
      // snapshots and short tails. Both reference and candidate execute on NPU.
      graph.load_state(zero);
      Vector expected;
      for (int token : {1,18,35,52}) expected = graph.replay_resident(token);
      auto expected_state = graph.export_state();
      graph.load_state(zero);
      equal(graph.prefill_resident({1,18,35,52}), expected);
      equal(graph.export_state(), expected_state);
      std::cout << Json({{"zero_state_chunk_bitwise", "passed"}}).dump() << std::endl;
    }
    graph.load_state(zero);
    for (int token : {1, 2, 7}) graph.replay_resident(token);
    const auto warm = graph.export_state();
    std::vector<size_t> lengths{0,1,2,3,8,17,32};
    if (chunk4) lengths.insert(lengths.end(), {4,5,7,9,64});
    for (size_t length : lengths) {
      // Zero-state pair plus nonzero-state prompts exercise state sharing and
      // odd tails. All cases also verify the handoff to resident decode.
      graph.load_state(length == 2 ? zero : warm);
      std::vector<int> tokens;
      for (size_t i = 0; i < length; ++i) tokens.push_back(int((17 * i + 1) % weights.vocabulary()));
      Vector logits;
      if (record) for (int token : tokens) logits = graph.replay_resident(token);
      else logits = graph.prefill_resident(tokens);
      snapshot.number(length);
      snapshot.vector(logits);
      auto state = graph.export_state();
      snapshot.state(state);
      if (!record) {
        rejected([&] { graph.prefill_resident({1, -1, 2}); });
        equal(graph.export_state(), state);
        graph.set_trace([](size_t, const Vector &, const Vector *) {});
        rejected([&] { graph.prefill_resident({1, 2}); });
        graph.set_trace({});
        equal(graph.export_state(), state);
      }
      auto next = graph.replay_resident(11);
      snapshot.vector(next);
      snapshot.state(graph.export_state());
      graph.load_state(state);
      equal(graph.replay_resident(11), next);
      // Branch via another batched call after decode, then compare final state.
      Vector branch;
      if (record) for (int token : {3, 5}) branch = graph.replay_resident(token);
      else branch = graph.prefill_resident({3, 5});
      snapshot.vector(branch);
      snapshot.state(graph.export_state());
      std::cout << Json({{"prompt_tokens", length}, {"status", "passed"}}).dump() << std::endl;
    }
    if (!record && snapshot.file.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("Snapshot trailing data");
    auto stats = graph.stats();
    std::cout << Json({{"status", "passed"}, {"mode", record ? "record" : "verify"},
        {"vectors", snapshot.vectors}, {"weights", output_int8 ? "int8_ffn_output" : int8 ? "int8_ffn" : "bf16"}, {"resident_bytes", stats.resident_bytes},
        {"root_bos", stats.root_bos}, {"prefill_pair_runs", stats.prefill_pair_runs},
        {"prefill_runs", stats.prefill_runs}, {"prefill_chunk_tokens", stats.prefill_chunk_tokens},
        {"prefill_chunk_runs", stats.prefill_chunk_runs}}).dump() << '\n';
  } catch (const std::exception &e) {
    std::cerr << "prefill model: " << e.what() << '\n'; return 1;
  }
}
