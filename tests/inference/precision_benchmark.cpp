// SPDX-License-Identifier: Apache-2.0
#include "rwkv/inference/graph.hpp"
#include "tokenizer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
using namespace rwkv::inference;
using Clock = std::chrono::steady_clock;
static nlohmann::json errors(const Vector &x, const Vector &ref) {
  if (x.size() != ref.size())
    throw std::runtime_error("shape mismatch");
  double max_abs = 0, mean_abs = 0, sq = 0, dot = 0, xx = 0, rr = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    if (!std::isfinite(x[i]) || !std::isfinite(ref[i]))
      throw std::runtime_error("nonfinite output");
    double e = double(x[i]) - ref[i];
    max_abs = std::max(max_abs, std::abs(e));
    mean_abs += std::abs(e);
    sq += e * e;
    dot += double(x[i]) * ref[i];
    xx += double(x[i]) * x[i];
    rr += double(ref[i]) * ref[i];
  }
  return {{"max_abs", max_abs},
          {"mean_abs", mean_abs / x.size()},
          {"relative_l2", std::sqrt(sq / std::max(rr, 1e-30))},
          {"cosine", dot / std::sqrt(std::max(xx * rr, 1e-30))}};
}
static Vector flatten(const State &s) {
  Vector v;
  for (const auto &l : s.layers)
    for (const auto *a : {&l.matrix, &l.attention_shift, &l.ffn_shift})
      v.insert(v.end(), a->begin(), a->end());
  return v;
}
int main(int argc, char **argv) {
  try {
    if (argc != 4 && argc != 5)
      throw std::runtime_error(
          "Usage: rwkv-precision-benchmark MODEL KERNELS STEPS [PROMPT]");
    int steps = std::stoi(argv[3]);
    if (steps < 1 || steps > 128)
      throw std::runtime_error("steps must be 1..128");
    Weights w(argv[1]);
    auto backend = full_npu_backend(argv[2]);
    Model model(w, *backend);
    std::vector<int> tokens;
    std::vector<int> prompt_tokens{1, 2, 7, 9};
    if (argc == 5) {
      trie_tokenizer tokenizer;
      if (tokenizer.load("assets/rwkv_vocab_v20230424.txt") != RWKV_SUCCESS)
        throw std::runtime_error("cannot load vocabulary");
      prompt_tokens = tokenizer.encode(argv[4]);
      if (prompt_tokens.empty())
        throw std::runtime_error("empty prompt");
    }
    std::cout << nlohmann::json({{"prompt", argc == 5 ? argv[4] : ""},
                                 {"prompt_tokens", prompt_tokens},
                                 {"steps", steps}})
                     .dump()
              << std::endl;
    std::vector<Vector> refs;
    std::map<int, Vector> states;
    auto checkpoint = [](int t) {
      return t == 1 || t == 8 || t == 32 || t == 128;
    };
    for (int arm = 0; arm < 2; ++arm) {
      if (arm || std::getenv("RWKV_XDNA_REFERENCE_BF16"))
        setenv("RWKV_XDNA_BF16", "1", 1);
      else
        unsetenv("RWKV_XDNA_BF16");
      auto begin = Clock::now();
      const char *reference_root =
          std::getenv("RWKV_XDNA_REFERENCE_KERNEL_DIR");
      const std::string kernel_root =
          !arm && reference_root ? reference_root : argv[2];
      DecodeGraph graph(w, *backend, kernel_root);
      auto stats = graph.stats();
      std::cout
          << nlohmann::json(
                 {{"arm", arm},
                  {"build_seconds",
                   std::chrono::duration<double>(Clock::now() - begin).count()},
                  {"kernels", kernel_root},
                  {"bf16_projections",
                   std::getenv("RWKV_XDNA_BF16") != nullptr},
                  {"runs", stats.persistent_runs},
                  {"resident_bytes", stats.resident_bytes}})
                 .dump()
          << std::endl;
      graph.load_state(model.initial_state());
      int next = prompt_tokens[0];
      double total = 0;
      int greedy_matches = 0;
      for (int t = 0; t < steps; ++t) {
        if (!arm)
          tokens.push_back(next);
        begin = Clock::now();
        auto y = graph.replay_resident(tokens[t]);
        double seconds =
            std::chrono::duration<double>(Clock::now() - begin).count();
        if (t)
          total += seconds;
        auto greedy = [](const Vector &v) {
          return int(std::max_element(v.begin(), v.end()) - v.begin());
        };
        nlohmann::json row = {{"arm", arm},
                              {"step", t + 1},
                              {"input_token", tokens[t]},
                              {"seconds", seconds},
                              {"greedy", greedy(y)}};
        if (!arm) {
          refs.push_back(y);
          next = size_t(t + 1) < prompt_tokens.size() ? prompt_tokens[t + 1]
                                                      : greedy(y);
          if (checkpoint(t + 1) || t + 1 == steps)
            states[t + 1] = flatten(graph.export_state());
        } else {
          row["logits_error"] = errors(y, refs[t]);
          bool same = greedy(y) == greedy(refs[t]);
          greedy_matches += same;
          row["greedy_matches"] = same;
          if (checkpoint(t + 1) || t + 1 == steps)
            row["state_error"] =
                errors(flatten(graph.export_state()), states.at(t + 1));
        }
        std::cout << row.dump() << std::endl;
      }
      std::cout << nlohmann::json(
                       {{"arm", arm},
                        {"measured_tokens", steps - 1},
                        {"total_seconds", total},
                        {"tokens_per_second",
                         steps > 1 ? (steps - 1) / total : 0},
                        {"greedy_matches", greedy_matches},
                        {"scope", "Identical-token cross-arm drift "
                                  "measurement; not a tolerance pass"}})
                       .dump()
                << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
