// SPDX-License-Identifier: Apache-2.0
// Teacher-forced comparison: both graphs consume the same token stream.
#include "rwkv/inference/graph.hpp"
#include "tokenizer.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <nlohmann/json.hpp>
using namespace rwkv::inference;
using Json = nlohmann::json;
static Json error(const Vector &actual, const Vector &reference) {
  if (actual.size() != reference.size() || actual.empty())
    throw std::runtime_error("Accuracy vector shape mismatch");
  double max_abs = 0, square = 0, aa = 0, bb = 0, dot = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    double a = actual[i], b = reference[i], delta = a - b;
    if (!std::isfinite(a) || !std::isfinite(b))
      throw std::runtime_error("Nonfinite accuracy output");
    max_abs = std::max(max_abs, std::abs(delta)); square += delta * delta;
    aa += a * a; bb += b * b; dot += a * b;
  }
  return {{"max_abs", max_abs}, {"rmse", std::sqrt(square / actual.size())},
          {"relative_l2", bb ? Json(std::sqrt(square / bb)) : Json(nullptr)},
          {"cosine", aa && bb ? Json(dot / std::sqrt(aa * bb)) : Json(nullptr)}};
}
static Vector flatten(const State &state) {
  Vector result;
  for (const auto &layer : state.layers)
    for (const auto *v : {&layer.attention_shift, &layer.ffn_shift, &layer.matrix})
      result.insert(result.end(), v->begin(), v->end());
  return result;
}
static double kl(const Vector &reference, const Vector &actual) {
  double bm = *std::max_element(reference.begin(), reference.end());
  double am = *std::max_element(actual.begin(), actual.end());
  double bs = 0, as = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    bs += std::exp(reference[i] - bm); as += std::exp(actual[i] - am);
  }
  double result = 0, bz = bm + std::log(bs), az = am + std::log(as);
  for (size_t i = 0; i < actual.size(); ++i)
    result += std::exp(reference[i] - bz) * ((reference[i] - bz) - (actual[i] - az));
  return std::max(result, 0.0);
}
static double nll(const Vector &logits, int target) {
  double maximum = *std::max_element(logits.begin(), logits.end()), sum = 0;
  for (float x : logits) sum += std::exp(x - maximum);
  return maximum + std::log(sum) - logits[target];
}
int main(int argc, char **argv) {
  try {
    bool text_file = false, checkpoint_nodes = false, output_int8 = false;
    while (argc > 1) {
      std::string flag(argv[argc - 1]);
      if (flag == "--text" && !text_file) text_file = true;
      else if (flag == "--checkpoint-nodes" && !checkpoint_nodes) checkpoint_nodes = true;
      else if (flag == "--int8-ffn-output" && !output_int8) output_int8 = true;
      else break;
      --argc;
    }
    if (argc < 4 || argc > 6)
      throw std::runtime_error("Usage: rwkv-accuracy MODEL BF16_ROOT INT8_ROOT [STEPS [TOKEN_ID_FILE|TEXT_FILE]] [--text] [--checkpoint-nodes] [--int8-ffn-output]");
    if (text_file && argc != 6)
      throw std::runtime_error("--text requires a text file");
    size_t steps = 128;
    if (argc >= 5) {
      std::string s(argv[4]);
      if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid step count");
      steps = std::stoull(s);
      if (!steps || steps > 1000000) throw std::runtime_error("Invalid step count");
    }
    std::vector<int> tokens{1, 2, 7, 9};
    if (argc == 6) {
      std::ifstream input(argv[5]);
      if (!input) throw std::runtime_error("Cannot read token file");
      tokens.clear();
      if (text_file) {
        std::string text((std::istreambuf_iterator<char>(input)), {});
        trie_tokenizer tokenizer;
        if (tokenizer.load("assets/rwkv_vocab_v20230424.txt") != RWKV_SUCCESS)
          throw std::runtime_error("Cannot load assets/rwkv_vocab_v20230424.txt");
        tokens = tokenizer.encode(text);
        if (tokenizer.decode(tokens) != text)
          throw std::runtime_error("Accuracy text did not round-trip");
      } else {
        int token;
        while (input >> token) tokens.push_back(token);
        if (!input.eof()) throw std::runtime_error("Invalid token ID file");
      }
      if (tokens.size() < steps)
        throw std::runtime_error("Token file must contain at least STEPS whitespace-separated IDs");
    }
    const size_t supplied_tokens = argc == 6 ? tokens.size() : 0;
    Weights weights(argv[1]);
    for (int token : tokens)
      if (token < 0 || size_t(token) >= weights.vocabulary())
        throw std::runtime_error("Token out of range");
    auto cpu = cpu_backend(); Model model(weights, *cpu);
    auto initial = model.initial_state();
    auto baseline = std::make_unique<DecodeGraph>(weights, argv[2]);
    baseline->load_state(initial);
    struct NodeSnapshot { std::vector<Vector> nodes, matrices; };
    std::map<size_t, NodeSnapshot> snapshots;
    std::vector<Vector> reference_logits;
    std::map<size_t, State> reference_states;
    auto checkpoint = [&](size_t step) {
      return step == 0 || step == 7 || step == 31 || step + 1 == steps;
    };
    for (size_t step = 0; step < steps; ++step) {
      if (step == 0 || (checkpoint_nodes && checkpoint(step))) {
        auto &snapshot = snapshots[step];
        baseline->set_trace([&snapshot](size_t node, const Vector &v, const Vector *state) {
          if (node != snapshot.nodes.size()) throw std::runtime_error("Unexpected baseline node order");
          snapshot.nodes.push_back(v); snapshot.matrices.push_back(state ? *state : Vector{});
        });
      }
      auto expected = baseline->replay_resident(tokens[step]);
      baseline->set_trace({});
      if (checkpoint(step)) reference_states[step] = baseline->export_state();
      if (step + 1 >= tokens.size())
        tokens.push_back(std::max_element(expected.begin(), expected.end()) - expected.begin());
      reference_logits.push_back(std::move(expected));
    }
    auto baseline_stats = baseline->stats();
    // Two complete resident graphs exceed the driver's context limit on this
    // device. Retain host references and release all baseline contexts first.
    baseline.reset();
    DecodeGraph candidate(weights, argv[3], output_int8 ? WeightMode::Int8FFNOutput : WeightMode::Int8FFN);
    candidate.load_state(initial);
    size_t first_token_nodes = 0, checkpoint_nodes_compared = 0, traced_checkpoints = 0;
    Vector first_candidate;
    size_t top1_matches = 0;
    double sum_kl = 0, worst_kl = 0, worst_abs = 0;
    size_t nll_tokens = 0;
    double baseline_nll = 0, candidate_nll = 0;
    for (size_t step = 0; step < steps; ++step) {
      int token = tokens[step];
      const auto &expected = reference_logits[step];
      size_t compared_nodes = 0;
      auto snapshot = snapshots.find(step);
      if (snapshot != snapshots.end()) {
        candidate.set_trace([&](size_t node, const Vector &v, const Vector *state) {
          const auto &reference = snapshot->second;
          if (node != compared_nodes++ || node >= reference.nodes.size())
            throw std::runtime_error("Unexpected candidate node order");
          Json entry{{"type", step == 0 ? "first_token_node" : "checkpoint_node"},
            {"step", step + 1}, {"node", node}, {"elements", v.size()},
            {"error", error(v, reference.nodes[node])}};
          if (state) entry["state_error"] = error(*state, reference.matrices[node]);
          std::cout << entry.dump() << '\n';
        });
      }
      auto actual = candidate.replay_resident(token);
      candidate.set_trace({});
      if (snapshot != snapshots.end()) {
        if (compared_nodes != baseline_stats.nodes)
          throw std::runtime_error("Incomplete checkpoint trace");
        checkpoint_nodes_compared += compared_nodes;
        ++traced_checkpoints;
        snapshots.erase(snapshot);
      }
      if (step == 0) {
        first_candidate = actual;
        first_token_nodes = compared_nodes;
      }
      int btop = std::max_element(expected.begin(), expected.end()) - expected.begin();
      int atop = std::max_element(actual.begin(), actual.end()) - actual.begin();
      top1_matches += btop == atop;
      double divergence = kl(expected, actual);
      sum_kl += divergence; worst_kl = std::max(worst_kl, divergence);
      auto metrics = error(actual, expected);
      worst_abs = std::max(worst_abs, metrics["max_abs"].get<double>());
      Json entry{{"type", "token"}, {"step", step + 1}, {"input_token", token},
        {"logits", metrics}, {"kl_reference_to_candidate", divergence},
        {"baseline_top1", btop}, {"candidate_top1", atop}};
      if (step + 1 < supplied_tokens) {
        double bn = nll(expected, tokens[step + 1]), an = nll(actual, tokens[step + 1]);
        entry["baseline_next_token_nll"] = bn;
        entry["candidate_next_token_nll"] = an;
        baseline_nll += bn; candidate_nll += an; ++nll_tokens;
      }
      if (checkpoint(step)) {
        auto state = candidate.export_state();
        const auto &reference = reference_states.at(step);
        entry["state_error"] = error(flatten(state), flatten(reference));
        auto &layers = entry["layer_state_errors"] = Json::array();
        for (size_t layer = 0; layer < state.layers.size(); ++layer) {
          const auto &a = state.layers[layer], &b = reference.layers.at(layer);
          layers.push_back({{"layer", layer},
            {"attention_shift", error(a.attention_shift, b.attention_shift)},
            {"ffn_shift", error(a.ffn_shift, b.ffn_shift)},
            {"matrix", error(a.matrix, b.matrix)}});
        }
      }
      std::cout << entry.dump() << std::endl;
    }
    auto branch = candidate.export_state();
    auto next = candidate.replay_resident(7);
    auto advanced = flatten(candidate.export_state());
    candidate.load_state(branch);
    if (candidate.replay_resident(7) != next || flatten(candidate.export_state()) != advanced)
      throw std::runtime_error("INT8 branch/resume changed state or logits");
    candidate.load_state(initial);
    if (candidate.replay_resident(tokens[0]) != first_candidate)
      throw std::runtime_error("INT8 reset changed first-token logits");
    std::cout << Json({{"type", "summary"}, {"status", "measured"}, {"steps", steps},
      {"precision", output_int8 ? "int8_ffn_output_bf16_others_fp32_state" : "int8_ffn_bf16_others_fp32_state"},
      {"baseline_precision", "bf16_fp32_state"},
      {"top1_agreement", double(top1_matches) / steps}, {"mean_kl", sum_kl / steps},
      {"max_kl", worst_kl}, {"max_abs_logits", worst_abs},
      {"reset_and_branch", "passed"}, {"first_token_nodes_compared", first_token_nodes},
      {"traced_checkpoints", traced_checkpoints}, {"checkpoint_nodes_compared", checkpoint_nodes_compared},
      {"baseline_resident_bytes", baseline_stats.resident_bytes},
      {"candidate_resident_bytes", candidate.stats().resident_bytes},
      {"nll_tokens", nll_tokens},
      {"baseline_perplexity", nll_tokens ? Json(std::exp(baseline_nll / nll_tokens)) : Json(nullptr)},
      {"candidate_perplexity", nll_tokens ? Json(std::exp(candidate_nll / nll_tokens)) : Json(nullptr)},
      {"mean_nll_increase", nll_tokens ? Json((candidate_nll - baseline_nll) / nll_tokens) : Json(nullptr)},
      {"perplexity_ratio", nll_tokens ? Json(std::exp((candidate_nll - baseline_nll) / nll_tokens)) : Json(nullptr)},
      {"workload", argc == 6 ? (text_file ? "supplied UTF-8 text" : "supplied teacher-forced token IDs") : "IDs 1,2,7,9 followed by baseline greedy tokens"},
      {"quality_gate", "No language-quality acceptance threshold is implied by this measurement."}}).dump() << '\n';
  } catch (const std::exception &e) {
    std::cerr << "rwkv-accuracy: " << e.what() << '\n';
    return 1;
  }
}
