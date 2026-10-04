// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rwkv/inference/graph.hpp"
#include "rwkv/xdna/session.hpp"
#include <chrono>
#include "sampler.h"
#include "tokenizer.h"
#include <condition_variable>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

// Independent graphs/BOs/state per request. Threads submit NPU work concurrently;
// no inference mutex or loop that completes one request before starting another.
inline int parallel_inference(const rwkv::inference::Weights &weights,
    const trie_tokenizer &tokenizer, const std::vector<int> &tokens,
    const std::map<std::string, std::string> &opts, size_t concurrency,
    bool diagnostic) {
  using namespace rwkv::inference;
  using Clock = std::chrono::steady_clock;
  struct Request {
    std::unique_ptr<RecurrentBackend> backend;
    std::unique_ptr<DecodeGraph> graph;
    State state;
    std::string output;
    size_t generated = 0, forwards = 0;
    double prefill = 0, generation = 0, ttft = 0;
    std::exception_ptr error;
  };
  std::vector<Request> requests(concurrency);
  auto setup = Clock::now();
  for (auto &r : requests) {
    r.backend = cpu_backend();
    r.state = Model(weights, *r.backend).initial_state();
    if (opts.at("--backend") == "npu") {
      r.graph = std::make_unique<DecodeGraph>(weights, opts.at("--kernel-dir"),
          opts.at("--weights") == "bf16" ? WeightMode::BFloat16 : WeightMode::Int8FFN,
          opts.at("--prefill") == "chunk4" ? PrefillMode::Chunked4 :
          opts.at("--prefill") == "batch2" ? PrefillMode::Batched2 : PrefillMode::Sequential);
      r.graph->load_state(r.state);
    } else {
      r.graph = std::make_unique<DecodeGraph>(weights, *r.backend);
    }
  }
  std::cerr << "Parallel requests: " << concurrency << "; independent graphs/state; setup "
            << std::chrono::duration<double>(Clock::now() - setup).count() << " s\n";
  std::mutex gate_mutex;
  std::condition_variable gate;
  size_t ready = 0, prefilled = 0;
  bool start = false, decode_start = false, cancel = false;
  Clock::time_point begin, generation_begin;
  auto worker = [&](size_t id) {
    auto &r = requests[id];
#ifdef _OPENMP
    omp_set_num_threads(std::stoi(opts.at("--threads")));
#endif
    try {
      std::ofstream dump;
      if (!opts.at("--dump-logits").empty()) {
        const auto path = opts.at("--dump-logits") + ".request-" + std::to_string(id);
        dump.open(path, std::ios::binary);
        if (!dump) throw std::runtime_error("Cannot open logits output: " + path);
      }
      {
        std::unique_lock<std::mutex> lock(gate_mutex);
        ++ready; gate.notify_all();
        gate.wait(lock, [&] { return start || cancel; });
        if (cancel) return;
      }
      auto forward = [&](int token) {
        return opts.at("--backend") == "npu" ? r.graph->replay_resident(token)
                                              : r.graph->replay(token, r.state);
      };
      Vector logits;
      if (opts.at("--prefill") == "batch2" || opts.at("--prefill") == "chunk4")
        logits = r.graph->prefill_resident(tokens);
      else {
        for (int token : tokens) {
          logits = forward(token);
          if (dump.is_open()) {
            dump.write(reinterpret_cast<const char *>(logits.data()), logits.size() * sizeof(float));
            if (!dump) throw std::runtime_error("Logits write failed");
          }
        }
      }
      r.prefill = std::chrono::duration<double>(Clock::now() - begin).count();
      {
        std::unique_lock<std::mutex> lock(gate_mutex);
        ++prefilled; gate.notify_all();
        gate.wait(lock, [&] { return decode_start || cancel; });
        if (cancel) return;
      }
      if (diagnostic) return;
      rwkvmobile::NucleusSampler sampler;
      // Same seed makes duplicated greedy/stochastic requests reproducible.
      sampler.set_seed(std::stoi(opts.at("--seed")));
      std::map<int, float> occurrences;
      const long limit = std::stol(opts.at("--max-tokens"));
      for (long i = 0; i < limit; ++i) {
        rwkvmobile::Tensor1D view{logits.data(), rwkvmobile::TensorDType::F32, logits.size()};
        sampler.apply_penalties(view, logits.size(), occurrences, {},
            std::stof(opts.at("--presence-penalty")), std::stof(opts.at("--frequency-penalty")),
            std::stof(opts.at("--penalty-decay")));
        const int token = sampler.sample(view, logits.size(), std::stof(opts.at("--temperature")),
            std::stoi(opts.at("--top-k")), std::stof(opts.at("--top-p")));
        if (token == tokenizer.eos_token_id) break;
        auto piece = tokenizer.decode(token);
        if (piece.empty()) throw std::runtime_error("Generated token missing from vocabulary");
        if (!r.generated) r.ttft = std::chrono::duration<double>(Clock::now() - begin).count();
        r.output += piece; ++r.generated; occurrences[token] += 1;
        if (i + 1 < limit) { logits = forward(token); ++r.forwards; }
      }
      r.generation = std::chrono::duration<double>(Clock::now() - generation_begin).count();
    } catch (...) {
      r.error = std::current_exception();
      std::lock_guard<std::mutex> lock(gate_mutex);
      cancel = true; gate.notify_all();
    }
  };
  std::vector<std::thread> workers;
  try {
    for (size_t id = 0; id < concurrency; ++id) workers.emplace_back(worker, id);
  } catch (...) {
    { std::lock_guard<std::mutex> lock(gate_mutex); cancel = true; gate.notify_all(); }
    for (auto &w : workers) w.join();
    throw;
  }
  {
    std::unique_lock<std::mutex> lock(gate_mutex);
    gate.wait(lock, [&] { return ready == concurrency || cancel; });
    begin = Clock::now(); start = true; gate.notify_all();
    gate.wait(lock, [&] { return prefilled == concurrency || cancel; });
    generation_begin = Clock::now(); decode_start = true; gate.notify_all();
  }
  for (auto &w : workers) w.join();
  const auto end = Clock::now();
  for (auto &r : requests) if (r.error) std::rethrow_exception(r.error);
  if (opts.at("--backend") == "npu")
    std::cerr << "NPU submitted commands awaiting completion: peak="
              << rwkv::xdna::dispatch_concurrency_peak() << '\n';
  size_t total = 0, forwards = 0;
  for (size_t id = 0; id < concurrency; ++id) {
    const auto &r = requests[id]; total += r.generated; forwards += r.forwards;
    if (!diagnostic) std::cout << "[request " << id << "]\n" << r.output << '\n';
    std::cerr << "Request " << id << ": prefill=" << r.prefill << " s, TTFT=" << r.ttft
              << " s, generated=" << r.generated << ", generation=" << r.generation << " s\n";
  }
  const double prefill_wall = std::chrono::duration<double>(generation_begin - begin).count();
  const double decode_wall = std::chrono::duration<double>(end - generation_begin).count();
  const double wall = std::chrono::duration<double>(end - begin).count();
  std::cerr << "Parallel aggregate (setup excluded): prefill " << tokens.size() * concurrency
            << " tokens / " << prefill_wall << " s = " << tokens.size() * concurrency / prefill_wall
            << " token/s; generated " << total << " tokens / " << decode_wall << " s = "
            << total / decode_wall << " token/s; decode forwards " << forwards / decode_wall
            << " token/s; end-to-end " << total / wall << " generated token/s; wall=" << wall << " s\n";
  return 0;
}
