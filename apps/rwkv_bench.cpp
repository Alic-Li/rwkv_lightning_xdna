// SPDX-License-Identifier: Apache-2.0
// Fixed token workload: sampling and state export are outside measured regions.
#include "rwkv/inference/graph.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <nlohmann/json.hpp>
using namespace rwkv::inference;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
static double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
static size_t count(const char *arg) {
  std::string text(arg);
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("Counts must be positive integers");
  auto n = std::stoull(text);
  if (!n || n > 1000000)
    throw std::invalid_argument("Counts must be in [1, 1000000]");
  return n;
}
static Json distribution(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const double total = std::accumulate(values.begin(), values.end(), 0.0);
  auto percentile = [&](double p) {
    return values[static_cast<size_t>(std::ceil(p * values.size())) - 1];
  };
  return {{"samples", values.size()}, {"mean_ms", total / values.size()},
          {"p50_ms", percentile(.5)}, {"p95_ms", percentile(.95)},
          {"min_ms", values.front()}, {"max_ms", values.back()},
          {"tokens_per_second", 1000 * values.size() / total}};
}
int main(int argc, char **argv) {
  try {
    bool int8 = false, batch2 = false, chunk4 = false;
    while (argc > 1 && std::string(argv[argc - 1]).rfind("--", 0) == 0) {
      const std::string flag(argv[--argc]);
      if (flag == "--prefill-chunk4") chunk4 = batch2 = true;
      else if (flag == "--prefill-batch2") batch2 = true;
      else if (flag == "--int8-ffn") int8 = true;
      else throw std::invalid_argument("Unknown flag: " + flag);
    }
    if (argc != 3 && argc != 6)
      throw std::invalid_argument("Usage: rwkv-bench MODEL KERNELS [PREFILL_TOKENS DECODE_TOKENS TRIALS] [--int8-ffn] [--prefill-batch2|--prefill-chunk4]");
    if (std::getenv("RWKV_XDNA_PROFILE"))
      throw std::invalid_argument("Unset RWKV_XDNA_PROFILE for uninstrumented benchmarks; use rwkv-cli for stage profiles");
    const size_t prefill = argc == 6 ? count(argv[3]) : 32;
    const size_t decode = argc == 6 ? count(argv[4]) : 64;
    const size_t trials = argc == 6 ? count(argv[5]) : 3;
    auto start = Clock::now();
    Weights weights(argv[1]);
    double load_s = seconds(start);
    auto backend = cpu_backend();
    Model model(weights, *backend);
    auto initial = model.initial_state();
    start = Clock::now();
    DecodeGraph graph(weights, argv[2], int8 ? WeightMode::Int8FFN : WeightMode::BFloat16,
                     chunk4 ? PrefillMode::Chunked4 : batch2 ? PrefillMode::Batched2 : PrefillMode::Sequential);
    double build_s = seconds(start);
    graph.load_state(initial);
    for (int i = 0; i < 4; ++i)
      graph.replay_resident(i % weights.vocabulary());
    std::vector<double> latencies;
    latencies.reserve(trials * decode);
    Json trial_results = Json::array();
    Vector logits;
    std::vector<int> prompt;
    for (size_t t = 0; t < prefill; ++t)
      prompt.push_back((t * 17 + 1) % weights.vocabulary());
    for (size_t trial = 0; trial < trials; ++trial) {
      start = Clock::now();
      graph.load_state(initial);
      const double reset_s = seconds(start);
      const auto runs_before = graph.stats().prefill_runs;
      start = Clock::now();
      if (batch2) logits = graph.prefill_resident(prompt);
      else for (int token : prompt) logits = graph.replay_resident(token);
      double prefill_s = seconds(start);
      const auto prefill_runs = batch2 ? graph.stats().prefill_runs - runs_before
                                      : prefill * graph.stats().persistent_runs;
      std::vector<double> sample;
      sample.reserve(decode);
      for (size_t t = 0; t < decode; ++t) {
        start = Clock::now();
        logits = graph.replay_resident(((prefill + t) * 17 + 1) % weights.vocabulary());
        sample.push_back(seconds(start) * 1000);
      }
      for (float x : logits)
        if (!std::isfinite(x))
          throw std::runtime_error("Nonfinite final logits");
      latencies.insert(latencies.end(), sample.begin(), sample.end());
      trial_results.push_back({{"state_reset_seconds", reset_s},
          {"prefill_seconds", prefill_s},
          {"prefill_runs", prefill_runs},
          {"prefill_runs_per_token", double(prefill_runs) / prefill},
          {"prefill_host_upload_bytes", prefill * graph.stats().persistent_upload_bytes},
          {"prefill_host_download_bytes", (batch2 ? (chunk4 ? std::max(size_t(1), prefill % 4) : size_t(1)) : prefill) * graph.stats().persistent_download_bytes},
          {"prefill_tokens_per_second", prefill / prefill_s},
          {"decode_samples_ms", sample},
          {"decode", distribution(sample)}});
    }
    auto stats = graph.stats();
    Json result = {{"schema_version", 1}, {"model", argv[1]},
      {"kernels", argv[2]}, {"precision", int8 ? "w8a8_ffn_int32_dot_fp32_partial_others_bf16_fp32_state" : "bf16_weights_inputs_fp32_state"},
      {"workload", "fixed synthetic token IDs"},
      {"prefill_mode", chunk4 ? "chunk4_ffn_final_logits" : batch2 ? "batch2_ffn_final_logits" : "sequential_all_logits"},
      {"prefill_chunk_tokens", stats.prefill_chunk_tokens}, {"prefill_chunk_runs", stats.prefill_chunk_runs},
      {"prefill_tokens", prefill}, {"decode_tokens", decode},
      {"warmup_tokens", 4}, {"weight_load_seconds", load_s},
      {"graph_build_seconds", build_s}, {"trials", trial_results},
      {"decode", distribution(latencies)}, {"runs_per_token", stats.persistent_runs},
      {"graph_nodes", stats.nodes}, {"graph_buffers", stats.buffers},
      {"replays", stats.replays},
      {"resident_bytes", stats.resident_bytes}, {"root_bos", stats.root_bos},
      {"host_upload_bytes_per_token", stats.persistent_upload_bytes},
      {"host_download_bytes_per_token", stats.persistent_download_bytes},
      {"npu_compute_utilization", nullptr}, {"device_dma_bandwidth", nullptr},
      {"counter_note", "Host BO traffic excludes internal DDR-to-AIE DMA; utilization and DMA require device counters, not host wait timing."}};
    std::cout << result.dump(2) << '\n';
  } catch (const std::exception &e) {
    std::cerr << "rwkv-bench: " << e.what() << '\n';
    return 1;
  }
}
