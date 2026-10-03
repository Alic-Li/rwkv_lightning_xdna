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
    const bool output_int8 = argc > 1 && std::string(argv[argc - 1]) == "--int8-ffn-output";
    const bool int8 = output_int8 || (argc > 1 && std::string(argv[argc - 1]) == "--int8-ffn");
    if (int8) --argc;
    if (argc != 3 && argc != 6)
      throw std::invalid_argument("Usage: rwkv-bench MODEL KERNELS [PREFILL_TOKENS DECODE_TOKENS TRIALS] [--int8-ffn|--int8-ffn-output]");
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
    DecodeGraph graph(weights, argv[2], output_int8 ? WeightMode::Int8FFNOutput :
                     int8 ? WeightMode::Int8FFN : WeightMode::BFloat16);
    double build_s = seconds(start);
    graph.load_state(initial);
    for (int i = 0; i < 4; ++i)
      graph.replay_resident(i % weights.vocabulary());
    std::vector<double> latencies;
    Json trial_results = Json::array();
    Vector logits;
    for (size_t trial = 0; trial < trials; ++trial) {
      graph.load_state(initial);
      start = Clock::now();
      for (size_t t = 0; t < prefill; ++t)
        logits = graph.replay_resident((t * 17 + 1) % weights.vocabulary());
      double prefill_s = seconds(start);
      std::vector<double> sample;
      for (size_t t = 0; t < decode; ++t) {
        start = Clock::now();
        logits = graph.replay_resident(((prefill + t) * 17 + 1) % weights.vocabulary());
        sample.push_back(seconds(start) * 1000);
      }
      for (float x : logits)
        if (!std::isfinite(x))
          throw std::runtime_error("Nonfinite final logits");
      latencies.insert(latencies.end(), sample.begin(), sample.end());
      trial_results.push_back({{"prefill_seconds", prefill_s},
          {"prefill_tokens_per_second", prefill / prefill_s},
          {"decode", distribution(sample)}});
    }
    auto stats = graph.stats();
    Json result = {{"schema_version", 1}, {"model", argv[1]},
      {"kernels", argv[2]}, {"precision", output_int8 ? "int8_ffn_output_bf16_others_fp32_state" :
        int8 ? "int8_ffn_bf16_others_fp32_state" : "bf16_weights_inputs_fp32_state"},
      {"workload", "fixed synthetic token IDs; sequential resident prefill"},
      {"prefill_tokens", prefill}, {"decode_tokens", decode},
      {"warmup_tokens", 4}, {"weight_load_seconds", load_s},
      {"graph_build_seconds", build_s}, {"trials", trial_results},
      {"decode", distribution(latencies)}, {"runs_per_token", stats.persistent_runs},
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
