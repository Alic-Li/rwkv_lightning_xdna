// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>

using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

static size_t packed_index(size_t row, size_t col, size_t k) {
  return ((row / 16) * (k / 256) + col / 256) * 4096 +
         (row % 16) * 256 + col % 256;
}

static Json check_shape(const std::filesystem::path &root, size_t k,
                        size_t rows, size_t iterations) {
  auto directory = [&](int batch) {
    auto path = root / ("bf16-prefill-projection-" + std::to_string(k) + "-" +
                        std::to_string(rows) + "-b" + std::to_string(batch));
    Json config;
    std::ifstream input(path / "config.json");
    input >> config;
    if (config.at("k") != k || config.at("rows") != rows ||
        config.at("batch") != batch || config.at("input_dtype") != "bfloat16" ||
        config.at("output_dtype") != "float32" ||
        config.at("layout") != "token_major")
      throw std::runtime_error("Invalid prefill projection artifact contract");
    return path;
  };
  const auto a = directory(1), b = directory(2);
  Session single(a / "design.xclbin", a / "instructions.bin");
  Session batched(b / "design.xclbin", b / "instructions.bin");
  Guarded x(batched, 2 * k * 2), weights(batched, rows * k * 2),
      out(batched, 2 * rows * 4), reference(single, 2 * rows * 4);
  auto batch_run = batched.prepare({x.data, weights.data, out.data});
  std::vector<DeviceRun> singles;
  for (size_t token = 0; token < 2; ++token)
    singles.push_back(single.prepare({x.data.slice(token * k * 2, k * 2),
        weights.data, reference.data.slice(token * rows * 4, rows * 4)}));

  std::mt19937 rng(730);
  std::uniform_real_distribution<float> random(-1, 1);
  std::vector<uint16_t> w(rows * k), input(2 * k);
  for (auto &v : w) v = bf16(random(rng) / 64);
  for (size_t col = 0; col < k; ++col) w[packed_index(0, col, k)] = 0;
  weights.data.upload(w.data(), weights.bytes);
  std::vector<float> actual(2 * rows), baseline(2 * rows);
  double worst = 0;
  for (int pass = 0; pass < 3; ++pass) {
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = bf16(pass == 1 && i < k ? 0 : random(rng) * (pass + 1));
    x.data.upload(input.data(), x.bytes);
    // Poison all outputs: every token and output row must be overwritten.
    std::fill(actual.begin(), actual.end(), std::nanf(""));
    out.data.upload(actual.data(), out.bytes);
    reference.data.upload(actual.data(), reference.bytes);
    for (auto &run : singles) run.execute();
    batch_run.execute();
    out.data.download(actual.data(), out.bytes);
    reference.data.download(baseline.data(), reference.bytes);
    if (std::memcmp(actual.data(), baseline.data(), out.bytes) != 0)
      throw std::runtime_error("Batched projection differs bitwise from batch 1");
    for (size_t token = 0; token < 2; ++token)
      for (size_t row = 0; row < rows; ++row) {
        double expected = 0;
        for (size_t col = 0; col < k; ++col)
          expected += double(expand(input[token * k + col])) *
                      expand(w[packed_index(row, col, k)]);
        const float value = actual[token * rows + row];
        const double error = std::abs(value - expected);
        worst = std::max(worst, error);
        if (!std::isfinite(value) || error > 2e-5 + 2e-5 * std::abs(expected))
          throw std::runtime_error("Batched projection FP64 oracle mismatch");
      }
    std::vector<uint16_t> unchanged_w(w.size()), unchanged_x(input.size());
    weights.data.download(unchanged_w.data(), weights.bytes);
    x.data.download(unchanged_x.data(), x.bytes);
    if (w != unchanged_w || input != unchanged_x)
      throw std::runtime_error("Projection changed inputs or weights");
    x.guard(); weights.guard(); out.guard(); reference.guard();
  }
  Json timing = Json::array();
  for (bool batch : {false, true, true, false}) {
    auto execute = [&] {
      if (batch) batch_run.execute();
      else for (auto &run : singles) run.execute();
    };
    for (int i = 0; i < 8; ++i) execute();
    std::vector<double> samples;
    samples.reserve(iterations);
    for (size_t i = 0; i < iterations; ++i) {
      auto start = Clock::now();
      execute();
      samples.push_back(std::chrono::duration<double, std::micro>(
          Clock::now() - start).count());
    }
    std::sort(samples.begin(), samples.end());
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / iterations;
    timing.push_back({{"batch", batch ? 2 : 1}, {"tokens_per_sample", 2},
        {"samples", iterations}, {"mean_us", mean},
        {"p50_us", samples[(iterations - 1) / 2]},
        {"p95_us", samples[size_t(std::ceil(.95 * iterations)) - 1]},
        {"microseconds_per_token", mean / 2},
        {"submissions_per_token", batch ? .5 : 1.0},
        {"weight_bytes_per_token", weights.bytes / (batch ? 2 : 1)}});
  }
  x.guard(); weights.guard(); out.guard(); reference.guard();
  return {{"k", k}, {"rows", rows}, {"status", "passed"},
      {"oracle_max_abs", worst}, {"input_passes", 3},
      {"batch1_batch2_bitwise", "passed"}, {"immutable_inputs_guards", "passed"},
      {"timing_abba", timing},
      {"scope", "Isolated BF16 projection, two tokens; excludes host transfers and graph setup. Weight bytes are static DMA payload, not measured bandwidth. Not whole-model prefill."}};
}

int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 3)
      throw std::runtime_error("Usage: rwkv-prefill-projection-test KERNEL_ROOT [ITERATIONS]");
    size_t iterations = 200;
    if (argc == 3) {
      std::string count(argv[2]);
      if (count.empty() || count.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid iterations");
      iterations = std::stoull(count);
      if (!iterations || iterations > 1000000)
        throw std::runtime_error("Iterations outside [1, 1000000]");
    }
    for (auto shape : {std::pair<size_t, size_t>{2048, 8192}, {8192, 2048}})
      std::cout << check_shape(argv[1], shape.first, shape.second, iterations).dump()
                << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "prefill projection: " << e.what() << '\n';
    return 1;
  }
}
