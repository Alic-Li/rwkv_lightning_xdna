// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
using V = std::vector<float>;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

static size_t packed(size_t row, size_t col, size_t k) {
  return ((row / 16) * (k / 256) + col / 256) * 4096 +
         row % 16 * 256 + col % 256;
}
static void same(const float *a, const float *b, size_t n, const char *name) {
  if (std::memcmp(a, b, n * 4))
    throw std::runtime_error(std::string("Bitwise mismatch: ") + name);
}
int main(int argc, char **argv) {
  try {
    if (argc < 3 || argc > 4)
      throw std::runtime_error("Usage: rwkv-prefill-ffn-test FUSED_KERNELS DECODE_KERNELS [ITERATIONS]");
    size_t iterations = 100;
    if (argc == 4) {
      std::string count(argv[3]);
      if (count.empty() || count.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid iterations");
      iterations = std::stoull(count);
      if (!iterations || iterations > 1000000)
        throw std::runtime_error("Iterations outside [1, 1000000]");
    }
    auto session = [](std::filesystem::path root, const char *name) {
      return Session(root / name / "design.xclbin", root / name / "instructions.bin");
    };
    auto fused = session(argv[1], "bf16-prefill-ffn-b2");
    auto decode = session(argv[2], "bf16-channel-mix");
    Guarded x(fused, 4096 * 4), parameters(fused, 6144 * 4),
        weights(fused, 33554432 * 2), fp32(fused, 26624 * 4), half(fused, 20480 * 2),
        diag(decode, 22528 * 4), reference(decode, 4096 * 4);
    auto shift = fp32.data.slice(0, 2048 * 4);
    auto raw = fp32.data.slice(2048 * 4, 16384 * 4);
    auto projected = fp32.data.slice(18432 * 4, 4096 * 4);
    auto output = fp32.data.slice(22528 * 4, 4096 * 4);
    auto mixed = half.data.slice(0, 4096 * 2);
    auto active = half.data.slice(4096 * 2, 16384 * 2);
    auto fused_run = fused.prepare({x.data, parameters.data, weights.data, fp32.data, half.data});
    std::vector<DeviceRun> singles;
    for (size_t t = 0; t < 2; ++t)
      singles.push_back(decode.prepare({x.data.slice(t * 2048 * 4, 2048 * 4),
          parameters.data, weights.data, diag.data, reference.data}));
    auto batch = [&] { fused_run.execute(); };
    std::mt19937 rng(730);
    std::uniform_real_distribution<float> random(-1, 1);
    std::vector<uint16_t> w(33554432);
    for (auto &v : w) v = bf16(random(rng) / 64);
    weights.data.upload(w.data(), weights.bytes);
    V params(6144), initial(2048), input(4096), previous;
    for (auto &v : params) v = random(rng) / 8;
    for (size_t i = 0; i < 2048; ++i) params[i] += 1;
    for (auto &v : initial) v = random(rng);
    parameters.data.upload(params.data(), parameters.bytes);
    auto reset = [&](const V &state) {
      shift.upload(state.data(), shift.size());
      diag.data.upload(state.data(), shift.size());
      previous = state;
    };
    reset(initial);
    V first_input, first_output, branch_input, branch_output, branch_state;
    double worst = 0;
    auto check = [&](float actual, double expected) {
      double error = std::abs(actual - expected);
      worst = std::max(worst, error);
      if (!std::isfinite(actual) || error > 2e-5 + 2e-5 * std::abs(expected))
        throw std::runtime_error("FP64 oracle mismatch");
    };
    V actual(4096), actual_raw(16384), actual_projected(4096), state(2048),
        diagnostic(22528), ref(4096);
    std::vector<uint16_t> actual_mixed(4096), actual_active(16384);
    for (int pass = 0; pass < 4; ++pass) {
      if (pass < 2) {
        for (auto &v : input) v = random(rng) * (pass + 1);
        if (pass == 0) first_input = input;
        else { std::fill(input.begin(), input.begin() + 2048, 0); branch_input = input; }
      } else if (pass == 2) { reset(initial); input = first_input; }
      else { reset(branch_state); input = branch_input; }
      x.data.upload(input.data(), x.bytes);
      std::fill(actual.begin(), actual.end(), std::nanf(""));
      output.upload(actual.data(), output.size());
      batch();
      output.download(actual.data(), output.size());
      raw.download(actual_raw.data(), raw.size());
      projected.download(actual_projected.data(), projected.size());
      mixed.download(actual_mixed.data(), mixed.size());
      active.download(actual_active.data(), active.size());
      shift.download(state.data(), shift.size());
      for (size_t t = 0; t < 2; ++t) {
        singles[t].execute();
        diag.data.download(diagnostic.data(), diag.bytes);
        reference.data.download(ref.data(), reference.bytes);
        same(actual.data() + t * 2048, ref.data() + 2048, 2048, "residual");
        same(actual_projected.data() + t * 2048, ref.data(), 2048, "value projection");
        same(actual_raw.data() + t * 8192, diagnostic.data() + 6144, 8192, "key projection");
        double mean = 0, variance = 0;
        for (size_t i = 0; i < 2048; ++i) mean += input[t * 2048 + i];
        mean /= 2048;
        for (size_t i = 0; i < 2048; ++i) {
          double d = input[t * 2048 + i] - mean; variance += d * d;
        }
        for (size_t i = 0; i < 2048; ++i) {
          check(diagnostic[i], float((input[t * 2048 + i] - mean) /
                std::sqrt(variance / 2048 + 1e-5f)) * params[i] + params[2048 + i]);
          if (diagnostic[2048 + i] != previous[i])
            throw std::runtime_error("Reference shift continuity mismatch");
          check(diagnostic[4096 + i], diagnostic[i] +
                (previous[i] - diagnostic[i]) * params[4096 + i]);
          previous[i] = diagnostic[i];
          if (actual_mixed[t * 2048 + i] != bf16(diagnostic[4096 + i]))
            throw std::runtime_error("Mixed BF16 mismatch");
        }
        for (size_t row = 0; row < 8192; ++row) {
          double dot = 0;
          for (size_t col = 0; col < 2048; ++col)
            dot += double(expand(w[packed(row, col, 2048)])) *
                   expand(actual_mixed[t * 2048 + col]);
          check(actual_raw[t * 8192 + row], dot);
          float relu = std::max(actual_raw[t * 8192 + row], 0.f);
          if (actual_active[t * 8192 + row] != bf16(relu * relu))
            throw std::runtime_error("ReLU-square BF16 mismatch");
        }
        for (size_t row = 0; row < 2048; ++row) {
          double dot = 0;
          for (size_t col = 0; col < 8192; ++col)
            dot += double(expand(w[16777216 + packed(row, col, 8192)])) *
                   expand(actual_active[t * 8192 + col]);
          check(actual_projected[t * 2048 + row], dot);
          check(actual[t * 2048 + row], actual_projected[t * 2048 + row] + input[t * 2048 + row]);
        }
      }
      same(state.data(), previous.data(), 2048, "final shift");
      if (pass == 0) { first_output = actual; branch_state = state; }
      if (pass == 1) branch_output = actual;
      if (pass == 2) same(actual.data(), first_output.data(), 4096, "reset");
      if (pass == 3) same(actual.data(), branch_output.data(), 4096, "branch");
      for (auto *b : {&x, &parameters, &weights, &fp32, &half, &diag, &reference}) b->guard();
    }
    V unchanged_x(4096), unchanged_parameters(6144);
    std::vector<uint16_t> unchanged_w(w.size());
    x.data.download(unchanged_x.data(), x.bytes);
    parameters.data.download(unchanged_parameters.data(), parameters.bytes);
    weights.data.download(unchanged_w.data(), weights.bytes);
    if (unchanged_x != input || unchanged_parameters != params || unchanged_w != w)
      throw std::runtime_error("Immutable input changed");
    Json timings = Json::array();
    for (bool batched : {false, true, true, false}) {
      std::vector<double> times;
      times.reserve(iterations);
      for (size_t i = 0; i < iterations + 4; ++i) {
        reset(initial); // Identical state, outside the timed interval.
        auto start = Clock::now();
        if (batched) batch(); else for (auto &run : singles) run.execute();
        double us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        if (i >= 4) times.push_back(us);
      }
      double mean = std::accumulate(times.begin(), times.end(), 0.0) / iterations;
      std::sort(times.begin(), times.end());
      timings.push_back({{"path", batched ? "batched_ffn" : "production_decode"},
          {"samples", iterations}, {"mean_us_per_two_tokens", mean},
          {"p95_us_per_two_tokens", times[size_t(std::ceil(.95 * iterations)) - 1]},
          {"runs_per_two_tokens", batched ? 1 : 2}});
    }
    for (auto *b : {&x, &parameters, &weights, &fp32, &half, &diag, &reference}) b->guard();
    std::cout << Json({{"status", "passed"}, {"input_passes", 4},
        {"oracle_max_abs", worst}, {"intermediates_output_shift_bitwise", "passed"},
        {"reset_branch_guards_immutable_inputs", "passed"}, {"timing_abba", timings},
        {"scope", "Complete two-token BF16 FFN stage; model attention/WKV and prefill integration not included."}}).dump() << '\n';
  } catch (const std::exception &e) {
    std::cerr << "prefill FFN: " << e.what() << '\n';
    return 1;
  }
}
