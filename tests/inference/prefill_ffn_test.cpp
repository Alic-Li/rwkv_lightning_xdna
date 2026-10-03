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
    bool projection_input = false, quantized = false, recurrence_input = false, chunk4 = false;
    while (argc > 1 && std::string(argv[argc - 1]).rfind("--", 0) == 0) {
      const std::string flag(argv[--argc]);
      if (flag == "--chunk4") chunk4 = true;
      else if (flag == "--projection-input") projection_input = true;
      else if (flag == "--recurrence-input") { recurrence_input = true; projection_input = true; }
      else if (flag == "--int8") quantized = true;
      else throw std::runtime_error("Unknown option: " + flag);
    }
    if (argc < 3 || argc > 4)
      throw std::runtime_error("Usage: rwkv-prefill-ffn-test FUSED_KERNELS DECODE_KERNELS [ITERATIONS] [--projection-input|--recurrence-input] [--int8] [--chunk4]");
    size_t iterations = 100;
    if (argc == 4) {
      std::string count(argv[3]);
      if (count.empty() || count.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid iterations");
      iterations = std::stoull(count);
      if (!iterations || iterations > 1000000)
        throw std::runtime_error("Iterations outside [1, 1000000]");
    }
    auto session = [](std::filesystem::path root, const std::string &name) {
      return Session(root / name / "design.xclbin", root / name / "instructions.bin");
    };
    const std::string prefix = quantized ? "int8" : "bf16";
    if (chunk4 && quantized)
      throw std::runtime_error("Chunk4 experiment supports BF16 only");
    const size_t batch_size = chunk4 ? 4 : 2;
    const size_t tokens_c = batch_size * 2048, tokens_h = batch_size * 8192;
    const size_t projected_offset = 2048 + tokens_h;
    const size_t output_offset = projected_offset + tokens_c;
    const size_t arena_floats = output_offset + tokens_c;
    const size_t half_elements = tokens_c + tokens_h;
    auto fused = session(argv[1], chunk4 ? std::string("bf16-chunk4-ffn-experiment") + (projection_input ? "-projection-input" : "") : prefix + "-prefill-ffn-b2" +
                         (recurrence_input ? "-recurrence-input" : projection_input ? "-projection-input" : ""));
    const size_t input_stride = recurrence_input ? 61440 : 4096;
    auto input_offset = [&](size_t t) { return projection_input ? t * input_stride + 2048 : t * 2048; };
    auto decode = session(argv[2], prefix + "-channel-mix");
    Guarded x(fused, (projection_input ? (batch_size - 1) * input_stride + 4096 : tokens_c) * 4), parameters(fused, 6144 * 4),
        weights(fused, quantized ? 8192 * 4160 : 33554432 * 2), fp32(fused, arena_floats * 4), half(fused, half_elements * 2),
        diag(decode, 22528 * 4), reference(decode, 4096 * 4);
    auto shift = fp32.data.slice(0, 2048 * 4);
    auto raw = fp32.data.slice(2048 * 4, tokens_h * 4);
    auto projected = fp32.data.slice(projected_offset * 4, tokens_c * 4);
    auto output = fp32.data.slice(output_offset * 4, tokens_c * 4);
    auto mixed = half.data.slice(0, tokens_c * 2);
    auto active = half.data.slice(tokens_c * 2, tokens_h * 2);
    auto fused_run = fused.prepare({x.data, parameters.data, weights.data, fp32.data, half.data});
    std::vector<DeviceRun> singles;
    for (size_t t = 0; t < batch_size; ++t)
      singles.push_back(decode.prepare({x.data.slice(input_offset(t) * 4, 2048 * 4),
          parameters.data, weights.data, diag.data, reference.data}));
    auto batch = [&] { fused_run.execute(); };
    std::mt19937 rng(730);
    std::uniform_real_distribution<float> random(-1, 1);
    std::vector<uint16_t> w(33554432);
    for (auto &v : w) v = bf16(random(rng) / 64);
    std::vector<uint8_t> weight_bytes(weights.bytes);
    if (quantized) {
      // Self-contained packed INT8 tiles with FP16-exact row scales. Include
      // signed endpoints, zero rows and varying scales; oracle weights below
      // use code*scale without BF16 weight rounding in the FP64 oracle.
      for (size_t matrix = 0; matrix < 2; ++matrix) {
        const size_t k = matrix ? 8192 : 2048;
        const size_t rows = matrix ? 2048 : 8192;
        for (size_t row = 0; row < rows; ++row) {
          const float scale = std::ldexp(1.f + float(row % 8) / 16.f, -14 + int(row % 4));
          for (size_t col = 0; col < k; ++col) {
            const size_t index = matrix * 16777216 + packed(row, col, k);
            const int8_t code = row == 0 ? 0 : int8_t(int((row * 31 + col * 17) % 255) - 127);
            weight_bytes[(index / 4096) * 4160 + index % 4096] = uint8_t(code);
            w[index] = bf16(float(code) * scale);
            if (col % 256 == 0)
              std::memcpy(weight_bytes.data() + (index / 4096) * 4160 + 4096 + (row % 16) * 4,
                          &scale, 4);
          }
        }
      }
    } else std::memcpy(weight_bytes.data(), w.data(), weights.bytes);
    weights.data.upload(weight_bytes.data(), weights.bytes);
    auto oracle_weight = [&](size_t index) -> float {
      if (!quantized) return expand(w[index]);
      const size_t tile = (index / 4096) * 4160;
      const int8_t code = static_cast<int8_t>(weight_bytes[tile + index % 4096]);
      float scale;
      std::memcpy(&scale, weight_bytes.data() + tile + 4096 + ((index % 4096) / 256) * 4, 4);
      // Do not round dequantized weights to BF16: resident W8A16 applies the
      // row scale after accumulating the integer-code dot product.
      return float(code) * scale;
    };
    V params(6144), initial(2048), input(tokens_c), previous;
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
    V packed_input(x.bytes / 4, 123.25f); // Nonzero unused projection lanes catch wrong gathers.
    V first_input, first_output, branch_input, branch_output, branch_state;
    double worst = 0;
    auto check = [&](float actual, double expected) {
      double error = std::abs(actual - expected);
      worst = std::max(worst, error);
      if (!std::isfinite(actual) || error > 2e-5 + 2e-5 * std::abs(expected))
        throw std::runtime_error("FP64 oracle mismatch");
    };
    V actual(tokens_c), actual_raw(tokens_h), actual_projected(tokens_c), state(2048),
        diagnostic(22528), ref(4096);
    std::vector<uint16_t> actual_mixed(tokens_c), actual_active(tokens_h);
    for (int pass = 0; pass < 4; ++pass) {
      if (pass < 2) {
        for (auto &v : input) v = random(rng) * (pass + 1);
        if (pass == 0) first_input = input;
        else { std::fill(input.begin(), input.begin() + 2048, 0); branch_input = input; }
      } else if (pass == 2) { reset(initial); input = first_input; }
      else { reset(branch_state); input = branch_input; }
      for (size_t t = 0; t < batch_size; ++t)
        std::copy_n(input.data() + t * 2048, 2048, packed_input.data() + input_offset(t));
      x.data.upload(packed_input.data(), x.bytes);
      std::fill(actual.begin(), actual.end(), std::nanf(""));
      output.upload(actual.data(), output.size());
      batch();
      output.download(actual.data(), output.size());
      raw.download(actual_raw.data(), raw.size());
      projected.download(actual_projected.data(), projected.size());
      mixed.download(actual_mixed.data(), mixed.size());
      active.download(actual_active.data(), active.size());
      shift.download(state.data(), shift.size());
      for (size_t t = 0; t < batch_size; ++t) {
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
            dot += double(oracle_weight(packed(row, col, 2048))) *
                   expand(actual_mixed[t * 2048 + col]);
          check(actual_raw[t * 8192 + row], dot);
          float relu = std::max(actual_raw[t * 8192 + row], 0.f);
          if (actual_active[t * 8192 + row] != bf16(relu * relu))
            throw std::runtime_error("ReLU-square BF16 mismatch");
        }
        for (size_t row = 0; row < 2048; ++row) {
          double dot = 0;
          for (size_t col = 0; col < 8192; ++col)
            dot += double(oracle_weight(16777216 + packed(row, col, 8192))) *
                   expand(actual_active[t * 8192 + col]);
          check(actual_projected[t * 2048 + row], dot);
          check(actual[t * 2048 + row], actual_projected[t * 2048 + row] + input[t * 2048 + row]);
        }
      }
      same(state.data(), previous.data(), 2048, "final shift");
      if (pass == 0) { first_output = actual; branch_state = state; }
      if (pass == 1) branch_output = actual;
      if (pass == 2) same(actual.data(), first_output.data(), tokens_c, "reset");
      if (pass == 3) same(actual.data(), branch_output.data(), tokens_c, "branch");
      for (auto *b : {&x, &parameters, &weights, &fp32, &half, &diag, &reference}) b->guard();
    }
    V unchanged_x(packed_input.size()), unchanged_parameters(6144);
    std::vector<uint8_t> unchanged_w(weight_bytes.size());
    x.data.download(unchanged_x.data(), x.bytes);
    parameters.data.download(unchanged_parameters.data(), parameters.bytes);
    weights.data.download(unchanged_w.data(), weights.bytes);
    if (unchanged_x != packed_input || unchanged_parameters != params || unchanged_w != weight_bytes)
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
          {"samples", iterations}, {"mean_us_per_batch", mean},
          {"p95_us_per_batch", times[size_t(std::ceil(.95 * iterations)) - 1]},
          {"runs_per_batch", batched ? size_t(1) : batch_size}});
    }
    for (auto *b : {&x, &parameters, &weights, &fp32, &half, &diag, &reference}) b->guard();
    std::cout << Json({{"status", "passed"}, {"input_passes", 4}, {"batch_tokens", batch_size}, {"weights", prefix},
        {"input_layout", recurrence_input ? "recurrence_projection_pairs" : projection_input ? "projection_residual_pairs" : "token_major"},
        {"oracle_max_abs", worst}, {"intermediates_output_shift_bitwise", "passed"},
        {"reset_branch_guards_immutable_inputs", "passed"}, {"timing_abba", timings},
        {"scope", "Complete FFN batch stage; model attention/WKV and prefill integration not included."}}).dump() << '\n';
  } catch (const std::exception &e) {
    std::cerr << "prefill FFN: " << e.what() << '\n';
    return 1;
  }
}
