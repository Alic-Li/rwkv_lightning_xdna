// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
using V = std::vector<float>;
int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 4)
      throw std::runtime_error("usage: channel-mix-test KERNEL_ROOT [TIMING_ITERATIONS [TRACE_FILE]]");
    size_t iterations = 0;
    if (argc >= 3) {
      const std::string value(argv[2]);
      if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid timing iterations");
      iterations = std::stoull(value);
      if (!iterations || iterations > 1000000)
        throw std::runtime_error("Invalid timing iterations");
    }
    auto session = [](std::filesystem::path root, std::string name) {
      return Session(root / name / "design.xclbin",
                     root / name / "instructions.bin");
    };
    auto fused = session(argv[1], "bf16-channel-mix");
    Guarded x(fused, 2048 * 4), params(fused, 6144 * 4),
        weights(fused, 33554432 * 2), diag(fused, 22528 * 4),
        out(fused, 4096 * 4);
    nlohmann::json config;
    std::ifstream metadata(std::filesystem::path(argv[1]) / "bf16-channel-mix/config.json");
    metadata >> config;
    size_t trace_bytes = config.value("trace_buffer_bytes", size_t(0));
    if (bool(trace_bytes) != (argc == 4))
      throw std::runtime_error("Trace artifact and TRACE_FILE must be supplied together");
    std::vector<DeviceBuffer> arguments{x.data, params.data, weights.data, diag.data, out.data};
    std::unique_ptr<Guarded> trace;
    if (trace_bytes) {
      if (trace_bytes > 64 * 1024 * 1024 || trace_bytes % 4)
        throw std::runtime_error("Invalid trace buffer size");
      trace = std::make_unique<Guarded>(fused, trace_bytes);
      std::vector<uint32_t> empty(trace_bytes / 4);
      trace->data.upload(empty.data(), trace_bytes);
      arguments.push_back(trace->data);
    }
    auto run = fused.prepare(arguments);
    std::mt19937 rng(730);
    std::uniform_real_distribution<float> d(-1, 1);
    std::vector<uint16_t> w(33554432);
    for (auto &v : w)
      v = bf16(d(rng) / 64);
    weights.data.upload(w.data(), weights.bytes);
    V input(2048), constants(6144), previous(2048), actual_diag(22528),
        actual(4096);
    for (auto &v : constants)
      v = d(rng) / 8;
    for (size_t i = 0; i < 2048; ++i)
      constants[i] += 1;
    for (auto &v : previous)
      v = d(rng);
    params.data.upload(constants.data(), params.bytes);
    diag.data.upload(previous.data(), previous.size() * 4);
    auto pos = [](size_t row, size_t col, size_t K) {
      return ((row / 16) * (K / 256) + col / 256) * 4096 + (row % 16) * 256 +
             col % 256;
    };
    double worst = 0;
    auto check = [&](float actual, double ref) {
      double e = std::abs(double(actual) - ref);
      worst = std::max(worst, e);
      if (!std::isfinite(actual) || e > 2e-5 + 2e-5 * std::abs(ref))
        throw std::runtime_error("ChannelMix oracle error=" +
                                 std::to_string(e));
    };
    for (int pass = 0; pass < 3; ++pass) {
      for (auto &v : input)
        v = d(rng) * (pass + 1);
      x.data.upload(input.data(), x.bytes);
      run.execute();
      diag.data.download(actual_diag.data(), diag.bytes);
      out.data.download(actual.data(), out.bytes);
      double mean = 0, var = 0;
      for (auto v : input)
        mean += v;
      mean /= 2048;
      for (auto v : input)
        var += (v - mean) * (v - mean);
      for (size_t i = 0; i < 2048; ++i) {
        check(actual_diag[i],
              float((input[i] - mean) / std::sqrt(var / 2048 + 1e-5f)) *
                      constants[i] +
                  constants[2048 + i]);
        if (actual_diag[2048 + i] != previous[i])
          throw std::runtime_error("shift snapshot mismatch");
        check(actual_diag[4096 + i],
              actual_diag[i] +
                  (previous[i] - actual_diag[i]) * constants[4096 + i]);
        previous[i] = actual_diag[i];
      }
      for (size_t row = 0; row < 8192; ++row) {
        double sum = 0;
        for (size_t col = 0; col < 2048; ++col)
          sum += double(expand(w[pos(row, col, 2048)])) *
                 expand(bf16(actual_diag[4096 + col]));
        check(actual_diag[6144 + row], sum);
        float relu = std::max(actual_diag[6144 + row], 0.f);
        check(actual_diag[14336 + row], relu * relu);
      }
      for (size_t row = 0; row < 2048; ++row) {
        double sum = 0;
        for (size_t col = 0; col < 8192; ++col)
          sum += double(expand(w[16777216 + pos(row, col, 8192)])) *
                 expand(bf16(actual_diag[14336 + col]));
        check(actual[row], sum);
        check(actual[2048 + row], actual[row] + input[row]);
      }
    }
    for (auto *p : {&x, &params, &weights, &diag, &out})
      p->guard();
    V ix(input.size()), cp(constants.size());
    x.data.download(ix.data(), x.bytes);
    params.data.download(cp.data(), params.bytes);
    std::vector<uint16_t> same(w.size());
    weights.data.download(same.data(), weights.bytes);
    if (ix != input || cp != constants || same != w)
      throw std::runtime_error("immutable input overwritten");
    std::cout << "ChannelMix oracle/shift/replay/guards passed max_abs="
              << worst << '\n';
    if (iterations) {
      std::vector<double> times;
      // Keep a fixed recurrent shift and input across samples. Reset is outside
      // the timer; all BOs, instructions and the XRT run are reused.
      for (size_t i = 0; i < iterations + 4; ++i) {
        diag.data.upload(previous.data(), previous.size() * 4);
        auto start = std::chrono::steady_clock::now();
        run.execute();
        auto elapsed = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
        if (i >= 4)
          times.push_back(elapsed);
      }
      std::sort(times.begin(), times.end());
      double total = 0;
      for (double t : times) total += t;
      std::cout << "channel_mix_timing samples=" << iterations
                << " mean_us=" << total / iterations
                << " p50_us=" << times[(iterations - 1) / 2]
                << " p95_us=" << times[size_t(std::ceil(iterations * .95)) - 1]
                << '\n';
      for (auto *p : {&x, &params, &weights, &diag, &out}) p->guard();
    }
    if (trace) {
      // Record exactly one dispatch, with a cleared buffer, after validation.
      std::vector<uint32_t> words(trace_bytes / 4);
      trace->data.upload(words.data(), trace_bytes);
      run.execute();
      trace->data.download(words.data(), trace_bytes);
      trace->guard();
      for (auto *p : {&x, &params, &weights, &diag, &out}) p->guard();
      if (words.back())
        throw std::runtime_error("Trace buffer filled; increase RWKV_XDNA_TRACE_BYTES");
      while (!words.empty() && words.back() == 0) words.pop_back();
      if (words.empty()) throw std::runtime_error("Empty hardware trace");
      std::ofstream output(argv[3]);
      for (auto word : words)
        output << std::hex << std::setfill('0') << std::setw(8) << word << '\n';
      if (!output) throw std::runtime_error("Cannot write hardware trace");
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
