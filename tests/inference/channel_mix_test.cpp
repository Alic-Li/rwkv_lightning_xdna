// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
using V = std::vector<float>;
int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 3)
      throw std::runtime_error(
          "usage: channel-mix-test KERNEL_ROOT [TIMING_ITERATIONS]");
    size_t iterations = 0;
    if (argc >= 3) {
      const std::string value(argv[2]);
      if (value.empty() ||
          value.find_first_not_of("0123456789") != std::string::npos)
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
    nlohmann::json config;
    std::ifstream metadata(std::filesystem::path(argv[1]) / "bf16-channel-mix/config.json");
    metadata >> config;
    const size_t C = config.at("channels"), H = config.at("hidden");
    Guarded x(fused, C * 4), params(fused, 3 * C * 4),
        weights(fused, 2 * C * H * 2), diag(fused, (3 * C + 2 * H) * 4),
        out(fused, 2 * C * 4);
    if (config.value("trace_buffer_bytes", size_t(0)))
      throw std::runtime_error("Expected untraced production artifact");
    std::vector<DeviceBuffer> arguments{x.data, params.data, weights.data,
                                        diag.data, out.data};
    auto run = fused.prepare(arguments);
    std::mt19937 rng(730);
    std::uniform_real_distribution<float> d(-1, 1);
    std::vector<uint16_t> w((2 * C * H));
    for (auto &v : w)
      v = bf16(d(rng) / 64);
    weights.data.upload(w.data(), weights.bytes);
    V input(C), constants((3 * C)), previous(C), actual_diag((3 * C + 2 * H)),
        actual(2 * C);
    for (auto &v : constants)
      v = d(rng) / 8;
    for (size_t i = 0; i < C; ++i)
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
      mean /= C;
      for (auto v : input)
        var += (v - mean) * (v - mean);
      for (size_t i = 0; i < C; ++i) {
        check(actual_diag[i],
              float((input[i] - mean) / std::sqrt(var / C + 1e-5f)) *
                      constants[i] +
                  constants[C + i]);
        if (actual_diag[C + i] != previous[i])
          throw std::runtime_error("shift snapshot mismatch");
        check(actual_diag[2 * C + i],
              actual_diag[i] +
                  (previous[i] - actual_diag[i]) * constants[2 * C + i]);
        previous[i] = actual_diag[i];
      }
      for (size_t row = 0; row < H; ++row) {
        double sum = 0;
        for (size_t col = 0; col < C; ++col)
          sum += double(expand(w[pos(row, col, C)])) *
                 expand(bf16(actual_diag[2 * C + col]));
        check(actual_diag[(3 * C) + row], sum);
        float relu = std::max(actual_diag[(3 * C) + row], 0.f);
        check(actual_diag[(3 * C + H) + row], relu * relu);
      }
      for (size_t row = 0; row < C; ++row) {
        double sum = 0;
        for (size_t col = 0; col < H; ++col)
          sum += double(expand(w[(C * H) + pos(row, col, H)])) *
                 expand(bf16(actual_diag[(3 * C + H) + col]));
        check(actual[row], sum);
        check(actual[C + row], actual[row] + input[row]);
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
                           std::chrono::steady_clock::now() - start)
                           .count();
        if (i >= 4)
          times.push_back(elapsed);
      }
      std::sort(times.begin(), times.end());
      double total = 0;
      for (double t : times)
        total += t;
      std::cout << "channel_mix_timing samples=" << iterations
                << " mean_us=" << total / iterations
                << " p50_us=" << times[(iterations - 1) / 2]
                << " p95_us=" << times[size_t(std::ceil(iterations * .95)) - 1]
                << '\n';
      for (auto *p : {&x, &params, &weights, &diag, &out})
        p->guard();
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
