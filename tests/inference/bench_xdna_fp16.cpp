// SPDX-License-Identifier: Apache-2.0
#include "rwkv/xdna/session.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
using namespace rwkv::xdna;
static uint16_t bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return static_cast<uint16_t>((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
static float expand(uint16_t x) {
  uint32_t u = uint32_t(x) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
struct Guarded {
  DeviceBuffer root, data;
  size_t bytes;
  Guarded(Session &s, size_t n)
      : root(s.allocate(n + 128)), data(root.slice(64, n)), bytes(n) {
    std::vector<uint8_t> poison(n + 128, 0xcd);
    root.upload(poison.data(), poison.size());
  }
  void guard() {
    std::vector<uint8_t> v(bytes + 128);
    root.download(v.data(), v.size());
    for (size_t i = 0; i < 64; ++i)
      if (v[i] != 0xcd || v[bytes + 64 + i] != 0xcd)
        throw std::runtime_error("guard overwritten");
  }
};
int main(int argc, char **argv) {
  try {
    if (argc != 5)
      throw std::runtime_error("Usage: bench_xdna_fp16 ROOT fp32|bf16 K ROWS");
    const std::string precision = argv[2];
    if (precision != "fp32" && precision != "bf16")
      throw std::runtime_error("precision must be fp32 or bf16");
    const bool bf = precision == "bf16";
    const size_t k = std::stoul(argv[3]), rows = std::stoul(argv[4]);
    if ((k != 256 && k != 2048 && k != 8192) || (rows != 2048 && rows != 8192))
      throw std::runtime_error("unsupported shape");
    const auto name = (bf ? "bf16-array-gemv-" : "array-gemv-") +
                      std::to_string(k) + (rows == 8192 ? "-8192" : "");
    const auto path = std::filesystem::path(argv[1]) / name;
    Session s(path / "design.xclbin", path / "instructions.bin");
    Guarded x(s, k * 4), w(s, rows * k * (bf ? 2 : 4)), y(s, rows * 4);
    std::mt19937 rng(171);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<float> input(k), matrix(rows * k), output(rows), ref(rows);
    std::vector<double> exact(rows);
    std::vector<uint16_t> packed16(bf ? matrix.size() : 0);
    for (auto &v : input)
      v = dist(rng);
    for (size_t r = 0; r < rows; ++r) {
      double sum = 0;
      for (size_t c = 0; c < k; ++c) {
        float v = dist(rng);
        if (bf)
          v = expand(bf16(v));
        const size_t pos =
            ((r / 16) * (k / 256) + c / 256) * 4096 + (r % 16) * 256 + c % 256;
        matrix[pos] = v;
        if (bf)
          packed16[pos] = bf16(v);
        sum += double(v) * (bf ? expand(bf16(input[c])) : input[c]);
      }
      exact[r] = sum;
      ref[r] = float(sum);
    }
    x.data.upload(input.data(), x.bytes);
    w.data.upload(bf ? static_cast<const void *>(packed16.data())
                     : matrix.data(),
                  w.bytes);
    auto run = s.prepare({x.data, w.data, y.data});
    std::vector<double> times;
    double submit = 0, wait = 0;
    for (int i = 0; i < 50; ++i) {
      RunTiming t;
      run.execute(30000, &t);
      if (i >= 20) {
        times.push_back(t.submit_us + t.wait_us);
        submit += t.submit_us;
        wait += t.wait_us;
      }
    }
    y.data.download(output.data(), y.bytes);
    double max_abs = 0, mean_abs = 0, dot = 0, aa = 0, bb = 0, max_rel = 0;
    for (size_t i = 0; i < rows; ++i) {
      const double e = std::abs(output[i] - exact[i]);
      if (!std::isfinite(output[i]) || e > 2e-5 + 2e-5 * std::abs(exact[i]))
        throw std::runtime_error("GEMV numeric failure at " +
                                 std::to_string(i));
      max_abs = std::max(max_abs, e);
      mean_abs += e / rows;
      max_rel = std::max(max_rel, e / std::max(1e-8, std::abs(exact[i])));
      dot += output[i] * exact[i];
      aa += double(output[i]) * output[i];
      bb += exact[i] * exact[i];
    }
    x.guard();
    w.guard();
    y.guard();
    std::vector<float> input_after(k);
    x.data.download(input_after.data(), x.bytes);
    if (input_after != input)
      throw std::runtime_error("input modified");
    std::vector<uint8_t> weight_after(w.bytes);
    w.data.download(weight_after.data(), w.bytes);
    if (std::memcmp(weight_after.data(),
                    bf ? static_cast<const void *>(packed16.data())
                       : matrix.data(),
                    w.bytes))
      throw std::runtime_error("weight modified");
    std::sort(times.begin(), times.end());
    std::cout
        << nlohmann::json(
               {{"case", name},
                {"precision", precision},
                {"accumulator", "fp32"},
                {"status", "passed"},
                {"warmup", 20},
                {"repetitions", times.size()},
                {"min_us", times.front()},
                {"median_us", times[times.size() / 2]},
                {"max_us", times.back()},
                {"mean_submit_us", submit / times.size()},
                {"mean_wait_us", wait / times.size()},
                {"max_abs", max_abs},
                {"mean_abs", mean_abs},
                {"max_relative", max_rel},
                {"cosine", dot / std::sqrt(aa * bb)},
                {"effective_gflops", 2.0 * rows * k / times.front() / 1000},
                {"effective_weight_GBps",
                 double(w.bytes) / times.front() / 1000},
                {"timing_scope",
                 "host wall-clock, prepared run; includes DMA and scheduling"}})
               .dump()
        << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
