// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
using V = std::vector<float>;
int main(int argc, char **argv) {
  try {
    const bool int8 = argc == 4 && std::string(argv[3]) == "--int8";
    if (int8) --argc;
    if (argc != 3)
      throw std::runtime_error("usage: prefill-output-test BATCH_ROOT DECODE_ROOT [--int8]");
    auto session = [](std::filesystem::path root, std::string name) {
      return Session(root / name / "design.xclbin",
                     root / name / "instructions.bin");
    };
    for (size_t stride : {2048, 55296, 61440}) {
      std::cout << "stride=" << stride << '\n';
      auto fused = session(argv[1], std::string(int8 ? "int8" : "bf16") + "-prefill-output-b2-s" + std::to_string(stride));
      auto single = session(argv[2], int8 ? "int8-projection-residual" : "bf16-projection-residual");
      Guarded x(fused, (stride + 2048) * 4), w(fused, int8 ? 4160 * 1024 : 4194304 * 2), res(fused, 4096 * 4),
          out(fused, 8192 * 4);
      auto run = fused.prepare({x.data, w.data, res.data.slice(0, 8192), res.data.slice(8192, 8192), out.data});
      Guarded reference(single, 8192 * 4);
      std::vector<DeviceRun> singles;
      for (size_t t = 0; t < 2; ++t)
        singles.push_back(single.prepare({x.data.slice(t * stride * 4, 8192), w.data,
            res.data.slice(t * 8192, 8192), reference.data.slice(t * 16384, 16384)}));
      std::mt19937 rng(192);
      std::uniform_real_distribution<float> d(-.25f, .25f);
      std::vector<uint16_t> weights(4194304);
      for (auto &v : weights)
        v = bf16(int8 ? std::round(d(rng) * 508) : d(rng));
      std::vector<uint8_t> packed(4160 * 1024);
      V scales(2048);
      for (size_t row = 0; row < 2048; ++row) scales[row] = std::ldexp(1.f, int(row % 7) - 12);
      if (int8) {
        for (size_t tile = 0; tile < 1024; ++tile) {
          for (size_t i = 0; i < 4096; ++i)
            packed[tile * 4160 + i] = uint8_t(int8_t(expand(weights[tile * 4096 + i])));
          std::memcpy(packed.data() + tile * 4160 + 4096, scales.data() + (tile / 8) * 16, 64);
        }
        w.data.upload(packed.data(), w.bytes);
      } else w.data.upload(weights.data(), w.bytes);
      V input(stride + 2048), r(4096), actual(8192), expected(8192);
      double worst = 0;
      for (int pass = 0; pass < 3; ++pass) {
        for (auto &v : input)
          v = d(rng) * (pass + 1);
        for (auto &v : r)
          v = d(rng);
        x.data.upload(input.data(), x.bytes);
        res.data.upload(r.data(), res.bytes);
        run.execute();
        out.data.download(actual.data(), out.bytes);
        for (auto &entry : singles) entry.execute();
        reference.data.download(expected.data(), reference.bytes);
        if (std::memcmp(actual.data(), expected.data(), out.bytes))
          throw std::runtime_error("Bitwise mismatch against two single-token runs");
        auto check = [&](float actual, double ref) {
          double e = std::abs(double(actual) - ref);
          worst = std::max(worst, e);
          if (!std::isfinite(actual) || e > 2e-5 + 2e-5 * std::abs(ref))
            throw std::runtime_error("independent oracle mismatch");
        };
        for (size_t token = 0; token < 2; ++token) {
          for (size_t row = 0; row < 2048; ++row) {
            double sum = 0;
            for (size_t col = 0; col < 2048; ++col) {
              size_t pos = ((row / 16) * 8 + col / 256) * 4096 + (row % 16) * 256 +
                           col % 256;
              sum += double(expand(weights[pos])) * expand(bf16(input[token * stride + col]));
            }
            if (int8) sum *= scales[row];
            check(actual[token * 4096 + row], sum);
            check(actual[token * 4096 + 2048 + row], actual[token * 4096 + row] + r[token * 2048 + row]);
          }
        }
      }
      for (auto *p : {&x, &w, &res, &out, &reference})
        p->guard();
      using Clock = std::chrono::steady_clock;
      for (int round = 0; round < 4; ++round) {
        const bool batch = round == 1 || round == 2;
        const auto start = Clock::now();
        for (int i = 0; i < 200; ++i) {
          if (batch) run.execute();
          else for (auto &entry : singles) entry.execute();
        }
        const double us = std::chrono::duration<double, std::micro>(Clock::now() - start).count() / 200;
        std::cout << (batch ? "batch" : "single_pair") << " mean_us=" << us << '\n';
      }
      std::cout << "projection/residual oracle/replay/guards passed max_abs="
                << worst << '\n';
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
