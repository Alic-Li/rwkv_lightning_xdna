// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <nlohmann/json.hpp>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
using V = std::vector<float>;
using Json = nlohmann::json;
static size_t packed(size_t row, size_t col, size_t k) {
  return ((row / 16) * (k / 256) + col / 256) * 4096 + row % 16 * 256 + col % 256;
}
static void same(const float *a, const float *b, size_t n) {
  if (std::memcmp(a, b, n * 4)) throw std::runtime_error("Bitwise output/arena mismatch");
}
int main(int argc, char **argv) {
  try {
    if (argc < 3 || argc > 4)
      throw std::runtime_error("Usage: rwkv-prefill-attention-test BATCH_ROOT DECODE_ROOT [ITERATIONS]");
    size_t iterations = 200;
    if (argc == 4) {
      std::string s(argv[3]);
      if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid iterations");
      iterations = std::stoull(s);
      if (!iterations || iterations > 1000000) throw std::runtime_error("Invalid iterations");
    }
    for (size_t branches : {3, 4}) {
      const auto name = "bf16-attention-projections-" + std::to_string(branches);
      auto session = [&](const char *root, const std::string &suffix) {
        auto p = std::filesystem::path(root) / (name + suffix);
        return Session(p / "design.xclbin", p / "instructions.bin");
      };
      auto batch = session(argv[1], "-b2"), decode = session(argv[2], "");
      const size_t nrkv = 3 * 2048 * 2048, nrank = branches * 256 * 2048;
      const size_t weight_elements = nrkv + 2 * nrank;
      const size_t stride = branches == 4 ? 61440 : 55296;
      const size_t vstride = branches == 4 ? stride : 6144;
      Guarded x(batch, 24576 * 4), w(batch, weight_elements * 2),
          arena(batch, stride * 2 * 4), value(batch, (vstride + 6144) * 4),
          rank(batch, branches * 1024 * 4), ra(decode, 55296 * 4),
          rv(decode, 6144 * 4), rr(decode, branches * 512 * 4);
      auto fused = batch.prepare({x.data, w.data, arena.data, value.data, rank.data});
      std::vector<DeviceRun> singles;
      for (size_t t = 0; t < 2; ++t)
        singles.push_back(decode.prepare({x.data.slice(t * 12288 * 4, 12288 * 4),
                                         w.data, ra.data, rv.data, rr.data}));
      std::mt19937 rng(713 + branches);
      std::uniform_real_distribution<float> random(-1, 1);
      std::vector<uint16_t> weights(nrkv + 2 * nrank);
      for (auto &v : weights) v = bf16(random(rng) / 64);
      w.data.upload(weights.data(), w.bytes);
      V input(24576), first, first_a, first_v, first_r;
      V actual_a(stride * 2), actual_v(vstride + 6144), actual_r(branches * 1024);
      V ref_a(55296), ref_v(6144), ref_r(branches * 512);
      double worst = 0;
      auto check = [&](float actual, double expected) {
        const double error = std::abs(double(actual) - expected);
        worst = std::max(worst, error);
        if (!std::isfinite(actual) || error > 2e-5 + 2e-5 * std::abs(expected))
          throw std::runtime_error("FP64 projection/activation oracle mismatch");
      };
      auto reset = [](Guarded &buffer, V &values) {
        std::fill(values.begin(), values.end(), -777.f);
        buffer.data.upload(values.data(), buffer.bytes);
      };
      for (size_t pass = 0; pass < 3; ++pass) {
        if (pass < 2) {
          for (auto &v : input) v = random(rng) * (pass + 1);
          if (pass == 0) first = input;
          else std::fill(input.begin(), input.begin() + 12288, 0.f);
        } else input = first;
        x.data.upload(input.data(), x.bytes);
        reset(arena, actual_a); reset(value, actual_v); reset(rank, actual_r);
        fused.execute();
        arena.data.download(actual_a.data(), arena.bytes);
        value.data.download(actual_v.data(), value.bytes);
        rank.data.download(actual_r.data(), rank.bytes);
        for (size_t t = 0; t < 2; ++t) {
          reset(ra, ref_a); reset(rv, ref_v); reset(rr, ref_r);
          singles[t].execute();
          ra.data.download(ref_a.data(), ra.bytes);
          rv.data.download(ref_v.data(), rv.bytes);
          rr.data.download(ref_r.data(), rr.bytes);
          same(actual_a.data() + t * stride, ref_a.data(), ref_a.size());
          same(actual_v.data() + t * vstride, ref_v.data(), ref_v.size());
          same(actual_r.data() + t * branches * 512, ref_r.data(), ref_r.size());
          for (size_t i = 55296; i < stride; ++i)
            if (actual_a[t * stride + i] != -777.f) throw std::runtime_error("Arena gap overwritten");
          const size_t slots[] = {0, 2, 3};
          for (size_t p = 0; p < 3; ++p) {
            const float *out = p == 2 && branches == 4 ? actual_v.data() + t * vstride
                : actual_a.data() + t * stride + (p == 0 ? 13 * 2048 : p == 1 ? 0 : 16 * 2048);
            for (size_t row = 0; row < 2048; ++row) {
              double dot = 0;
              const size_t base = (row / 256) * 3 * 256 * 2048 + p * 256 * 2048;
              for (size_t col = 0; col < 2048; ++col)
                dot += double(expand(weights[base + packed(row % 256, col, 2048)])) *
                       expand(bf16(input[t * 12288 + slots[p] * 2048 + col]));
              check(out[row], dot);
            }
          }
          const size_t inputs[] = {1, 4, 5, 3};
          const size_t outputs[] = {4096, 2048, 12 * 2048};
          for (size_t p = 0; p < branches; ++p) {
            const float *raw = actual_r.data() + t * branches * 512 + p * 256;
            const float *active = raw + branches * 256;
            for (size_t row = 0; row < 256; ++row) {
              double dot = 0;
              const size_t base = nrkv + (row / 128) * branches * 128 * 2048 + p * 128 * 2048;
              for (size_t col = 0; col < 2048; ++col)
                dot += double(expand(weights[base + packed(row % 128, col, 2048)])) *
                       expand(bf16(input[t * 12288 + inputs[p] * 2048 + col]));
              check(raw[row], dot);
              check(active[row], p == 0 ? std::tanh(double(raw[row]))
                                  : p == 2 ? 1.0 / (1.0 + std::exp(-double(raw[row]))) : double(raw[row]));
            }
            const float *out = p == 3 ? actual_v.data() + t * vstride + 2048
                                     : actual_a.data() + t * stride + outputs[p];
            for (size_t row = 0; row < 2048; ++row) {
              double dot = 0;
              const size_t base = nrkv + nrank + (row / 512) * branches * 512 * 256 + p * 512 * 256;
              for (size_t col = 0; col < 256; ++col)
                dot += double(expand(weights[base + packed(row % 512, col, 256)])) * expand(bf16(active[col]));
              check(out[row], dot);
            }
          }
        }
        for (size_t i = 6144; i < vstride; ++i)
          if (actual_v[i] != -777.f) throw std::runtime_error("Value gap overwritten");
        if (pass == 0) { first_a = actual_a; first_v = actual_v; first_r = actual_r; }
        if (pass == 2) {
          same(actual_a.data(), first_a.data(), actual_a.size());
          same(actual_v.data(), first_v.data(), actual_v.size());
          same(actual_r.data(), first_r.data(), actual_r.size());
        }
        for (auto *b : {&x, &w, &arena, &value, &rank, &ra, &rv, &rr}) b->guard();
      }
      V same_x(input.size()); std::vector<uint16_t> same_w(weights.size());
      x.data.download(same_x.data(), x.bytes); w.data.download(same_w.data(), w.bytes);
      if (same_x != input || same_w != weights) throw std::runtime_error("Immutable input changed");
      Json timings = Json::array();
      for (bool paired : {false, true, true, false}) {
        std::vector<double> us;
        for (size_t i = 0; i < iterations + 4; ++i) {
          const auto start = std::chrono::steady_clock::now();
          if (paired) fused.execute(); else for (auto &run : singles) run.execute();
          const double elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
          if (i >= 4) us.push_back(elapsed);
        }
        const double mean = std::accumulate(us.begin(), us.end(), 0.0) / us.size();
        std::sort(us.begin(), us.end());
        timings.push_back({{"path", paired ? "batched" : "two_decode"}, {"samples", iterations},
            {"mean_us", mean}, {"p95_us", us[size_t(std::ceil(.95 * us.size())) - 1]}});
      }
      for (auto *b : {&x, &w, &arena, &value, &rank, &ra, &rv, &rr}) b->guard();
      std::cout << Json({{"status", "passed"}, {"branches", branches}, {"passes", 3},
          {"oracle_max_abs", worst}, {"bitwise_outputs_replay_guards_immutable_inputs", "passed"},
          {"timing_abba", timings}}).dump() << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "prefill attention: " << e.what() << '\n'; return 1;
  }
}
