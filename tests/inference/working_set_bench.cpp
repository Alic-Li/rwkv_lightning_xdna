// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <nlohmann/json.hpp>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using V = std::vector<float>;
static Json summary(std::vector<double> values) {
  const double mean = std::accumulate(values.begin(), values.end(), 0.) / values.size();
  std::sort(values.begin(), values.end());
  return {{"mean_us", mean}, {"p50_us", values[values.size()/2]},
          {"p95_us", values[(values.size()-1)*95/100]}};
}
int main(int argc, char **argv) {
  try {
    const bool int8 = argc == 3 && std::string(argv[2]) == "--int8";
    if (argc != 2 && !int8) throw std::runtime_error("usage: working-set-bench KERNEL_ROOT [--int8]");
    const auto root = std::filesystem::path(argv[1]);
    const auto path = root / (int8 ? "int8-projection-residual" : "bf16-projection-residual");
    Session projection(path / "design.xclbin", path / "instructions.bin");
    Session norm(root / "upstream-norm/design.xclbin", root / "upstream-norm/instructions.bin");
    constexpr size_t banks = 128, samples = 128;
    const size_t weight_bytes = int8 ? 4160 * 1024 : 4194304 * 2;
    Guarded x(projection, 8192), residual(projection, 8192), output(projection, 16384),
        gamma(norm, 8192), beta(norm, 8192), normalized(norm, 8192);
    V input(2048), r(2048), ones(2048, 1), zeros(2048), actual(4096);
    std::mt19937 rng(192);
    std::uniform_real_distribution<float> random(-.25f, .25f);
    for (auto &v : input) v = random(rng);
    for (auto &v : r) v = random(rng);
    x.data.upload(input.data(), x.bytes); residual.data.upload(r.data(), residual.bytes);
    gamma.data.upload(ones.data(), gamma.bytes); beta.data.upload(zeros.data(), beta.bytes);
    auto other = norm.prepare({x.data, gamma.data, beta.data, normalized.data});
    std::vector<uint16_t> base(4194304);
    for (auto &v : base) v = bf16(int8 ? std::round(random(rng) * 508) : random(rng));
    base[0] = bf16(0);
    V scales(2048);
    for (size_t row = 0; row < 2048; ++row) scales[row] = std::ldexp(1.f, int(row % 7) - 12);
    std::vector<double> oracle(2048);
    for (size_t row = 0; row < 2048; ++row)
      for (size_t col = 0; col < 2048; ++col) {
        const size_t pos = ((row / 16) * 8 + col / 256) * 4096 + row % 16 * 256 + col % 256;
        oracle[row] += double(expand(base[pos])) * expand(bf16(input[col]));
      }
    auto packed = [&](size_t bank) {
      auto weights = base;
      weights[0] = bf16(float(int(bank) - 64) / (int8 ? 1 : 256));
      std::vector<uint8_t> bytes(weight_bytes);
      if (!int8) std::memcpy(bytes.data(), weights.data(), bytes.size());
      else for (size_t tile = 0; tile < 1024; ++tile) {
        for (size_t i = 0; i < 4096; ++i)
          bytes[tile * 4160 + i] = uint8_t(int8_t(expand(weights[tile * 4096 + i])));
        std::memcpy(bytes.data() + tile * 4160 + 4096, scales.data() + (tile / 8) * 16, 64);
      }
      return bytes;
    };
    std::vector<Guarded> weights;
    std::vector<DeviceRun> runs;
    std::vector<V> expected;
    weights.reserve(banks); runs.reserve(banks);
    double worst = 0;
    for (size_t b = 0; b < banks; ++b) {
      weights.emplace_back(projection, weight_bytes);
      auto bytes = packed(b);
      weights.back().data.upload(bytes.data(), bytes.size());
      runs.push_back(projection.prepare({x.data, weights.back().data, residual.data, output.data}));
      runs.back().execute(); output.data.download(actual.data(), output.bytes);
      for (size_t row = 0; row < 2048; ++row) {
        double sum = oracle[row];
        if (!row) sum += double(int(b) - 64) / (int8 ? 1 : 256) * expand(bf16(input[0]));
        if (int8) sum *= scales[row];
        for (size_t part = 0; part < 2; ++part) {
          const double ref = sum + (part ? r[row] : 0);
          const double error = std::abs(actual[part * 2048 + row] - ref);
          worst = std::max(worst, error);
          if (!std::isfinite(actual[part * 2048 + row]) || error > 2e-5 + 2e-5 * std::abs(ref))
            throw std::runtime_error("Projection FP64 oracle mismatch");
        }
      }
      expected.push_back(actual);
    }
    other.execute();
    V norm_out(2048); normalized.data.download(norm_out.data(), normalized.bytes);
    const double mean = std::accumulate(input.begin(), input.end(), 0.) / input.size();
    double variance = 0;
    for (float v : input) variance += (v-mean)*(v-mean)/input.size();
    for (size_t i = 0; i < input.size(); ++i)
      if (!std::isfinite(norm_out[i]) || std::abs(norm_out[i] - (input[i]-mean)/std::sqrt(variance+1e-5)) > 2e-5)
        throw std::runtime_error("Switch-program norm oracle mismatch");
    std::vector<size_t> counts{1, 4, 32, 128};
    for (int round = 0; round < 2; ++round) {
      for (size_t count : counts) {
        for (int mode = 0; mode < 2; ++mode) {
          const bool alternate = (round == 0 ? mode : 1-mode) != 0;
          // A complete working-set sweep warms addresses/queues. No uploads,
          // downloads, packing, allocation or correctness work in timed loops.
          for (size_t i = 0; i < std::max(size_t(8), count); ++i) {
            if (alternate) other.execute();
            runs[i % count].execute();
          }
          std::vector<double> elapsed, submits, waits, other_elapsed;
          for (size_t i = 0; i < samples; ++i) {
            if (alternate) {
              auto begin = Clock::now(); other.execute();
              other_elapsed.push_back(std::chrono::duration<double, std::micro>(Clock::now()-begin).count());
            }
            RunTiming measured;
            auto begin = Clock::now(); runs[i % count].execute(30000, &measured);
            elapsed.push_back(std::chrono::duration<double, std::micro>(Clock::now()-begin).count());
            submits.push_back(measured.submit_us); waits.push_back(measured.wait_us);
          }
          output.data.download(actual.data(), output.bytes);
          if (actual != expected[(samples-1) % count]) throw std::runtime_error("Timed output mismatch");
          Json row{{"precision", int8 ? "w8a16" : "bf16"}, {"round", round}, {"weight_banks", count},
              {"weight_bytes_per_run", weight_bytes}, {"working_set_bytes", count * weight_bytes},
              {"mode", alternate ? "after_norm_program" : "same_program"}, {"samples", samples},
              {"projection", summary(elapsed)}, {"submit", summary(submits)}, {"wait", summary(waits)}};
          if (alternate) row["preceding_norm"] = summary(other_elapsed);
          std::cout << row.dump() << std::endl;
        }
      }
      std::reverse(counts.begin(), counts.end());
    }
    for (size_t b = 0; b < banks; ++b) {
      runs[b].execute(); output.data.download(actual.data(), output.bytes);
      if (actual != expected[b]) throw std::runtime_error("Post-timing bank mismatch");
      auto original = packed(b), got = original;
      weights[b].data.download(got.data(), got.size());
      if (got != original) throw std::runtime_error("Weight bank mutated");
      weights[b].guard();
    }
    for (auto *buffer : {&x, &residual, &gamma, &beta, &output, &normalized}) buffer->guard();
    auto immutable = [](Guarded &buffer, const V &expected) {
      V got(expected.size()); buffer.data.download(got.data(), buffer.bytes);
      if (got != expected) throw std::runtime_error("Input mutated");
    };
    immutable(x, input); immutable(residual, r); immutable(gamma, ones); immutable(beta, zeros);
    std::cout << Json({{"status", "passed"}, {"banks_verified", banks}, {"projection_fp64_max_abs", worst},
      {"scope", "Host projection wall time; excludes preceding norm time. Working-set and program-alternation sensitivity, not hardware bandwidth/utilization counters."}}).dump() << '\n';
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
