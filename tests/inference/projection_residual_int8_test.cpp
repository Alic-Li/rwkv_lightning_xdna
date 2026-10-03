// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include "quantization.hpp"
#include "weight_layout.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
using namespace rwkv::inference;
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
using Json = nlohmann::json;
namespace q = rwkv::inference::quantization;
struct Stage {
  Session session;
  Guarded x, w, residual, output;
  DeviceRun run;
  Stage(const std::filesystem::path &root, const char *name, size_t bytes)
      : session(root / name / "design.xclbin", root / name / "instructions.bin"),
        x(session, 2048 * 4), w(session, bytes), residual(session, 2048 * 4),
        output(session, 4096 * 4),
        run(session.prepare({x.data, w.data, residual.data, output.data})) {}
  void guards() {
    for (auto *p : {&x, &w, &residual, &output}) p->guard();
  }
};
template<class T> static void unchanged(Guarded &buffer, const std::vector<T> &expected) {
  std::vector<T> actual(expected.size());
  buffer.data.download(actual.data(), buffer.bytes);
  if (actual != expected) throw std::runtime_error("Input or weight mutated");
}
static Json timing(Stage &s, size_t count) {
  std::vector<double> us;
  for (size_t i = 0; i < count + 4; ++i) {
    auto begin = std::chrono::steady_clock::now();
    s.run.execute();
    double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
    if (i >= 4) us.push_back(t);
  }
  double mean = 0;
  for (double t : us) mean += t / us.size();
  std::sort(us.begin(), us.end());
  return {{"samples", count}, {"mean_us", mean}, {"p50_us", us[count/2]}, {"p95_us", us[(count-1)*95/100]}};
}
int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::runtime_error("Usage: projection-residual-int8-test BF16_ROOT INT8_ROOT");
    Tensor tensor;
    tensor.shape = {2048, 2048}; tensor.data.resize(2048 * 2048);
    std::mt19937 rng(193);
    std::uniform_real_distribution<float> d(-.25f, .25f);
    for (size_t row = 0; row < 2048; ++row)
      for (size_t col = 0; col < 2048; ++col)
        tensor.data[row * 2048 + col] = row % 127 == 0 ? 0 : std::ldexp(d(rng), int(row % 7) - 4);
    auto quant = q::quantize(tensor);
    auto packed = q::pack(quant);
    auto bf = weight_layout::to_bf16(weight_layout::projection(tensor, false, 0, 2048, 2048));
    Stage baseline(argv[1], "bf16-projection-residual", bf.size() * 2);
    Stage candidate(argv[2], "int8-projection-residual", packed.size());
    baseline.w.data.upload(bf.data(), baseline.w.bytes);
    candidate.w.data.upload(packed.data(), candidate.w.bytes);
    Vector input(2048), residual(2048), a(4096), b(4096);
    double worst = 0, delta_square = 0, reference_square = 0;
    auto check = [&](float actual, double expected) {
      double e = std::abs(double(actual) - expected);
      worst = std::max(worst, e);
      if (!std::isfinite(actual) || e > 2e-5 + 2e-5 * std::abs(expected))
        throw std::runtime_error("Independent FP64 projection/residual oracle mismatch");
    };
    for (int pass = 0; pass < 4; ++pass) {
      for (auto &v : input) v = pass == 3 ? 0 : d(rng) * (pass + 1);
      for (auto &v : residual) v = d(rng);
      for (auto *s : {&baseline, &candidate}) {
        s->x.data.upload(input.data(), s->x.bytes);
        s->residual.data.upload(residual.data(), s->residual.bytes);
        s->run.execute();
        s->guards();
        unchanged(s->x, input); unchanged(s->residual, residual);
      }
      baseline.output.data.download(b.data(), baseline.output.bytes);
      candidate.output.data.download(a.data(), candidate.output.bytes);
      for (size_t row = 0; row < 2048; ++row) {
        double qb = 0, bb = 0;
        for (size_t col = 0; col < 2048; ++col) {
          float x = expand(bf16(input[col]));
          qb += double(quant.codes[row * 2048 + col]) * x;
          bb += double(expand(bf16(tensor.data[row * 2048 + col]))) * x;
        }
        qb *= quant.scales[row];
        check(a[row], qb); check(a[2048 + row], qb + residual[row]);
        check(b[row], bb); check(b[2048 + row], bb + residual[row]);
        delta_square += std::pow(double(a[row]) - b[row], 2);
        reference_square += double(b[row]) * b[row];
      }
    }
    unchanged(baseline.w, bf); unchanged(candidate.w, packed);
    // Restore a nonzero input for timing; transfers and guards stay outside it.
    for (auto &v : input) v = d(rng);
    baseline.x.data.upload(input.data(), baseline.x.bytes);
    candidate.x.data.upload(input.data(), candidate.x.bytes);
    Json abba = Json::array();
    for (auto *s : {&baseline, &candidate, &candidate, &baseline})
      abba.push_back({{"weights", s == &baseline ? "bf16" : "int8"}, {"timing", timing(*s, 200)}});
    for (auto *s : {&baseline, &candidate}) {
      s->guards(); unchanged(s->x, input); unchanged(s->residual, residual);
    }
    unchanged(baseline.w, bf); unchanged(candidate.w, packed);
    std::cout << Json({{"status", "passed"}, {"oracle_max_abs", worst},
      {"oracle_tolerance", "2e-5 + 2e-5*abs(reference), FP64 dot and scale"},
      {"input_passes", 4}, {"zero_rows_and_input", "passed"},
      {"guards_and_immutable_inputs", "passed"}, {"abba", abba},
      {"synthetic_projection_relative_l2_vs_bf16", std::sqrt(delta_square/reference_square)},
      {"bf16_weight_bytes", baseline.w.bytes}, {"int8_weight_bytes", candidate.w.bytes}}).dump() << '\n';
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
