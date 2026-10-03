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
    if (argc != 3) throw std::runtime_error("Usage: recurrence-projection-test FUSED_ROOT BASELINE_ROOT");
    auto session = [](const char *root, const char *name) {
      auto p = std::filesystem::path(root) / name;
      return Session(p / "design.xclbin", p / "instructions.bin");
    };
    auto fused = session(argv[1], "bf16-prefill-recurrence-projection-b2");
    auto recurrence = session(argv[2], "prefill-value-recurrence-b2");
    auto projection = session(argv[2], "bf16-prefill-output-b2-s61440");
    Guarded state(fused, 131072 * 4), aux(fused, 122880 * 4), first(fused, 57344 * 4),
        weights(fused, 4194304 * 2), residual(fused, 4096 * 4),
        ref_state(recurrence, 131072 * 4), ref_aux(recurrence, 122880 * 4), output(projection, 8192 * 4);
    auto run = fused.prepare({state.data, aux.data, first.data, weights.data, residual.data});
    auto rec = recurrence.prepare({ref_state.data, ref_aux.data, first.data});
    auto proj = projection.prepare({ref_aux.data.slice(24 * 2048 * 4, 63488 * 4), weights.data,
        residual.data.slice(0, 8192), residual.data.slice(8192, 8192), output.data});
    std::mt19937 rng(812);
    std::uniform_real_distribution<float> random(-.2f, .2f);
    V initial(131072), input(122880), fv(57344), res(4096), actual(131072), expected(131072),
        a(122880), b(122880), result(8192), saved, saved_aux;
    std::vector<uint16_t> w(4194304);
    for (auto &x : initial) x = random(rng);
    for (auto &x : input) x = random(rng);
    for (auto &x : fv) x = random(rng);
    for (auto &x : res) x = random(rng);
    for (auto &x : w) x = bf16(random(rng) / 16);
    first.data.upload(fv.data(), first.bytes);
    residual.data.upload(res.data(), residual.bytes);
    weights.data.upload(w.data(), weights.bytes);
    double worst = 0;
    for (int pass = 0; pass < 3; ++pass) {
      const V &start = pass == 1 ? saved : initial;
      for (auto *x : {&state, &ref_state}) x->data.upload(start.data(), x->bytes);
      for (auto *x : {&aux, &ref_aux}) x->data.upload(input.data(), x->bytes);
      run.execute(); rec.execute(); proj.execute();
      state.data.download(actual.data(), state.bytes);
      ref_state.data.download(expected.data(), ref_state.bytes);
      aux.data.download(a.data(), aux.bytes);
      ref_aux.data.download(b.data(), ref_aux.bytes);
      output.data.download(result.data(), output.bytes);
      if (std::memcmp(actual.data(), expected.data(), state.bytes))
        throw std::runtime_error("State differs from sequential stages");
      for (size_t t = 0; t < 2; ++t)
        for (size_t v = 0; v < 30; ++v)
          for (size_t j = 0; j < 2048; ++j) {
            const float ref = v == 25 || v == 26 ? result[t * 4096 + (v - 25) * 2048 + j]
                                                : input[t * 61440 + v * 2048 + j];
            if (std::memcmp(&a[t * 61440 + v * 2048 + j], &ref, 4))
              throw std::runtime_error("Output differs or input arena mutated");
          }
      auto check = [&](float got, double ref) {
        const double e = std::abs(double(got) - ref);
        worst = std::max(worst, e);
        if (!std::isfinite(got) || e > 2e-5 + 2e-5 * std::abs(ref))
          throw std::runtime_error("FP64 projection oracle failed");
      };
      for (size_t t = 0; t < 2; ++t)
        for (size_t row = 0; row < 2048; ++row) {
          double sum = 0;
          for (size_t col = 0; col < 2048; ++col) {
            const size_t pos = ((row / 16) * 8 + col / 256) * 4096 + row % 16 * 256 + col % 256;
            sum += double(expand(w[pos])) * expand(bf16(b[t * 61440 + 24 * 2048 + col]));
          }
          check(a[t * 61440 + 25 * 2048 + row], sum);
          check(a[t * 61440 + 26 * 2048 + row], sum + res[t * 2048 + row]);
        }
      if (pass == 0) { saved = actual; saved_aux = a; }
      if (pass == 2 && (saved != actual || saved_aux != a)) throw std::runtime_error("Reset mismatch");
    }
    auto unchanged = [](Guarded &buffer, const auto &expected) {
      auto actual = expected;
      buffer.data.download(actual.data(), buffer.bytes);
      if (actual != expected) throw std::runtime_error("Immutable input changed");
    };
    unchanged(first, fv); unchanged(weights, w); unchanged(residual, res);
    for (auto *x : {&state, &aux, &first, &weights, &residual, &ref_state, &ref_aux, &output}) x->guard();
    std::cout << "bitwise state/output, FP64 projection, reset and guards passed max_abs=" << worst << std::endl;
    for (int round = 0; round < 4; ++round) {
      const bool candidate = round == 1 || round == 2;
      const auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < 200; ++i) {
        if (candidate) run.execute(); else { rec.execute(); proj.execute(); }
      }
      std::cout << (candidate ? "fused" : "separate") << " mean_us="
          << std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 200 << std::endl;
    }
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
