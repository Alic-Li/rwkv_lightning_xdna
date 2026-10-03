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
    if (argc != 3) throw std::runtime_error("Usage: prefill-recurrence-test BATCH_ROOT DECODE_ROOT");
    auto session = [](const char *root, const char *name) {
      auto p = std::filesystem::path(root) / name;
      return Session(p / "design.xclbin", p / "instructions.bin");
    };
    for (bool value : {false, true}) {
      auto batch = session(argv[1], value ? "prefill-value-recurrence-b2" : "prefill-recurrence-b2");
      auto decode = session(argv[2], value ? "fused-value-recurrence-stage" : "fused-recurrence-stage");
      const size_t stride = (value ? 30 : 27) * 2048;
      Guarded state(batch, 131072 * 4), aux(batch, 2 * stride * 4), first(batch, 57344 * 4),
          reference_state(decode, 131072 * 4), reference_aux(decode, 2 * stride * 4);
      std::vector<DeviceBuffer> args{state.data, aux.data};
      if (value) args.push_back(first.data);
      auto paired = batch.prepare(args);
      std::vector<DeviceRun> singles;
      for (size_t t = 0; t < 2; ++t) {
        args = {reference_state.data, reference_aux.data.slice(t * stride * 4, stride * 4)};
        if (value) args.push_back(first.data.slice(t * 55296 * 4, 8192));
        singles.push_back(decode.prepare(args));
      }
      V initial(131072), input(2 * stride), fv(57344), actual(131072), expected(131072),
          a(2 * stride), b(2 * stride), oracle(131072), saved, saved_aux;
      std::mt19937 rng(731);
      std::uniform_real_distribution<float> random(-.2f, .2f);
      for (auto &v : initial) v = random(rng);
      for (auto &v : input) v = random(rng);
      for (auto &v : fv) v = random(rng);
      first.data.upload(fv.data(), first.bytes);
      double worst = 0;
      for (int pass = 0; pass < 3; ++pass) {
        V start = pass == 1 ? saved : initial;
        for (auto *s : {&state, &reference_state}) s->data.upload(start.data(), s->bytes);
        for (auto *x : {&aux, &reference_aux}) x->data.upload(input.data(), x->bytes);
        paired.execute();
        for (auto &run : singles) run.execute();
        state.data.download(actual.data(), state.bytes);
        reference_state.data.download(expected.data(), reference_state.bytes);
        aux.data.download(a.data(), aux.bytes);
        reference_aux.data.download(b.data(), reference_aux.bytes);
        if (std::memcmp(actual.data(), expected.data(), state.bytes) ||
            std::memcmp(a.data(), b.data(), aux.bytes))
          throw std::runtime_error("Pair differs bitwise from sequential recurrence");
        if (pass == 0) { saved = actual; saved_aux = a; }
        if (pass == 2 && (actual != saved || a != saved_aux))
          throw std::runtime_error("Reset/replay changed output");
        // Independent FP64 state/y update, using the baseline's prepared
        // vectors. Preparation and finish are checked bitwise above.
        oracle = start;
        auto check = [&](float got, double ref) {
          const double error = std::abs(double(got) - ref);
          worst = std::max(worst, error);
          if (!std::isfinite(got) || error > 2e-5 + 2e-5 * std::abs(ref))
            throw std::runtime_error("FP64 state/y oracle mismatch");
        };
        for (size_t t = 0; t < 2; ++t)
          for (size_t h = 0; h < 32; ++h) {
            auto at = [&](size_t v, size_t j) { return b[t * stride + v * 2048 + h * 64 + j]; };
            for (size_t j = 0; j < 64; ++j) {
              double sa = 0, y = 0;
              for (size_t i = 0; i < 64; ++i)
                sa += double(oracle[h * 4096 + i * 64 + j]) * at(17, i);
              for (size_t i = 0; i < 64; ++i) {
                auto &s = oracle[h * 4096 + i * 64 + j];
                s = float(double(s) * at(14, i) + sa * at(18, i) + double(at(16, j)) * at(15, i));
                y += double(s) * at(13, i);
              }
              check(a[t * stride + 8 * 2048 + h * 64 + j], y);
            }
          }
        for (size_t i = 0; i < actual.size(); ++i) check(actual[i], oracle[i]);
      }
      for (auto *x : {&state, &aux, &first, &reference_state, &reference_aux}) x->guard();
      V immutable(57344);
      first.data.download(immutable.data(), first.bytes);
      if (immutable != fv) throw std::runtime_error("First-value input mutated");
      std::cout << "value=" << value << " bitwise/oracle/reset/guards passed max_abs=" << worst << '\n';
      for (int round = 0; round < 4; ++round) {
        const bool fused = round == 1 || round == 2;
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 200; ++i) {
          if (fused) paired.execute();
          else for (auto &run : singles) run.execute();
        }
        std::cout << (fused ? "batch" : "single_pair") << " mean_us="
                  << std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / 200 << '\n';
      }
    }
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
