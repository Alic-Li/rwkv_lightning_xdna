// SPDX-License-Identifier: Apache-2.0
#include "device_test_utils.hpp"
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
using namespace rwkv::xdna;
using namespace rwkv::xdna::test;
using V = std::vector<float>;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
int main(int argc, char **argv) {
  try {
    if (argc != 4)
      throw std::runtime_error(
          "Usage: mode-worker-test CANDIDATE_ROOT BASELINE_ROOT SHARED_ROOT");
    auto session = [](const char *root, const char *name) {
      auto p = std::filesystem::path(root) / name;
      return Session(p / "design.xclbin", p / "instructions.bin");
    };
    auto p = session(argv[1], "project"), n = session(argv[1], "norm"),
         bp = session(argv[2], "bf16-projection-residual"),
         bn = session(argv[2], "upstream-norm"),
         third = session(argv[2], "fused-norm-mix-6"),
         sp = session(argv[3], "bf16-projection-residual"),
         sn = session(argv[3], "upstream-norm");
    Guarded xp(p, 6160 * 4), xn(n, 6160 * 4), w(p, 8388608), residual(p, 8192),
        zeros(p, 8192), yp(p, 16384), yn(n, 16384), rp(bp, 16384), rn(bn, 8192),
        params(third, 16384 * 4), state(third, 8192), pair(third, 16384),
        mixed(third, 12288 * 4);
    std::mt19937 rng(149);
    std::uniform_real_distribution<float> random(-.25f, .25f);
    V input(6160, 0), r(2048), zero(2048, 0), np(16384, 0);
    for (size_t i = 0; i < 2048; ++i) {
      input[i] = random(rng);
      input[2048 + i] = .75f + random(rng);
      input[4096 + i] = random(rng);
      r[i] = random(rng);
      np[i] = input[2048 + i];
      np[2048 + i] = input[4096 + i];
    }
    xn.data.upload(input.data(), xn.bytes);
    input[6144] = 1;
    xp.data.upload(input.data(), xp.bytes);
    residual.data.upload(r.data(), residual.bytes);
    zeros.data.upload(zero.data(), zeros.bytes);
    params.data.upload(np.data(), params.bytes);
    state.data.upload(zero.data(), state.bytes);
    std::vector<uint16_t> weights(4194304);
    for (auto &v : weights)
      v = bf16(random(rng));
    w.data.upload(weights.data(), w.bytes);
    auto pr = p.prepare({xp.data, w.data, residual.data, yp.data});
    auto nr = n.prepare({xn.data, w.data, zeros.data, yn.data});
    auto bpr =
        bp.prepare({xp.data.slice(0, 8192), w.data, residual.data, rp.data});
    auto bnr = bn.prepare({xn.data.slice(0, 8192), xn.data.slice(8192, 8192),
                           xn.data.slice(16384, 8192), rn.data});
    auto spr =
        sp.prepare({xp.data.slice(0, 8192), w.data, residual.data, rp.data});
    auto snr = sn.prepare({xn.data.slice(0, 8192), xn.data.slice(8192, 8192),
                           xn.data.slice(16384, 8192), rn.data});
    auto tr = third.prepare({xn.data.slice(0, 8192), params.data, state.data,
                             pair.data, mixed.data});
    bpr.execute();
    bnr.execute();
    V expected_p(4096), expected_n(2048), got(4096);
    rp.data.download(expected_p.data(), rp.bytes);
    rn.data.download(expected_n.data(), rn.bytes);
    double worst = 0;
    for (size_t row = 0; row < 2048; ++row) {
      double sum = 0;
      for (size_t col = 0; col < 2048; ++col) {
        size_t pos =
            ((row / 16) * 8 + col / 256) * 4096 + row % 16 * 256 + col % 256;
        sum += double(expand(weights[pos])) * expand(bf16(input[col]));
      }
      for (size_t part = 0; part < 2; ++part) {
        double ref = sum + (part ? r[row] : 0),
               err = std::abs(expected_p[part * 2048 + row] - ref);
        worst = std::max(worst, err);
        if (err > 2e-5 + 2e-5 * std::abs(ref))
          throw std::runtime_error("FP64 projection mismatch");
      }
    }
    double mean =
               std::accumulate(input.begin(), input.begin() + 2048, 0.) / 2048,
           variance = 0;
    for (size_t i = 0; i < 2048; ++i)
      variance += (input[i] - mean) * (input[i] - mean) / 2048;
    for (size_t i = 0; i < 2048; ++i)
      if (std::abs(expected_n[i] -
                   ((input[i] - mean) / std::sqrt(variance + 1e-5) *
                        input[2048 + i] +
                    input[4096 + i])) > 2e-5)
        throw std::runtime_error("FP64 norm mismatch");
    auto check = [&]() {
      yp.data.download(got.data(), yp.bytes);
      if (got != expected_p)
        throw std::runtime_error("Mode projection differs");
      yn.data.download(got.data(), yn.bytes);
      for (size_t i = 0; i < 4096; ++i)
        if (got[i] != expected_n[i % 2048])
          throw std::runtime_error("Mode norm differs");
    };
    // Exercise unequal runs of each opcode before alternating; stale FIFO state
    // must not depend on a fixed norm/project cadence.
    for (int i = 0; i < 3; ++i)
      nr.execute();
    for (int i = 0; i < 2; ++i)
      pr.execute();
    check();
    for (bool enter : {false, true})
      for (int arm : {0, 1, 1, 0, 0, 2, 2, 0}) {
        auto &norm_run = arm == 1 ? nr : (arm == 2 ? snr : bnr);
        auto &project_run = arm == 1 ? pr : (arm == 2 ? spr : bpr);
        for (int i = 0; i < 16; ++i) {
          if (enter)
            tr.execute();
          norm_run.execute();
          project_run.execute();
        }
        double norm_us = 0, project_us = 0;
        for (int i = 0; i < 256; ++i) {
          if (enter)
            tr.execute();
          auto start = Clock::now();
          norm_run.execute();
          auto mid = Clock::now();
          project_run.execute();
          auto end = Clock::now();
          norm_us +=
              std::chrono::duration<double, std::micro>(mid - start).count();
          project_us +=
              std::chrono::duration<double, std::micro>(end - mid).count();
        }
        check();
        if (arm != 1) {
          rp.data.download(got.data(), rp.bytes);
          if (got != expected_p)
            throw std::runtime_error("Baseline/shared projection differs");
          V v(2048);
          rn.data.download(v.data(), rn.bytes);
          if (v != expected_n)
            throw std::runtime_error("Baseline/shared norm differs");
        }
        std::cout << Json({{"variant",
                            arm == 1 ? "reused_workers"
                                     : (arm == 2 ? "independent_shared_workers"
                                                 : "separate_programs")},
                           {"entry_after_third_program", enter},
                           {"samples", 256},
                           {"norm_mean_us", norm_us / 256},
                           {"projection_mean_us", project_us / 256},
                           {"pair_mean_us", (norm_us + project_us) / 256}})
                         .dump()
                  << std::endl;
      }
    V saved(2048);
    state.data.download(saved.data(), state.bytes);
    if (saved != expected_n)
      throw std::runtime_error("Third stage state mismatch");
    V mixes(12288);
    mixed.data.download(mixes.data(), mixed.bytes);
    for (size_t i = 0; i < mixes.size(); ++i)
      if (mixes[i] != expected_n[i % 2048])
        throw std::runtime_error("Third stage output mismatch");
    std::vector<uint16_t> saved_w(weights.size());
    w.data.download(saved_w.data(), w.bytes);
    if (saved_w != weights)
      throw std::runtime_error("Weights mutated");
    for (int fixture = 0; fixture < 3; ++fixture) {
      for (size_t i = 0; i < 2048; ++i) {
        input[i] = fixture == 0 ? .125f
                                : (fixture == 1 ? random(rng) * 32
                                                : .125f + random(rng) * 1e-4f);
        input[2048 + i] = .75f + random(rng);
        input[4096 + i] = random(rng);
      }
      input[6144] = 0;
      xn.data.upload(input.data(), xn.bytes);
      input[6144] = 1;
      xp.data.upload(input.data(), xp.bytes);
      bpr.execute();
      bnr.execute();
      rp.data.download(expected_p.data(), rp.bytes);
      rn.data.download(expected_n.data(), rn.bytes);
      for (int i = 0; i < fixture + 1; ++i)
        nr.execute();
      for (int i = 0; i < 4 - fixture; ++i)
        pr.execute();
      check();
    }
    auto immutable = [](Guarded &b, const V &ref) {
      V v(ref.size());
      b.data.download(v.data(), b.bytes);
      if (v != ref)
        throw std::runtime_error("Input mutated");
    };
    immutable(xp, input);
    input[6144] = 0;
    immutable(xn, input);
    immutable(residual, r);
    immutable(zeros, zero);
    immutable(params, np);
    for (auto *b : {&xp, &xn, &w, &residual, &zeros, &yp, &yn, &rp, &rn,
                    &params, &state, &pair, &mixed})
      b->guard();
    std::cout << Json({{"status", "passed"},
                       {"fp64_projection_max_abs", worst},
                       {"exact_modes_replay_immutable_guards", "passed"}})
                     .dump()
              << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
