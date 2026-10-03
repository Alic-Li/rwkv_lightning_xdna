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
static void reference(V &state, V &aux) {
  auto sigmoid = [](double x) { return 1 / (1 + std::exp(-x)); };
  for (size_t h = 0; h < 32; ++h) {
    auto at = [&](size_t v, size_t j) -> float & {
      return aux[v * 2048 + h * 64 + j];
    };
    double ss = 0;
    for (size_t j = 0; j < 64; ++j) {
      float u = at(0, j) * at(3, j);
      ss += double(u) * u;
    }
    float den = std::max(float(std::sqrt(ss)), 1e-12f);
    for (size_t j = 0; j < 64; ++j) {
      float kk = at(0, j) * at(3, j) / den,
            a = float(sigmoid(at(1, j) + at(5, j)));
      at(19, j) = kk;
      at(20, j) = a;
      at(17, j) = -kk;
      at(18, j) = kk * a;
      at(15, j) = at(0, j) * (1 + (a - 1) * at(4, j));
      at(14, j) = float(
          std::exp(-0.6065306597126334f * float(sigmoid(at(2, j) + at(6, j)))));
    }
    for (size_t j = 0; j < 64; ++j) {
      double sa = 0, y = 0;
      for (size_t i = 0; i < 64; ++i)
        sa += double(at(17, i)) * state[h * 4096 + i * 64 + j];
      for (size_t i = 0; i < 64; ++i) {
        auto &s = state[h * 4096 + i * 64 + j];
        s = float(double(s) * at(14, i) + double(at(18, i)) * sa +
                  double(at(15, i)) * at(16, j));
        y += double(at(13, i)) * s;
      }
      at(8, j) = float(y);
    }
    double mean = 0, var = 0, dot = 0;
    for (size_t j = 0; j < 64; ++j)
      mean += at(8, j);
    mean /= 64;
    for (size_t j = 0; j < 64; ++j) {
      double d = at(8, j) - mean;
      var += d * d;
      dot += double(at(13, j)) * at(15, j) * at(11, j);
    }
    double scale = 1 / std::sqrt(var / 64 + 64e-5f);
    for (size_t j = 0; j < 64; ++j) {
      at(21, j) = float((at(8, j) - mean) * scale) * at(9, j) + at(10, j);
      at(22, j) = float(dot * at(16, j));
      at(23, j) = at(21, j) + at(22, j);
      at(24, j) = at(23, j) * at(12, j);
    }
  }
}
int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 5)
      throw std::runtime_error("usage: recurrence-stage-test KERNEL_ROOT [TIMING_ITERATIONS [TRACE_FILE|--alternate-value|--fused-value [TRACE_FILE]|--fused-value-alternate]]");
    size_t iterations = 0;
    if (argc >= 3) {
      const std::string value(argv[2]);
      if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid timing iterations");
      iterations = std::stoull(value);
      if (!iterations || iterations > 1000000)
        throw std::runtime_error("Invalid timing iterations");
    }
    const bool alternate = argc >= 4 && (std::string(argv[3]) == "--alternate-value" || std::string(argv[3]) == "--fused-value-alternate");
    const bool fused_value = argc >= 4 && (std::string(argv[3]) == "--fused-value" || std::string(argv[3]) == "--fused-value-alternate");
    if (argc == 5 && (!fused_value || alternate))
      throw std::runtime_error("Extra argument requires --fused-value TRACE_FILE");
    const bool wants_trace = (argc == 4 && !alternate && !fused_value) || argc == 5;
    auto session = [](std::filesystem::path root, const char *name) {
      return Session(root / name / "design.xclbin",
                     root / name / "instructions.bin");
    };
    const auto stage_name = fused_value ? "fused-value-recurrence-stage" : "fused-recurrence-stage";
    auto fused = session(argv[1], stage_name);
    const size_t arena_floats = (fused_value ? 30 : 27) * 2048;
    Guarded fs(fused, 131072 * 4), fa(fused, arena_floats * 4);
    nlohmann::json config;
    std::ifstream metadata(std::filesystem::path(argv[1]) / stage_name / "config.json");
    metadata >> config;
    size_t trace_bytes = config.value("trace_buffer_bytes", size_t(0));
    if (bool(trace_bytes) != wants_trace)
      throw std::runtime_error("Trace artifact and TRACE_FILE must be supplied together");
    std::vector<DeviceBuffer> arguments{fs.data, fa.data};
    std::unique_ptr<Guarded> trace;
    if (trace_bytes) {
      if (trace_bytes > 64 * 1024 * 1024 || trace_bytes % 4)
        throw std::runtime_error("Invalid trace buffer size");
      trace = std::make_unique<Guarded>(fused, trace_bytes);
      std::vector<uint32_t> empty(trace_bytes / 4);
      trace->data.upload(empty.data(), trace_bytes);
    }
    std::unique_ptr<Guarded> first_input;
    V value_data(6144), first_data(2048);
    if (fused_value) {
      if (!config.value("fused_value", false) || config.at("lanes") != 7 || config.at("arena_vectors") != 30)
        throw std::runtime_error("Incompatible fused value artifact");
      first_input = std::make_unique<Guarded>(fused, 2048 * 4);
      arguments.push_back(first_input->data);
    }
    if (trace) arguments.push_back(trace->data);
    auto run = fused.prepare(arguments);
    V state(131072), aux(arena_floats), ref, expected, actual_s(state.size()),
        actual_a(aux.size());
    std::mt19937 rng(20261003);
    std::uniform_real_distribution<float> d(-.125f, .125f);
    for (auto &f : state)
      f = d(rng);
    for (auto &f : aux)
      f = d(rng);
    fs.data.upload(state.data(), fs.bytes);
    double worst = 0;
    for (int pass = 0; pass < 3; ++pass) {
      for (int v : {0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16})
        for (size_t i = 0; i < 2048; ++i)
          aux[v * 2048 + i] = (v == 9 ? 1.f : 0.f) + d(rng) * (pass + 1);
      if (fused_value) {
        for (auto &x : value_data) x = d(rng) * (pass + 1);
        for (auto &x : first_data) x = d(rng) * (pass + 1);
        std::copy(value_data.begin(), value_data.end(), aux.begin() + 27 * 2048);
        first_input->data.upload(first_data.data(), first_input->bytes);
      }
      expected = aux;
      if (fused_value)
        for (size_t j = 0; j < 2048; ++j) {
          const float a = float(1 / (1 + std::exp(-double(value_data[2048+j] + value_data[4096+j]))));
          expected[16*2048+j] = value_data[j] + (first_data[j] - value_data[j]) * a;
        }
      ref = state;
      reference(ref, expected);
      fa.data.upload(aux.data(), fa.bytes);
      run.execute();
      fs.data.download(actual_s.data(), fs.bytes);
      fa.data.download(actual_a.data(), fa.bytes);
      auto check = [&](const V &a, const V &b) {
        for (size_t i = 0; i < a.size(); ++i) {
          double e = std::abs(double(a[i]) - b[i]);
          worst = std::max(worst, e);
          if (!std::isfinite(a[i]) || e > 2e-5 + 2e-5 * std::abs(b[i]))
            throw std::runtime_error(
                "independent oracle failed index=" + std::to_string(i) +
                " error=" + std::to_string(e));
        }
      };
      check(actual_s, ref);
      check(actual_a, expected);
      if (fused_value && !std::equal(value_data.begin(), value_data.end(),
                                    actual_a.begin() + 27 * 2048))
        throw std::runtime_error("Fused value arena inputs overwritten");
      state = actual_s;
      aux = actual_a;
    }
    fs.guard();
    fa.guard();
    if (fused_value) {
      first_input->guard();
      V same_first(2048);
      first_input->data.download(same_first.data(), first_input->bytes);
      if (same_first != first_data)
        throw std::runtime_error("Fused value input overwritten");
    }
    std::cout << "recurrence oracle/replay/guards passed max_abs=" << worst
              << '\n';
    if (iterations) {
      std::unique_ptr<Session> value_session;
      std::unique_ptr<Guarded> va, vf, vo;
      std::unique_ptr<DeviceRun> value_run;
      if (alternate) {
        const auto root = std::filesystem::path(argv[1]) / "fused-value";
        value_session = std::make_unique<Session>(root / "design.xclbin", root / "instructions.bin");
        va = std::make_unique<Guarded>(*value_session, 6144 * 4);
        vf = std::make_unique<Guarded>(*value_session, 2048 * 4);
        vo = std::make_unique<Guarded>(*value_session, 2048 * 4);
        V params(6144), first(2048, .2f), result(2048);
        std::fill_n(params.begin(), 2048, .1f);
        va->data.upload(params.data(), va->bytes);
        vf->data.upload(first.data(), vf->bytes);
        value_run = std::make_unique<DeviceRun>(value_session->prepare(
            std::vector<DeviceBuffer>{va->data, vf->data, vo->data}));
        value_run->execute();
        vo->data.download(result.data(), vo->bytes);
        for (float x : result)
          if (!std::isfinite(x) || std::abs(x - .15f) > 2e-5f)
            throw std::runtime_error("Alternating value oracle failed");
      }
      std::vector<double> times;
      double submit_us = 0, wait_us = 0, pair_us = 0;
      // Keep fixed recurrent state and input across samples. Reset is outside
      // the timer; all BOs, instructions and the XRT run are reused.
      for (size_t i = 0; i < iterations + 4; ++i) {
        fs.data.upload(state.data(), fs.bytes);
        fa.data.upload(aux.data(), fa.bytes);
        auto pair_start = std::chrono::steady_clock::now();
        if (value_run) value_run->execute();
        auto start = std::chrono::steady_clock::now();
        RunTiming timing;
        run.execute(30000, &timing);
        auto elapsed = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
        if (i >= 4) {
          times.push_back(elapsed);
          submit_us += timing.submit_us;
          wait_us += timing.wait_us;
          pair_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - pair_start).count();
        }
      }
      std::sort(times.begin(), times.end());
      double total = 0;
      for (double t : times) total += t;
      if (va) {
        va->guard(); vf->guard(); vo->guard();
      }
      std::cout << "recurrence_timing samples=" << iterations
                << " fused_value=" << fused_value
                << " alternate_value=" << alternate
                << " mean_submit_us=" << submit_us / iterations
                << " mean_wait_us=" << wait_us / iterations
                << " mean_pair_us=" << pair_us / iterations
                << " mean_us=" << total / iterations
                << " p50_us=" << times[(iterations - 1) / 2]
                << " p95_us=" << times[size_t(std::ceil(iterations * .95)) - 1]
                << '\n';
      fs.guard();
      fa.guard();
      if (first_input) first_input->guard();
    }
    if (trace) {
      // Record exactly one dispatch, with a cleared buffer, after validation.
      std::vector<uint32_t> words(trace_bytes / 4);
      trace->data.upload(words.data(), trace_bytes);
      run.execute();
      trace->data.download(words.data(), trace_bytes);
      trace->guard();
      fs.guard();
      fa.guard();
      if (first_input) first_input->guard();
      if (words.back())
        throw std::runtime_error("Trace buffer filled; increase RWKV_XDNA_TRACE_BYTES");
      while (!words.empty() && words.back() == 0) words.pop_back();
      if (words.empty()) throw std::runtime_error("Empty hardware trace");
      std::ofstream output(argv[argc - 1]);
      for (auto word : words)
        output << std::hex << std::setfill('0') << std::setw(8) << word << '\n';
      if (!output) throw std::runtime_error("Cannot write hardware trace");
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
