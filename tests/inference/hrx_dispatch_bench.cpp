// SPDX-License-Identifier: Apache-2.0
// HRX dispatch control inspired by FastFlowLM's HRX host binary and public API.
// Uses unchanged upstream-norm artifacts. This is not a model speed benchmark.
#include "rwkv/xdna/session.hpp"
#include <hrx_amdxdna.h>
#include <hrx_runtime.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Vec = std::vector<float>;
constexpr size_t channels = 2048, bytes = channels * sizeof(float);
constexpr size_t guard = 256, total = bytes + 2 * guard;
constexpr int steps = 16, samples = 128;
void check(hrx_status_t status) {
  if (hrx_status_is_ok(status)) return;
  char *message = nullptr;
  size_t length = 0;
  hrx_status_to_string(status, &message, &length);
  std::string error = message ? message : "HRX error";
  hrx_status_free_message(message);
  hrx_status_ignore(status);
  throw std::runtime_error(error);
}
std::vector<uint8_t> read(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("Cannot open " + path.string());
  return {std::istreambuf_iterator<char>(file), {}};
}
void check_guards(const uint8_t *data) {
  for (size_t i = 0; i < guard; ++i)
    if (data[i] != 0x5a || data[guard + bytes + i] != 0x5a)
      throw std::runtime_error("Buffer guard changed");
}
struct HrxBuffer {
  hrx_buffer_t handle = nullptr;
  uint8_t *mapped = nullptr;
  explicit HrxBuffer(hrx_stream_t stream) {
    check(hrx_buffer_allocate(stream, total,
        HRX_MEMORY_TYPE_HOST_VISIBLE | HRX_MEMORY_TYPE_HOST_CACHED |
            HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_PERSISTENT, &handle));
    void *p = nullptr;
    auto status = hrx_buffer_map_with_mode(handle, HRX_MAPPING_MODE_PERSISTENT,
        HRX_MAP_READ | HRX_MAP_WRITE, 0, total, &p);
    if (!hrx_status_is_ok(status)) {
      hrx_buffer_release(handle);
      handle = nullptr;
      check(status);
    }
    mapped = static_cast<uint8_t *>(p);
    std::memset(mapped, 0x5a, total);
    check(hrx_buffer_flush_range(handle, 0, total));
  }
  ~HrxBuffer() { if (handle) hrx_buffer_release(handle); }
  HrxBuffer(const HrxBuffer &) = delete;
  void upload(const Vec &v) {
    std::memcpy(mapped + guard, v.data(), bytes);
    check(hrx_buffer_flush_range(handle, 0, total));
  }
  Vec download() {
    check(hrx_buffer_invalidate_range(handle, 0, total));
    check_guards(mapped);
    Vec v(channels);
    std::memcpy(v.data(), mapped + guard, bytes);
    return v;
  }
  hrx_buffer_ref_t ref() { return {handle, guard, bytes}; }
};
struct HrxProgram {
  hrx_device_t device = nullptr;
  hrx_stream_t stream = nullptr;
  hrx_executable_t executable = nullptr;
  uint32_t ordinal = 0;
  explicit HrxProgram(const std::filesystem::path &path) {
    // Same CREATE + transaction interface used in FastFlowLM's HRX library.
    check(hrx_gpu_initialize(0));
    check(hrx_gpu_device_get(0, &device));
    auto xc = read(path / "design.xclbin"), txn = read(path / "instructions.bin");
    hrx_const_byte_span_t bin = {xc.data(), xc.size()};
    auto run = hrx_amdxdna_executable_run_default();
    run.transaction = {txn.data(), txn.size()};
    auto ep = hrx_amdxdna_executable_entry_point_default();
    ep.name = {"norm", 4};
    ep.runs = &run;
    ep.run_count = 1;
    auto params = hrx_amdxdna_executable_create_params_default();
    params.xclbins = &bin;
    params.xclbin_count = 1;
    params.entry_points = &ep;
    params.entry_point_count = 1;
    check(hrx_amdxdna_executable_create(device, &params, &executable));
    try {
      check(hrx_executable_lookup_export_by_name(executable, "norm", &ordinal));
      check(hrx_stream_create(device, 0, &stream));
    } catch (...) { hrx_executable_release(executable); throw; }
  }
  ~HrxProgram() {
    if (stream) hrx_stream_release(stream);
    if (executable) hrx_executable_release(executable);
  }
  void record(const std::array<hrx_buffer_ref_t, 4> &bindings) {
    const hrx_dispatch_config_t config = {{1, 1, 1}, {1, 1, 1}, 0};
    check(hrx_stream_dispatch(stream, executable, ordinal, &config, nullptr, 0,
        bindings.data(), bindings.size(), HRX_DISPATCH_FLAG_NONE));
  }
  void finish() { check(hrx_stream_flush(stream)); check(hrx_stream_wait(stream)); }
};
struct XrtBuffer {
  rwkv::xdna::DeviceBuffer root, data;
  explicit XrtBuffer(rwkv::xdna::Session &session)
      : root(session.allocate(total)), data(root.slice(guard, bytes)) {
    std::vector<uint8_t> poison(total, 0x5a);
    root.upload(poison.data(), total);
  }
  void upload(const Vec &v) { data.upload(v.data(), bytes); }
  Vec download() {
    std::vector<uint8_t> all(total);
    root.download(all.data(), total);
    check_guards(all.data());
    Vec v(channels);
    std::memcpy(v.data(), all.data() + guard, bytes);
    return v;
  }
};
Vec reference(Vec x, const Vec &gamma, const Vec &beta) {
  {
    double mean = 0, variance = 0;
    for (float v : x) mean += v / double(channels);
    for (float v : x) variance += (v - mean) * (v - mean) / channels;
    for (size_t i = 0; i < channels; ++i)
      x[i] = float((x[i] - mean) / std::sqrt(variance + 1e-5) * gamma[i] + beta[i]);
  }
  return x;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("Usage: rwkv-hrx-dispatch-bench NORM_ARTIFACT_DIR");
    const std::filesystem::path path(argv[1]);
    HrxProgram hrx(path);
    HrxBuffer a(hrx.stream), b(hrx.stream), g(hrx.stream), bt(hrx.stream);
    rwkv::xdna::Session xrt(path / "design.xclbin", path / "instructions.bin");
    XrtBuffer xa(xrt), xb(xrt), xg(xrt), xbt(xrt);
    auto even = xrt.prepare({xa.data, xg.data, xbt.data, xb.data});
    auto odd = xrt.prepare({xb.data, xg.data, xbt.data, xa.data});
    Vec gamma(channels), beta(channels);
    for (size_t i = 0; i < channels; ++i) {
      gamma[i] = .9f + float(i % 17) * .005f;
      beta[i] = float(int(i % 13) - 6) * .002f;
    }
    g.upload(gamma); bt.upload(beta); xg.upload(gamma); xbt.upload(beta);
    std::vector<Vec> fixtures;
    for (int fixture = 0; fixture < 4; ++fixture) {
      Vec input(channels);
      for (size_t i = 0; i < channels; ++i)
        input[i] = fixture == 0 ? .125f :
            std::sin(float(i) * (.123f + .017f * fixture)) *
            (fixture == 1 ? .00001f : .2f) + .125f;
      fixtures.push_back(input);
    }
    std::array<Vec, 4> xrt_results;
    double max_abs_fp64 = 0;
    for (size_t f = 0; f < fixtures.size(); ++f) {
      xa.upload(fixtures[f]);
      auto current = fixtures[f];
      // Validate each transition against FP64 using its actual FP32 input.
      // Repeated near-constant normalization amplifies earlier rounding errors;
      // an independently evolved FP64 trajectory tests a different contract.
      for (int i = 0; i < steps; ++i) {
        auto expected = reference(current, gamma, beta);
        (i % 2 ? odd : even).execute();
        current = (i % 2 ? xa : xb).download();
        for (size_t j = 0; j < channels; ++j) {
          const double error = std::abs(double(current[j]) - expected[j]);
          max_abs_fp64 = std::max(max_abs_fp64, error);
          if (!std::isfinite(current[j]) || error > 2e-5)
            throw std::runtime_error("Per-step FP64 reference mismatch");
        }
      }
      xrt_results[f] = current;
    }
    for (int trial = 0; trial < 4; ++trial) {
      // Reverse order every other trial; each backend gets four warm samples.
      for (int index = 0; index < 3; ++index) {
        const int mode = trial % 2 ? 2 - index : index;
        double elapsed = 0;
        for (int sample = -4; sample < samples; ++sample) {
          const size_t f = size_t(sample + 4) % fixtures.size();
          if (mode == 0) xa.upload(fixtures[f]); else a.upload(fixtures[f]);
          auto start = Clock::now();
          for (int step = 0; step < steps; ++step) {
            if (mode == 0) (step % 2 ? odd : even).execute();
            else {
              hrx.record({(step % 2 ? b : a).ref(), g.ref(), bt.ref(),
                          (step % 2 ? a : b).ref()});
              if (mode == 1 || step == steps - 1) hrx.finish();
            }
          }
          const double us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
          if (sample >= 0) elapsed += us;
          auto result = mode == 0 ? xa.download() : a.download();
          if (result != xrt_results[f]) throw std::runtime_error("XRT/HRX mismatch");
          if (mode == 0) {
            xb.download();
            if (xg.download() != gamma || xbt.download() != beta)
              throw std::runtime_error("XRT immutable buffer changed");
          } else {
            b.download();
            if (g.download() != gamma || bt.download() != beta)
              throw std::runtime_error("HRX immutable buffer changed");
          }
        }
        std::cout << nlohmann::json({{"trial", trial},
            {"mode", mode == 0 ? "xrt_individual" : mode == 1 ? "hrx_individual" : "hrx_batch16"},
            {"chain16_us", elapsed / samples}, {"samples", samples},
            {"max_abs_fp64_per_step", max_abs_fp64}, {"bitwise_xrt", true}, {"guards", "pass"}}).dump() << '\n';
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
