// SPDX-License-Identifier: Apache-2.0
// HRX dispatch control inspired by FastFlowLM's HRX host binary and public API.
// Shared support for optional HRX/XRT dispatch controls.
#pragma once
#include "rwkv/xdna/session.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <hrx_amdxdna.h>
#include <hrx_runtime.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

namespace hrx_test {
using Clock = std::chrono::steady_clock;
using Vec = std::vector<float>;
constexpr size_t channels = 2048, bytes = channels * sizeof(float);
constexpr size_t guard = 256, total = bytes + 2 * guard;
inline void check(hrx_status_t status) {
  if (hrx_status_is_ok(status))
    return;
  char *message = nullptr;
  size_t length = 0;
  hrx_status_to_string(status, &message, &length);
  std::string error = message ? message : "HRX error";
  hrx_status_free_message(message);
  hrx_status_ignore(status);
  throw std::runtime_error(error);
}
inline std::vector<uint8_t> read(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot open " + path.string());
  return {std::istreambuf_iterator<char>(file), {}};
}
inline void check_guards(const uint8_t *data, size_t size) {
  for (size_t i = 0; i < guard; ++i)
    if (data[i] != 0x5a || data[guard + size + i] != 0x5a)
      throw std::runtime_error("Buffer guard changed");
}
struct HrxBuffer {
  hrx_buffer_t handle = nullptr;
  uint8_t *mapped = nullptr;
  size_t bytes, total;
  explicit HrxBuffer(hrx_stream_t stream, size_t size = hrx_test::bytes)
      : bytes(size), total(size + 2 * guard) {
    check(hrx_buffer_allocate(
        stream, total,
        HRX_MEMORY_TYPE_HOST_VISIBLE | HRX_MEMORY_TYPE_HOST_CACHED |
            HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
        &handle));
    void *p = nullptr;
    auto status =
        hrx_buffer_map_with_mode(handle, HRX_MAPPING_MODE_PERSISTENT,
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
  ~HrxBuffer() {
    if (handle)
      hrx_buffer_release(handle);
  }
  HrxBuffer(const HrxBuffer &) = delete;
  void upload(const Vec &v) {
    std::memcpy(mapped + guard, v.data(), bytes);
    check(hrx_buffer_flush_range(handle, 0, total));
  }
  Vec download() {
    check(hrx_buffer_invalidate_range(handle, 0, total));
    check_guards(mapped, bytes);
    Vec v(bytes / 4);
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
    auto init = hrx_gpu_initialize(0);
    if (hrx_status_code(init) == HRX_STATUS_ALREADY_EXISTS)
      hrx_status_ignore(init);
    else
      check(init);
    check(hrx_gpu_device_get(0, &device));
    auto xc = read(path / "design.xclbin"),
         txn = read(path / "instructions.bin");
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
    } catch (...) {
      hrx_executable_release(executable);
      throw;
    }
  }
  ~HrxProgram() {
    if (stream)
      hrx_stream_release(stream);
    if (executable)
      hrx_executable_release(executable);
  }
  void record(const std::array<hrx_buffer_ref_t, 4> &bindings,
              hrx_stream_t target = nullptr) {
    const hrx_dispatch_config_t config = {{1, 1, 1}, {1, 1, 1}, 0};
    check(hrx_stream_dispatch(target ? target : stream, executable, ordinal,
                              &config, nullptr, 0, bindings.data(),
                              bindings.size(), HRX_DISPATCH_FLAG_NONE));
  }
  void finish() {
    check(hrx_stream_flush(stream));
    check(hrx_stream_wait(stream));
  }
};
struct XrtBuffer {
  rwkv::xdna::DeviceBuffer root, data;
  size_t bytes, total;
  explicit XrtBuffer(rwkv::xdna::Session &session,
                     size_t size = hrx_test::bytes)
      : root(session.allocate(size + 2 * guard)), data(root.slice(guard, size)),
        bytes(size), total(size + 2 * guard) {
    std::vector<uint8_t> poison(total, 0x5a);
    root.upload(poison.data(), total);
  }
  void upload(const Vec &v) { data.upload(v.data(), bytes); }
  Vec download() {
    std::vector<uint8_t> all(total);
    root.download(all.data(), total);
    check_guards(all.data(), bytes);
    Vec v(bytes / 4);
    std::memcpy(v.data(), all.data() + guard, bytes);
    return v;
  }
};
inline Vec reference(Vec x, const Vec &gamma, const Vec &beta) {
  {
    double mean = 0, variance = 0;
    for (float v : x)
      mean += v / double(channels);
    for (float v : x)
      variance += (v - mean) * (v - mean) / channels;
    for (size_t i = 0; i < channels; ++i)
      x[i] = float((x[i] - mean) / std::sqrt(variance + 1e-5) * gamma[i] +
                   beta[i]);
  }
  return x;
}
} // namespace hrx_test
