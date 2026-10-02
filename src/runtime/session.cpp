// SPDX-License-Identifier: Apache-2.0
#include "rwkv/xdna/session.hpp"
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#if __has_include(<xrt/xrt_xclbin.h>)
#include <xrt/xrt_xclbin.h>
#else
#include <xrt/experimental/xrt_xclbin.h>
#endif
#include <chrono>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace rwkv::xdna {
namespace {
std::vector<uint32_t> read_instructions(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("Cannot open instructions: " + path.string());
  auto size = stream.tellg();
  if (size <= 0 || size % 4 ||
      static_cast<uint64_t>(size) / 4 > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("Invalid instruction stream length");
  std::vector<uint32_t> data(static_cast<size_t>(size) / 4);
  stream.seekg(0);
  if (!stream.read(reinterpret_cast<char *>(data.data()), size))
    throw std::runtime_error("Truncated instruction stream");
  return data;
}
using Clock = std::chrono::steady_clock;
double micros(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}
} // namespace

struct Session::Impl {
  xrt::device device;
  xrt::hw_context context;
  xrt::kernel kernel;
  std::vector<uint32_t> instructions;
  xrt::bo instruction_buffer;
  std::vector<xrt::bo> buffers;
  std::vector<size_t> sizes;

  Impl(const std::filesystem::path &binary, const std::filesystem::path &inst,
       const std::string &name, unsigned index)
      : device(index),
        context(device, device.register_xclbin(xrt::xclbin(binary.string()))),
        kernel(context, name), instructions(read_instructions(inst)),
        instruction_buffer(device, instructions.size() * 4,
                           XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1)) {
    instruction_buffer.write(instructions.data());
    instruction_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
};

Session::Session(const std::filesystem::path &binary,
                 const std::filesystem::path &inst, const std::string &name,
                 unsigned index)
    : impl_(std::make_unique<Impl>(binary, inst, name, index)) {}
Session::~Session() = default;

Timing Session::execute(std::vector<Buffer> &buffers, unsigned timeout_ms) {
  if (buffers.empty() || buffers.size() > 32 || timeout_ms == 0)
    throw std::invalid_argument("Invalid buffer count or timeout");
  auto &s = *impl_;
  std::vector<size_t> sizes;
  for (const auto &buffer : buffers) {
    if (buffer.bytes.empty())
      throw std::invalid_argument("Empty buffer");
    sizes.push_back(buffer.bytes.size());
  }
  // Reuse device allocations on repeated calls with the same buffer sizes.
  if (sizes != s.sizes) {
    std::vector<xrt::bo> allocations;
    for (size_t i = 0; i < sizes.size(); ++i)
      allocations.emplace_back(s.device, sizes[i], XRT_BO_FLAGS_HOST_ONLY,
                               s.kernel.group_id(static_cast<int>(3 + i)));
    s.buffers = std::move(allocations);
    s.sizes = sizes;
  }
  auto total_start = Clock::now();
  for (size_t i = 0; i < buffers.size(); ++i) {
    s.buffers[i].write(buffers[i].bytes.data());
    s.buffers[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  xrt::run run(s.kernel);
  run.set_arg(0, 3u);
  run.set_arg(1, s.instruction_buffer);
  run.set_arg(2, static_cast<uint32_t>(s.instructions.size()));
  for (size_t i = 0; i < buffers.size(); ++i)
    run.set_arg(static_cast<int>(3 + i), s.buffers[i]);
  auto start = Clock::now();
  run.start();
  const auto state = run.wait(timeout_ms);
  if (state != ERT_CMD_STATE_COMPLETED)
    throw std::runtime_error("NPU dispatch failed: ERT state=" +
                             std::to_string(static_cast<int>(state)));
  auto end = Clock::now();
  for (size_t i = 0; i < buffers.size(); ++i)
    if (buffers[i].output) {
      s.buffers[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      s.buffers[i].read(buffers[i].bytes.data());
    }
  return {micros(start, end), micros(total_start, Clock::now())};
}
} // namespace rwkv::xdna
