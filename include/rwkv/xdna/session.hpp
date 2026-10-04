// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rwkv::xdna {
// Process-wide high watermark of submitted DeviceRun commands awaiting completion.
size_t dispatch_concurrency_peak();

struct Buffer {
  std::vector<uint8_t> bytes;
  bool output = false;
};
struct Timing {
  double dispatch_us = 0;
  double transfer_and_dispatch_us = 0;
};
// Host wall times. wait_us includes device execution, scheduling and DMA;
// it must not be reported as pure NPU compute time.
struct RunTiming {
  double submit_us = 0;
  double wait_us = 0;
};

// Persistent NPU-accessible allocation. Host access is explicit; slicing does
// not copy or synchronize data. HOST_ONLY on XDNA is shared system DDR.
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer slice(size_t offset, size_t bytes) const;
  void upload(const void *data, size_t bytes, size_t offset = 0);
  void download(void *data, size_t bytes, size_t offset = 0) const;
  size_t size() const;
  // Byte offset when this view is wholly contained in another view of the same
  // allocation. No device access, copies, or XRT handles escape this query.
  std::optional<size_t> offset_within(const DeviceBuffer &) const;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  explicit DeviceBuffer(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
  friend class Session;
};
// Prepared runs retain their Session and BOs. Copies share one run; serialize
// execution and host writes to all buffers bound to it.
class DeviceRun {
public:
  void execute(unsigned timeout_ms = 30000, RunTiming *timing = nullptr);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  explicit DeviceRun(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
  friend class Session;
};

// One fixed-shape, static-instruction NPU design. No Python dependency.
// Buffers follow the compiled runtime sequence's order, starting at XRT arg 3.
// Sessions with the same device, xclbin UUID and kernel share a hardware
// context, while retaining independent instruction buffers and prepared runs.
// A Session is not thread-safe; use separate sessions for concurrent callers.
class Session {
public:
  Session(const std::filesystem::path &xclbin,
          const std::filesystem::path &instructions,
          const std::string &kernel = "MLIR_AIE", unsigned device = 0);
  ~Session();
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  DeviceBuffer allocate(size_t bytes, unsigned argument = 3);
  DeviceRun prepare(const std::vector<DeviceBuffer> &buffers);
  Timing execute(std::vector<Buffer> &buffers, unsigned timeout_ms = 30000);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
} // namespace rwkv::xdna
