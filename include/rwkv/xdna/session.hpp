// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace rwkv::xdna {
struct Buffer {
  std::vector<uint8_t> bytes;
  bool output = false;
};
struct Timing {
  double dispatch_us = 0;
  double transfer_and_dispatch_us = 0;
};

// One fixed-shape, static-instruction NPU design. No Python dependency.
// Buffers follow the compiled runtime sequence's order, starting at XRT arg 3.
// A Session is not thread-safe; use separate sessions for concurrent callers.
class Session {
public:
  Session(const std::filesystem::path &xclbin,
          const std::filesystem::path &instructions,
          const std::string &kernel = "MLIR_AIE", unsigned device = 0);
  ~Session();
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  Timing execute(std::vector<Buffer> &buffers, unsigned timeout_ms = 30000);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace rwkv::xdna
