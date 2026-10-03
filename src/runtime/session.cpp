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
#include <map>
#include <mutex>
#include <tuple>
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
// Instruction variants of an identical xclbin use one hardware context.
// Weak entries do not extend device-program lifetime beyond the owning sessions.
struct Program {
  xrt::device device;
  xrt::hw_context context;
  xrt::kernel kernel;
  Program(const xrt::xclbin &binary, const std::string &name, unsigned index)
      : device(index), context(device, device.register_xclbin(binary)), kernel(context, name) {}
};
std::shared_ptr<Program> program(const std::filesystem::path &binary,
                                 const std::string &name, unsigned index) {
  xrt::xclbin image(binary.string());
  using Key = std::tuple<unsigned, std::string, std::string>;
  static std::mutex mutex;
  static std::map<Key, std::weak_ptr<Program>> cache;
  std::lock_guard<std::mutex> lock(mutex);
  for (auto it = cache.begin(); it != cache.end();)
    if (it->second.expired()) it = cache.erase(it); else ++it;
  const auto uuid = image.get_uuid();
  const auto &bytes = uuid.get();
  const std::string uuid_key(reinterpret_cast<const char *>(bytes), sizeof(bytes));
  auto &entry = cache[{index, uuid_key, name}];
  if (auto existing = entry.lock()) return existing;
  auto created = std::make_shared<Program>(image, name, index);
  entry = created;
  return created;
}
} // namespace

struct Session::Impl {
  std::shared_ptr<Program> shared_program;
  xrt::device &device;
  xrt::kernel &kernel;
  std::vector<uint32_t> instructions;
  xrt::bo instruction_buffer;
  std::vector<xrt::bo> buffers;
  std::vector<size_t> sizes;
  std::unique_ptr<xrt::run> run;

  Impl(const std::filesystem::path &binary, const std::filesystem::path &inst,
       const std::string &name, unsigned index)
      : shared_program(program(binary, name, index)),
        device(shared_program->device), kernel(shared_program->kernel),
        instructions(read_instructions(inst)),
        instruction_buffer(device, instructions.size() * 4,
                           XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1)) {
    instruction_buffer.write(instructions.data());
    instruction_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
};

Session::Session(const std::filesystem::path &binary,
                 const std::filesystem::path &inst, const std::string &name,
                 unsigned index)
    : impl_(std::make_shared<Impl>(binary, inst, name, index)) {}
Session::~Session() = default;

struct DeviceBuffer::Impl {
  std::shared_ptr<xrt::bo> root;
  size_t offset = 0;
  xrt::bo bo;
  explicit Impl(xrt::bo value)
      : root(std::make_shared<xrt::bo>(std::move(value))), bo(*root) {}
  Impl(std::shared_ptr<xrt::bo> allocation, size_t start, size_t bytes)
      : root(std::move(allocation)), offset(start), bo(*root, bytes, start) {}
};
size_t DeviceBuffer::size() const { return impl_ ? impl_->bo.size() : 0; }
std::optional<size_t> DeviceBuffer::offset_within(const DeviceBuffer &other) const {
  if (!impl_ || !other.impl_ || impl_->root != other.impl_->root ||
      impl_->offset < other.impl_->offset)
    return std::nullopt;
  const size_t offset = impl_->offset - other.impl_->offset;
  if (offset > other.size() || size() > other.size() - offset)
    return std::nullopt;
  return offset;
}
DeviceBuffer DeviceBuffer::slice(size_t offset, size_t bytes) const {
  if (!bytes || offset > size() || bytes > size() - offset)
    throw std::invalid_argument("Device buffer slice out of bounds");
  // Flatten against the root: nesting XRT sub-BOs can lose parent offsets.
  return DeviceBuffer(
      std::make_shared<Impl>(impl_->root, impl_->offset + offset, bytes));
}
void DeviceBuffer::upload(const void *data, size_t bytes, size_t offset) {
  if (!data || !bytes || offset > size() || bytes > size() - offset)
    throw std::invalid_argument("Device upload out of bounds");
  impl_->root->write(data, bytes, impl_->offset + offset);
  impl_->root->sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, impl_->offset + offset);
}
void DeviceBuffer::download(void *data, size_t bytes, size_t offset) const {
  if (!data || !bytes || offset > size() || bytes > size() - offset)
    throw std::invalid_argument("Device download out of bounds");
  impl_->root->sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, impl_->offset + offset);
  impl_->root->read(data, bytes, impl_->offset + offset);
}
struct DeviceRun::Impl {
  // Retain the owning context and every BO for the lifetime of the run.
  std::shared_ptr<void> owner;
  std::vector<DeviceBuffer> buffers;
  xrt::run run;
  Impl(std::shared_ptr<void> o, std::vector<DeviceBuffer> b, xrt::kernel &k)
      : owner(std::move(o)), buffers(std::move(b)), run(k) {}
};
DeviceBuffer Session::allocate(size_t bytes, unsigned argument) {
  if (!bytes || argument < 3 || argument > 34)
    throw std::invalid_argument("Invalid device allocation");
  return DeviceBuffer(std::make_shared<DeviceBuffer::Impl>(
      xrt::bo(impl_->device, bytes, XRT_BO_FLAGS_HOST_ONLY,
              impl_->kernel.group_id(static_cast<int>(argument)))));
}
DeviceRun Session::prepare(const std::vector<DeviceBuffer> &buffers) {
  if (buffers.empty() || buffers.size() > 32)
    throw std::invalid_argument("Invalid prepared buffer count");
  auto p = std::make_shared<DeviceRun::Impl>(impl_, buffers, impl_->kernel);
  p->run.set_arg(0, 3u);
  p->run.set_arg(1, impl_->instruction_buffer);
  p->run.set_arg(2, static_cast<uint32_t>(impl_->instructions.size()));
  for (size_t i = 0; i < buffers.size(); ++i) {
    if (!buffers[i].size())
      throw std::invalid_argument("Empty prepared buffer");
    p->run.set_arg(static_cast<int>(3 + i), buffers[i].impl_->bo);
  }
  return DeviceRun(std::move(p));
}
void DeviceRun::execute(unsigned timeout_ms, RunTiming *timing) {
  if (!timeout_ms)
    throw std::invalid_argument("Invalid device timeout");
  const auto begin = timing ? Clock::now() : Clock::time_point{};
  impl_->run.start();
  const auto submitted = timing ? Clock::now() : Clock::time_point{};
  try {
    auto state = impl_->run.wait(timeout_ms);
    if (state != ERT_CMD_STATE_COMPLETED)
      throw std::runtime_error("Resident NPU dispatch failed: ERT state=" +
                               std::to_string(static_cast<int>(state)));
    if (timing)
      *timing = {micros(begin, submitted), micros(submitted, Clock::now())};
  } catch (...) {
    // Do not release/rebind BOs while a timed-out command may still use them.
    try {
      impl_->run.abort();
    } catch (...) {
    }
    throw;
  }
}

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
    s.run.reset();
    s.buffers = std::move(allocations);
    s.sizes = sizes;
  }
  auto total_start = Clock::now();
  for (size_t i = 0; i < buffers.size(); ++i) {
    s.buffers[i].write(buffers[i].bytes.data());
    s.buffers[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  // The fixed-shape run and argument bindings are reusable across tokens.
  if (!s.run) {
    s.run = std::make_unique<xrt::run>(s.kernel);
    s.run->set_arg(0, 3u);
    s.run->set_arg(1, s.instruction_buffer);
    s.run->set_arg(2, static_cast<uint32_t>(s.instructions.size()));
    for (size_t i = 0; i < buffers.size(); ++i)
      s.run->set_arg(static_cast<int>(3 + i), s.buffers[i]);
  }
  auto &run = *s.run;
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
