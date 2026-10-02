// SPDX-License-Identifier: Apache-2.0
// Focused native runlist capability probe, not an inference graph backend.
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("kernel directory required");
    std::string dir = argv[1];
    std::ifstream file(dir + "/instructions.bin",
                       std::ios::binary | std::ios::ate);
    auto bytes = file.tellg();
    if (bytes <= 0 || bytes % 4)
      throw std::runtime_error("Invalid instructions");
    std::vector<uint32_t> words(size_t(bytes) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char *>(words.data()), bytes);
    xrt::device device(0);
    xrt::hw_context ctx(
        device, device.register_xclbin(xrt::xclbin(dir + "/design.xclbin")));
    xrt::kernel kernel(ctx, "MLIR_AIE");
    xrt::bo inst(device, words.size() * 4, XCL_BO_FLAGS_CACHEABLE,
                 kernel.group_id(1));
    inst.write(words.data());
    inst.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::bo state(device, 4096 * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    xrt::bo input(device, 384 * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    xrt::bo output(device, 4160 * 4, XRT_BO_FLAGS_HOST_ONLY,
                   kernel.group_id(5));
    std::vector<float> zeros(4096), packed(384), result(4160);
    state.write(zeros.data());
    state.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::run first(kernel), second(kernel);
    for (auto *r : {&first, &second}) {
      r->set_arg(0, 3u);
      r->set_arg(1, inst);
      r->set_arg(2, uint32_t(words.size()));
      r->set_arg(3, state);
      r->set_arg(4, input);
      r->set_arg(5, output);
    }
    xrt::runlist list(ctx);
    list.add(first);
    list.add(second);
    for (int repetition = 1; repetition <= 2; ++repetition) {
      std::fill(packed.begin(), packed.end(), 0);
      for (int j = 0; j < 64; ++j) {
        packed[j] = float(repetition);
        packed[64 + j] = 1;
        packed[128 + j] = 1;
        packed[192 + j] = 1;
      }
      input.write(packed.data());
      input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      list.execute();
      if (list.wait(std::chrono::milliseconds(30000)) !=
          std::cv_status::no_timeout)
        throw std::runtime_error("runlist timed out");
      output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      output.read(result.data());
      for (int j = 0; j < 64; ++j)
        if (result[4096 + j] != float(64 * repetition))
          throw std::runtime_error("Runlist output mismatch");
    }
    std::cout
        << "Two-run list reused with changed inputs: numerical check passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
