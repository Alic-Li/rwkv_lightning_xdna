// SPDX-License-Identifier: Apache-2.0
#include <memory>
#include <rwkv/xdna/session.hpp>

// Link the installed public API without requiring an NPU for the package test.
std::unique_ptr<rwkv::xdna::Session>
make_session(const std::filesystem::path &image,
             const std::filesystem::path &instructions) {
  return std::make_unique<rwkv::xdna::Session>(image, instructions);
}
int main() { return 0; }
