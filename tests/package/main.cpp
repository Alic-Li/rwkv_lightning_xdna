// SPDX-License-Identifier: Apache-2.0
#include <memory>
#include <rwkv/inference/graph.hpp>
#include <rwkv/inference/model.hpp>
#include <rwkv/xdna/session.hpp>

// Link the installed public API without requiring an NPU for the package test.
std::unique_ptr<rwkv::xdna::Session>
make_session(const std::filesystem::path &image,
             const std::filesystem::path &instructions) {
  return std::make_unique<rwkv::xdna::Session>(image, instructions);
}
std::unique_ptr<rwkv::inference::DecodeGraph>
make_graph(const rwkv::inference::Weights &weights,
           rwkv::inference::RecurrentBackend &backend) {
  return std::make_unique<rwkv::inference::DecodeGraph>(weights, backend);
}
int main() {
  auto backend = rwkv::inference::cpu_backend();
  return backend ? 0 : 1;
}
