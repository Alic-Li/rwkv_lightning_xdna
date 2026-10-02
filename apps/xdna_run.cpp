// SPDX-License-Identifier: Apache-2.0
#include "rwkv/xdna/session.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("Usage: xdna-run <case/manifest.json>");
    auto path = std::filesystem::absolute(argv[1]);
    std::ifstream input(path);
    auto manifest = nlohmann::json::parse(input);
    if (manifest.at("schema_version") != 1)
      throw std::runtime_error("Unsupported manifest schema");
    auto resolve = [&](const std::string &name) {
      return path.parent_path() / name;
    };
    std::vector<rwkv::xdna::Buffer> buffers;
    for (const auto &arg : manifest.at("buffers")) {
      auto bytes = arg.at("bytes").get<size_t>();
      if (!bytes || bytes > (1ULL << 30))
        throw std::runtime_error("Buffer size outside supported range");
      bool output = arg.at("direction") == "out";
      if (!output && arg.at("direction") != "in")
        throw std::runtime_error("Unknown buffer direction");
      rwkv::xdna::Buffer buffer{std::vector<uint8_t>(bytes, 0xcd), output};
      if (!output) {
        auto file = resolve(arg.at("file").get<std::string>());
        if (std::filesystem::file_size(file) != bytes)
          throw std::runtime_error("Input file size does not match manifest");
        std::ifstream data(file, std::ios::binary);
        if (!data.read(reinterpret_cast<char *>(buffer.bytes.data()), bytes))
          throw std::runtime_error("Cannot read input buffer");
      }
      buffers.push_back(std::move(buffer));
    }
    rwkv::xdna::Session session(
        resolve(manifest.at("xclbin").get<std::string>()),
        resolve(manifest.at("instructions").get<std::string>()),
        manifest.value("kernel", "MLIR_AIE"));
    auto repetitions = manifest.value("repetitions", 3);
    if (repetitions < 1 || repetitions > 10000)
      throw std::runtime_error("Invalid repetition count");
    nlohmann::json timings = nlohmann::json::array();
    for (int i = 0; i < repetitions; ++i) {
      for (auto &buffer : buffers)
        if (buffer.output)
          std::fill(buffer.bytes.begin(), buffer.bytes.end(), 0xcd);
      auto timing = session.execute(buffers);
      timings.push_back(
          {{"dispatch_us", timing.dispatch_us},
           {"transfer_and_dispatch_us", timing.transfer_and_dispatch_us}});
      // Save every repetition, so the offline checker verifies every dispatch.
      for (size_t b = 0; b < buffers.size(); ++b)
        if (buffers[b].output) {
          auto name = manifest.at("buffers")[b].at("file").get<std::string>();
          auto output = resolve(name + "." + std::to_string(i));
          std::ofstream stream(output, std::ios::binary);
          stream.write(reinterpret_cast<const char *>(buffers[b].bytes.data()),
                       buffers[b].bytes.size());
          if (!stream)
            throw std::runtime_error("Cannot write NPU output");
        }
    }
    std::cout
        << nlohmann::json({{"status", "executed"}, {"timings", timings}}).dump()
        << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "xdna-run: " << error.what() << '\n';
    return 1;
  }
}
