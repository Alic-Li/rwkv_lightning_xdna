// SPDX-License-Identifier: Apache-2.0
#include "artifacts.hpp"
#include <fstream>
#include <stdexcept>
namespace rwkv::inference {
void check_artifact(const std::filesystem::path &root, const std::string &name,
                    const nlohmann::json &expected) {
  std::ifstream input(root / name / "config.json");
  if (!input)
    throw std::runtime_error("Missing production artifact: " + name);
  nlohmann::json config;
  input >> config;
  for (auto it = expected.begin(); it != expected.end(); ++it)
    if (!config.contains(it.key()) || config.at(it.key()) != it.value())
      throw std::runtime_error("Incompatible production artifact: " + name +
                               " field " + it.key());
  for (const auto *file : {"design.xclbin", "instructions.bin"})
    if (!std::filesystem::is_regular_file(root / name / file))
      throw std::runtime_error("Missing device binary: " +
                               (root / name / file).string());
}
void validate_resident_artifacts(const std::filesystem::path &root,
                                 const Weights &w) {
  if (w.channels() != 2048 || w.heads() != 32 || w.vocabulary() != 65536)
    throw std::runtime_error(
        "BF16 NPU inference requires C=2048, heads=32, vocabulary=65536");
  check_artifact(root, "upstream-norm",
                 {{"schema_version", 1},
                  {"dtype", "float32"},
                  {"channels", 2048},
                  {"epsilon", 1e-5}});
  check_artifact(root, "fused-norm-mix-6",
                 {{"schema_version", 1},
                  {"dtype", "float32"},
                  {"channels", 2048},
                  {"mixes", 6},
                  {"exact_fp32", false}});
  check_artifact(root, "fused-value",
                 {{"schema_version", 1},
                  {"dtype", "float32"},
                  {"channels", 2048},
                  {"exact_fp32", false}});
  check_artifact(root, "fused-recurrence-stage",
                 {{"schema_version", 1},
                  {"dtype", "float32"},
                  {"channels", 2048},
                  {"head_size", 64},
                  {"arena_vectors", 27},
                  {"exact_fp32", false}});
  check_artifact(root, "bf16-channel-mix",
                 {{"schema_version", 1},
                  {"dtype", "bfloat16"},
                  {"channels", 2048},
                  {"hidden", 8192},
                  {"key_cores", 4},
                  {"value_cores", 4},
                  {"exact_fp32", false}});
  check_artifact(root, "bf16-projection-residual",
                 {{"schema_version", 1},
                  {"dtype", "bfloat16"},
                  {"channels", 2048},
                  {"cores", 11},
                  {"exact_fp32", false}});
  for (int branches : {3, 4})
    check_artifact(root,
                   "bf16-attention-projections-" + std::to_string(branches),
                   {{"schema_version", 1},
                    {"dtype", "bfloat16"},
                    {"channels", 2048},
                    {"rank", 256},
                    {"branches", branches},
                    {"cores", 14},
                    {"exact_fp32", false}});
  check_artifact(root, "bf16-array-gemv-2048-65536",
                 {{"schema_version", 1},
                  {"dtype", "bfloat16"},
                  {"rows", 65536},
                  {"cores", 8},
                  {"k", 2048}});
}
} // namespace rwkv::inference
