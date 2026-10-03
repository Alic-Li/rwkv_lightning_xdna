// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rwkv/inference/graph.hpp"
#include <nlohmann/json.hpp>
namespace rwkv::inference {
void check_artifact(const std::filesystem::path &root, const std::string &name,
                    const nlohmann::json &expected);
void validate_resident_artifacts(const std::filesystem::path &root,
                                 const Weights &weights, WeightMode mode);
void validate_prefill_artifacts(const std::filesystem::path &, WeightMode,
                                size_t chunk_tokens);
} // namespace rwkv::inference
