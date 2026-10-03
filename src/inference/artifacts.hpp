// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rwkv/inference/model.hpp"
#include <nlohmann/json.hpp>
namespace rwkv::inference {
void check_artifact(const std::filesystem::path &root, const std::string &name,
                    const nlohmann::json &expected);
void validate_resident_artifacts(const std::filesystem::path &root,
                                 const Weights &weights);
} // namespace rwkv::inference
