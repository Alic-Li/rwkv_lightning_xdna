// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rwkv/inference/model.hpp"
#include <cmath>
#include <stdexcept>
namespace rwkv::inference {
inline void validate_element(Op op, const Vector &x, const Vector &y,
                             const Vector &z, const Vector &w, size_t group,
                             float eps) {
  if (int(op) < 0 || int(op) > 12 || x.empty() || !group || x.size() % group)
    throw std::invalid_argument("Invalid elementwise operation/group");
  for (const auto *v : {&y, &z, &w})
    if (!v->empty() && v->size() != x.size())
      throw std::invalid_argument("Elementwise shape mismatch");
  int arity = 1;
  switch (op) {
  case Op::Add:
  case Op::Decay:
  case Op::Multiply:
  case Op::NormalizeKey:
    arity = 2;
    break;
  case Op::Mix:
  case Op::KeyScale:
  case Op::Norm:
    arity = 3;
    break;
  case Op::ValueResidual:
  case Op::Rkv:
    arity = 4;
    break;
  default:
    break;
  }
  if ((arity >= 2 && y.empty()) || (arity >= 3 && z.empty()) ||
      (arity >= 4 && w.empty()))
    throw std::invalid_argument("Missing elementwise operand");
  if (op == Op::Norm && (!(eps > 0) || !std::isfinite(eps)))
    throw std::invalid_argument("Invalid normalization epsilon");
}
} // namespace rwkv::inference
