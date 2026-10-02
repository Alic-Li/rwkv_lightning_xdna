// SPDX-License-Identifier: Apache-2.0
#include "ops_internal.hpp"
#include "rwkv/inference/model.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace rwkv::inference {
namespace {
float sigmoid(float x) {
  return x >= 0 ? 1.f / (1.f + std::exp(-x))
                : std::exp(x) / (1.f + std::exp(x));
}
} // namespace
Vector RecurrentBackend::linear(const Vector &x, const Tensor &w,
                                bool transposed) {
  size_t rows = w.shape.at(0), cols = w.shape.at(1);
  const size_t output = transposed ? cols : rows,
               input = transposed ? rows : cols;
  if (x.size() != input)
    throw std::runtime_error("Linear dimension mismatch");
  Vector y(output);
#pragma omp parallel for if (output * input > 65536)
  for (size_t i = 0; i < output; ++i) {
    float sum = 0;
    for (size_t j = 0; j < input; ++j)
      sum += x[j] * w.data[transposed ? j * cols + i : i * cols + j];
    y[i] = sum;
  }
  return y;
}

std::vector<Vector> RecurrentBackend::prefill(
    Vector &state, const std::vector<Vector> &r, const std::vector<Vector> &d,
    const std::vector<Vector> &k, const std::vector<Vector> &v,
    const std::vector<Vector> &a, const std::vector<Vector> &b, size_t n) {
  const size_t t = r.size();
  if (d.size() != t || k.size() != t || v.size() != t || a.size() != t ||
      b.size() != t)
    throw std::runtime_error("Prefill length mismatch");
  std::vector<Vector> y;
  for (size_t i = 0; i < t; ++i)
    y.push_back(step(state, r[i], d[i], k[i], v[i], a[i], b[i], n));
  return y;
}
Vector RecurrentBackend::norm(const Vector &x, const Tensor &w, const Tensor &b,
                              size_t group, float eps) {
  return element(Op::Norm, x, w.data, b.data, {}, group, eps);
}
Vector RecurrentBackend::element(Op op, const Vector &x, const Vector &y,
                                 const Vector &z, const Vector &w, size_t group,
                                 float eps) {
  validate_element(op, x, y, z, w, group, eps);
  const size_t count = x.size();
  for (const auto *v : {&y, &z, &w})
    if (!v->empty() && v->size() != count)
      throw std::runtime_error("Elementwise shape mismatch");
  if (!group || count % group)
    throw std::runtime_error("Reduction group mismatch");
  Vector out(count);
  if (op == Op::NormalizeKey || op == Op::Rkv || op == Op::Norm) {
    for (size_t start = 0; start < count; start += group) {
      if (op == Op::NormalizeKey) {
        double ss = 0;
        for (size_t j = 0; j < group; ++j) {
          float u = x[start + j] * y[start + j];
          ss += double(u) * u;
        }
        float den = float(std::max(std::sqrt(ss), 1e-12));
        for (size_t j = 0; j < group; ++j)
          out[start + j] = (x[start + j] * y[start + j]) / den;
      } else if (op == Op::Rkv) {
        float dot = 0;
        for (size_t j = 0; j < group; ++j)
          dot += x[start + j] * y[start + j] * z[start + j];
        for (size_t j = 0; j < group; ++j)
          out[start + j] = dot * w[start + j];
      } else {
        double mean = 0, var = 0;
        for (size_t j = 0; j < group; ++j)
          mean += x[start + j];
        mean /= group;
        for (size_t j = 0; j < group; ++j) {
          double d = x[start + j] - mean;
          var += d * d;
        }
        double scale = 1 / std::sqrt(var / group + eps);
        for (size_t j = 0; j < group; ++j)
          out[start + j] = float((x[start + j] - mean) * scale) * y[start + j] +
                           z[start + j];
      }
    }
    return out;
  }
  for (size_t i = 0; i < count; ++i) {
    switch (op) {
    case Op::Add:
      out[i] = x[i] + y[i];
      break;
    case Op::Mix:
      out[i] = x[i] + (y[i] - x[i]) * z[i];
      break;
    case Op::Tanh:
      out[i] = std::tanh(x[i]);
      break;
    case Op::Sigmoid:
      out[i] = sigmoid(x[i] + (y.empty() ? 0 : y[i]));
      break;
    case Op::Decay:
      out[i] = std::exp(-std::exp(-0.5f) * sigmoid(x[i] + y[i]));
      break;
    case Op::Multiply:
      out[i] = x[i] * y[i];
      break;
    case Op::ValueResidual:
      out[i] = x[i] + (y[i] - x[i]) * sigmoid(z[i] + w[i]);
      break;
    case Op::ReluSquared: {
      float a = std::max(x[i], 0.f);
      out[i] = a * a;
      break;
    }
    case Op::KeyScale:
      out[i] = x[i] * (1 + (y[i] - 1) * z[i]);
      break;
    case Op::Negate:
      out[i] = -x[i];
      break;
    default:
      throw std::runtime_error("Unknown elementwise operation");
    }
  }
  return out;
}
} // namespace rwkv::inference
