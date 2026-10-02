// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <stdexcept>
// Minimal non-owning adapter for the supplied sampler. Model logits are F32.
// _Float16 is the native IEEE half storage type on the supported Clang/GCC
// host.
namespace half_float {
using half = _Float16;
}
namespace rwkvmobile {
enum class TensorDType { F32, F16 };
struct Tensor1D {
  void *data_ptr = nullptr;
  TensorDType dtype = TensorDType::F32;
  size_t size = 0;
};
inline float tensor1d_get_f32(const Tensor1D &t, size_t i) {
  if (i >= t.size)
    throw std::out_of_range("Tensor1D index");
  return t.dtype == TensorDType::F32
             ? static_cast<float *>(t.data_ptr)[i]
             : float(static_cast<half_float::half *>(t.data_ptr)[i]);
}
inline void tensor1d_set_f32(Tensor1D &t, size_t i, float v) {
  if (i >= t.size)
    throw std::out_of_range("Tensor1D index");
  if (t.dtype == TensorDType::F32)
    static_cast<float *>(t.data_ptr)[i] = v;
  else
    static_cast<half_float::half *>(t.data_ptr)[i] = half_float::half(v);
}
inline void tensor1d_add_bias(Tensor1D &t, size_t i, float v) {
  tensor1d_set_f32(t, i, tensor1d_get_f32(t, i) + v);
}
inline Tensor1D tensor1d_subview(const Tensor1D &t, size_t start, size_t size) {
  if (start > t.size || size > t.size - start)
    throw std::out_of_range("Tensor1D subview");
  size_t width =
      t.dtype == TensorDType::F32 ? sizeof(float) : sizeof(half_float::half);
  return {static_cast<char *>(t.data_ptr) + start * width, t.dtype, size};
}
} // namespace rwkvmobile
