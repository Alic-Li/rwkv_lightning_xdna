// SPDX-License-Identifier: Apache-2.0
// Native BF16 4x8x8 MMUL only. BFP emulation is forbidden for this probe.
#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
#error BFP emulation violates the precision contract
#endif
#define bf16_f32_ONLY
#define DIM_M 64
#define DIM_K 64
#define DIM_N 64
#include "../../third_party/mlir-aie/aie_kernels/linalg/mm.cc"
extern "C" void rwkv7_gemm_probe(bfloat16 *a, bfloat16 *b, float *c) {
  for (int i = 0; i < 4096; i += 32)
    aie::store_v(c + i, aie::zeros<float, 32>());
  matmul_bf16_f32(a, b, c);
}
