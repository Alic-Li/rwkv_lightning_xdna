# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Extracted without behavioral changes from test/python/npu/test_kernels_e2e.py
# at Xilinx/mlir-aie d53582d. Only the offline design builder is retained.
import aie.iron as iron
import numpy as np
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program, kernels
from aie.iron.algorithms import kernel_design as kd

_CASCADE_DIM = 16


@iron.jit
def _cascade_design(a_put: In, b_put: In, a_get: In, b_get: In, c_out: Out):
    from aie.iron import CascadeFlow
    from aie.iron.device import Tile

    m = k = n = _CASCADE_DIM
    get = kernels.cascade_mm(dim_m=m, dim_k=k, dim_n=n)
    put = kernels.cascade_mm_put(dim_m=m, dim_k=k, dim_n=n)
    zero = get.contract.initializers[0][1](get)
    a_ty, b_ty, c_ty = get.arg_types()
    fifos = {
        name: ObjectFifo(ty, name=name)
        for name, ty in (("ap", a_ty), ("bp", b_ty), ("ag", a_ty), ("bg", b_ty))
    }
    of_c = ObjectFifo(c_ty, name="c")
    unused = np.zeros(m * n, dtype=np.dtype(kd.shape_dtype(c_ty)[1]))

    def put_core(of_a, of_b, scratch, k_put):
        a, b = of_a.acquire(1), of_b.acquire(1)
        k_put(a, b, scratch)
        of_a.release(1)
        of_b.release(1)

    def get_core(of_a, of_b, of_c, k_zero, k_get):
        a, b, c = of_a.acquire(1), of_b.acquire(1), of_c.acquire(1)
        k_zero(c)
        k_get(a, b, c)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)

    from aie.iron.buffer import Buffer

    scratch = Buffer(c_ty, name="scratch", initial_value=unused)
    # The cascade runs north to south: the PUT tile sits above the GET tile.
    w_put = Worker(
        put_core,
        [fifos["ap"].cons(), fifos["bp"].cons(), scratch, put],
        tile=Tile(0, 3),
    )
    w_get = Worker(
        get_core,
        [fifos["ag"].cons(), fifos["bg"].cons(), of_c.prod(), zero, get],
        tile=Tile(0, 2),
    )
    CascadeFlow(w_put, w_get)

    def seq(ap, bp, ag, bg, c, h_ap, h_bp, h_ag, h_bg, h_c):
        for handle, host in ((h_ap, ap), (h_bp, bp), (h_ag, ag), (h_bg, bg)):
            handle.fill(host)
        h_c.drain(c, wait=True)

    rt = Runtime(
        seq,
        [a_ty, b_ty, a_ty, b_ty, c_ty]
        + [fifos[name].prod() for name in ("ap", "bp", "ag", "bg")]
        + [of_c.cons()],
    )
    return Program(
        iron.get_current_device(), rt, workers=[w_put, w_get]
    ).resolve_program()

