# SPDX-License-Identifier: Apache-2.0
"""BF16 key/ReLU^2/value/residual pipeline with an on-chip hidden vector."""

import json
import os
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_full import KERNEL_ROOT, typ, external


@iron.jit
def design(x: In, weights: In, residual: In, hidden: Out, result: Out):
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    key = external(
        "rwkv7_ffn_key_tile",
        "ffn_pipeline_bf16.cc",
        [typ(2048), wt(4096), typ(2048), np.int32, np.int32],
        optimization="-O3",
    )
    value = key.object_file.bind(
        "rwkv7_ffn_value_tile", [typ(8192), wt(4096), typ(256), np.int32, np.int32]
    )
    zero = key.object_file.bind("rwkv7_ffn_zero2048", [typ(2048)])
    vz = key.object_file.bind("rwkv7_ffn_zero256", [typ(256)])
    relu = key.object_file.bind("rwkv7_ffn_activate2048", [typ(2048), typ(2048)])
    add = key.object_file.bind("rwkv7_ffn_residual", [typ(4096), typ(4096)])
    fx = ObjectFifo(typ(2048), name="input", depth=1)
    w1 = [ObjectFifo(wt(4096), name=f"keyw{i}", depth=2) for i in range(4)]
    raw = ObjectFifo(typ(8192), name="raw", depth=1)
    act = ObjectFifo(typ(8192), name="active", depth=1)
    kr = raw.prod().join([i * 2048 for i in range(4)], obj_types=[typ(2048)] * 4)
    ka = act.prod().join([i * 2048 for i in range(4)], obj_types=[typ(2048)] * 4)
    w2 = [ObjectFifo(wt(4096), name=f"valuew{i}", depth=1) for i in range(8)]
    pair = ObjectFifo(typ(4096), name="residual_pair", depth=1)
    joins = pair.prod().join(
        [0, 1024, 2048], obj_types=[typ(1024), typ(1024), typ(2048)]
    )
    groups = [ObjectFifo(typ(1024), name=f"value_group{i}", depth=1) for i in range(2)]
    vo = []
    for group in groups:
        vo += group.prod().join([0, 256, 512, 768], obj_types=[typ(256)] * 4)
    copy = key.object_file.bind("rwkv7_ffn_copy1024", [typ(1024), typ(1024)])
    final = ObjectFifo(typ(4096), name="result", depth=1)

    def first(x, w, r, a, f, z, activation):
        xv, rv, av = x.acquire(1), r.acquire(1), a.acquire(1)
        z(rv)
        for row in range_(128):
            for col in range_(8):
                wv = w.acquire(1)
                f(xv, wv, rv, row, col)
                w.release(1)
        activation(rv, av)
        x.release(1)
        r.release(1)
        a.release(1)

    def second(x, w, o, f, z):
        xv, ov = x.acquire(1), o.acquire(1)
        z(ov)
        for row in range_(16):
            for col in range_(32):
                wv = w.acquire(1)
                f(xv, wv, ov, row, col)
                w.release(1)
        x.release(1)
        o.release(1)

    def finish(p, o, f):
        pv, ov = p.acquire(1), o.acquire(1)
        f(pv, ov)
        p.release(1)
        o.release(1)

    workers = [
        Worker(
            first,
            [fx.cons(), w1[i].cons(), kr[i].prod(), ka[i].prod(), key, zero, relu],
            stack_size=12288,
        )
        for i in range(4)
    ]
    workers += [
        Worker(
            second,
            [act.cons(), w2[i].cons(), vo[i].prod(), value, vz],
            stack_size=12288,
        )
        for i in range(8)
    ]
    workers += [
        Worker(finish, [groups[i].cons(), joins[i].prod(), copy], stack_size=12288)
        for i in range(2)
    ]
    workers += [Worker(finish, [pair.cons(), final.prod(), add], stack_size=12288)]

    def seq(x, w, res, h, out, hx, hw1, hw2, hr, ha, hres, ho):
        hx.fill(x)
        for i in range(4):
            hw1[i].fill(
                w, tap=TAP((33554432,), i * 4194304, [1, 1, 1, 4194304], [0, 0, 0, 1])
            )
        for i in range(8):
            hw2[i].fill(
                w,
                tap=TAP(
                    (33554432,),
                    16777216 + i * 2097152,
                    [1, 1, 1, 2097152],
                    [0, 0, 0, 1],
                ),
            )
        hr.drain(h, tap=TAP((16384,), 0, [1, 1, 1, 8192], [0, 0, 0, 1]), wait=True)
        ha.drain(h, tap=TAP((16384,), 8192, [1, 1, 1, 8192], [0, 0, 0, 1]), wait=True)
        hres.fill(res)
        ho.drain(out, wait=True)

    return Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [
                typ(2048),
                wt(33554432),
                typ(2048),
                typ(16384),
                typ(4096),
                fx.prod(),
                [f.prod() for f in w1],
                [f.prod() for f in w2],
                raw.cons(),
                act.cons(),
                joins[2].prod(),
                final.cons(),
            ],
        ),
        workers=workers,
    ).resolve_program()


if __name__ == "__main__":
    path = KERNEL_ROOT / "bf16-ffn-pipeline"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                dtype="bfloat16",
                channels=2048,
                hidden=8192,
                key_cores=4,
                value_cores=8,
                exact_fp32=os.environ.get("RWKV_XDNA_EXACT", "1") == "1",
            )
        )
        + "\n"
    )
    print(path, flush=True)
