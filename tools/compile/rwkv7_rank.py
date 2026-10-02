# SPDX-License-Identifier: Apache-2.0
"""Low-rank projection/activation/projection pipeline with on-chip broadcast."""

import json
import os
import numpy as np
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_full import ROOT, typ, external as base_external


def external(*args):
    return base_external(*args, optimization="-O3")


def rank(activation):
    @iron.jit
    def design(x: In, w1: In, w2: In, raw: Out, activated: Out, y: Out):
        fn = external(
            "rwkv7_rank_tile64",
            "rank_fp32.cc",
            [typ(256), typ(4096), typ(64), np.int32],
        )
        zero = fn.object_file.bind("rwkv7_rank_zero64", [typ(64)])
        act = fn.object_file.bind("rwkv7_rank_activate64", [typ(64), typ(64), np.int32])
        mul = fn.object_file.bind("rwkv7_gemv_tile", [typ(256), typ(4096), typ(16)])
        init = fn.object_file.bind("rwkv7_zero", [typ(16)])
        xs = [ObjectFifo(typ(256), name=f"input{i}", depth=1) for i in range(4)]
        ws1 = [ObjectFifo(typ(4096), name=f"weight1_{i}", depth=1) for i in range(4)]
        rawfifo = ObjectFifo(typ(256), name="raw", depth=1)
        active = ObjectFifo(typ(256), name="active", depth=1)
        raws = rawfifo.prod().join([0, 64, 128, 192], obj_types=[typ(64)] * 4)
        acts = active.prod().join([0, 64, 128, 192], obj_types=[typ(64)] * 4)
        ws = [ObjectFifo(typ(4096), name=f"weight2_{i}", depth=1) for i in range(8)]
        ys = [ObjectFifo(typ(16), name=f"out{i}", depth=1) for i in range(8)]

        def first(x, w, h, activated, f, z, a):
            out, ov = h.acquire(1), activated.acquire(1)
            z(out)
            for row in range_(4):
                for _ in range_(8):
                    xv, wv = x.acquire(1), w.acquire(1)
                    f(xv, wv, out, row)
                    x.release(1)
                    w.release(1)
            a(out, ov, activation)
            h.release(1)
            activated.release(1)

        def second(x, w, y, f, z):
            xv = x.acquire(1)
            for _ in range_(16):
                wv, out = w.acquire(1), y.acquire(1)
                z(out)
                f(xv, wv, out)
                w.release(1)
                y.release(1)
            x.release(1)

        workers = [
            Worker(
                first,
                [
                    xs[i].cons(),
                    ws1[i].cons(),
                    raws[i].prod(),
                    acts[i].prod(),
                    fn,
                    zero,
                    act,
                ],
                stack_size=12288,
            )
            for i in range(4)
        ]
        workers += [
            Worker(
                second,
                [active.cons(), ws[i].cons(), ys[i].prod(), mul, init],
                stack_size=12288,
            )
            for i in range(8)
        ]

        def seq(x, w1, w2, raw, activated, y, hx, hw, hr, ha, hws, hys):
            for i in range(4):
                hx[i].fill(x, tap=TAP((2048,), 0, [4, 1, 1, 2048], [0, 0, 0, 1]))
                hw[i].fill(
                    w1,
                    tap=TAP(
                        (256 * 2048,), i * 64 * 2048, [1, 1, 1, 64 * 2048], [0, 0, 0, 1]
                    ),
                )
            hr.drain(raw, wait=True)
            ha.drain(activated, wait=True)
            for i in range(8):
                hws[i].fill(
                    w2,
                    tap=TAP(
                        (2048 * 256,), i * 256 * 256, [1, 1, 1, 256 * 256], [0, 0, 0, 1]
                    ),
                )
                hys[i].drain(
                    y,
                    tap=TAP((2048,), i * 256, [1, 1, 1, 256], [0, 0, 0, 1]),
                    wait=True,
                )

        return Program(
            iron.get_current_device(),
            Runtime(
                seq,
                [
                    typ(2048),
                    typ(256 * 2048),
                    typ(2048 * 256),
                    typ(256),
                    typ(256),
                    typ(2048),
                    [f.prod() for f in xs],
                    [f.prod() for f in ws1],
                    rawfifo.cons(),
                    active.cons(),
                    [f.prod() for f in ws],
                    [f.cons() for f in ys],
                ],
            ),
            workers=workers,
        ).resolve_program()

    return design


if __name__ == "__main__":
    for a in [0, 1, 2]:
        p = ROOT / f"build/kernels/rwkv7-full/fused-rank-{a}"
        p.mkdir(parents=True, exist_ok=True)
        rank(a).compile(p / "design.xclbin", p / "instructions.bin")
        (p / "config.json").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    dtype="float32",
                    exact_fp32=os.environ.get("RWKV_XDNA_EXACT", "1") == "1",
                    channels=2048,
                    rank=256,
                    activation=a,
                )
            )
            + "\n"
        )
        print(p, flush=True)
