# SPDX-License-Identifier: Apache-2.0
"""Official FP32 LayerNorm feeding resident mix/shift without a host boundary."""

import json
import os
import aie.iron as iron
from aie.iron import In, InOut, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_full import KERNEL_ROOT, typ, external


def norm_mix(count):
    @iron.jit
    def design(x: In, parameters: In, old: InOut, normalized_pair: Out, mixed: Out):
        packed = ObjectFifo(typ(8192), name="norm_input", depth=1)
        joins = packed.prod().join([0, 2048, 4096, 6144], obj_types=[typ(2048)] * 4)
        pair = ObjectFifo(typ(4096), name="pair", depth=1)
        coeff = ObjectFifo(typ(2048), name="coeff", depth=1)
        out = ObjectFifo(typ(4096), name="mixed", depth=1)
        splits = out.cons().split([0, 2048], obj_types=[typ(2048)] * 2)
        norm = external("rwkv7_norm_pair", "norm_mix_fp32.cc", [typ(8192), typ(4096)])
        mix = external(
            "rwkv7_mix_pair_shift", "mix_fp32.cc", [typ(4096), typ(2048), typ(4096)]
        )

        def normalize(p, o, f):
            pv, ov = p.acquire(1), o.acquire(1)
            f(pv, ov)
            p.release(1)
            o.release(1)

        def mixes(p, c, o, f):
            pv = p.acquire(1)
            for _ in range_(count):
                cv, ov = c.acquire(1), o.acquire(1)
                f(pv, cv, ov)
                c.release(1)
                o.release(1)
            p.release(1)

        workers = [
            Worker(normalize, [packed.cons(), pair.prod(), norm], stack_size=12288),
            Worker(
                mixes, [pair.cons(), coeff.cons(), out.prod(), mix], stack_size=12288
            ),
        ]

        def seq(x, p, old, pair, mixed, hx, hw, hb, hp, hc, hn, hm, hs):
            hx.fill(x)
            hw.fill(p, tap=TAP(((2 + count) * 2048,), 0, [1, 1, 1, 2048], [0, 0, 0, 1]))
            hb.fill(
                p, tap=TAP(((2 + count) * 2048,), 2048, [1, 1, 1, 2048], [0, 0, 0, 1])
            )
            hp.fill(old)
            hc.fill(
                p,
                tap=TAP(
                    ((2 + count) * 2048,), 4096, [1, 1, 1, count * 2048], [0, 0, 0, 1]
                ),
            )
            hn.drain(pair, wait=True)
            hm.drain(mixed, wait=True)
            hs.drain(
                old, tap=TAP((2048,), 0, [count, 1, 1, 2048], [0, 0, 0, 1]), wait=True
            )

        return Program(
            iron.get_current_device(),
            Runtime(
                seq,
                [
                    typ(2048),
                    typ((2 + count) * 2048),
                    typ(2048),
                    typ(4096),
                    typ(count * 2048),
                    *[f.prod() for f in joins],
                    coeff.prod(),
                    pair.cons(),
                    splits[0].cons(),
                    splits[1].cons(),
                ],
            ),
            workers=workers,
        ).resolve_program()

    return design


if __name__ == "__main__":
    if os.environ.get("RWKV_XDNA_EXACT", "1") != "0":
        raise RuntimeError("norm/mix uses the native upstream FP32 arithmetic contract")
    for count in [1, 6]:
        path = KERNEL_ROOT / f"fused-norm-mix-{count}"
        path.mkdir(parents=True, exist_ok=True)
        norm_mix(count).compile(path / "design.xclbin", path / "instructions.bin")
        (path / "config.json").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    dtype="float32",
                    channels=2048,
                    mixes=count,
                    exact_fp32=False,
                )
            )
            + "\n"
        )
        print(path, flush=True)
