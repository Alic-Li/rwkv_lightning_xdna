# SPDX-License-Identifier: Apache-2.0
"""Official FP32 LayerNorm feeding resident mix/shift without a host boundary."""

import json
import aie.iron as iron
from aie.iron import In, InOut, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import C, KERNEL_ROOT, typ, external


def norm_mix(count):
    @iron.jit
    def design(x: In, parameters: In, old: InOut, normalized_pair: Out, mixed: Out):
        packed = ObjectFifo(typ((4 * C)), name="norm_input", depth=1)
        joins = packed.prod().join([0, C, (2 * C), (3 * C)], obj_types=[typ(C)] * 4)
        pair = ObjectFifo(typ((2 * C)), name="pair", depth=1)
        coeff = ObjectFifo(typ(C), name="coeff", depth=1)
        out = ObjectFifo(typ((2 * C)), name="mixed", depth=1)
        splits = out.cons().split([0, C], obj_types=[typ(C)] * 2)
        norm = external(
            "rwkv7_norm_pair", "norm_mix_fp32.cc", [typ((4 * C)), typ((2 * C))]
        )
        mix = external(
            "rwkv7_mix_pair_shift", "mix_fp32.cc", [typ((2 * C)), typ(C), typ((2 * C))]
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
            hw.fill(p, tap=TAP(((2 + count) * C,), 0, [1, 1, 1, C], [0, 0, 0, 1]))
            hb.fill(p, tap=TAP(((2 + count) * C,), C, [1, 1, 1, C], [0, 0, 0, 1]))
            hp.fill(old)
            hc.fill(
                p,
                tap=TAP(
                    ((2 + count) * C,), (2 * C), [1, 1, 1, count * C], [0, 0, 0, 1]
                ),
            )
            hn.drain(pair, wait=True)
            hm.drain(mixed, wait=True)
            hs.drain(old, tap=TAP((C,), 0, [count, 1, 1, C], [0, 0, 0, 1]), wait=True)

        return Program(
            iron.get_current_device(),
            Runtime(
                seq,
                [
                    typ(C),
                    typ((2 + count) * C),
                    typ(C),
                    typ((2 * C)),
                    typ(count * C),
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
    for count in [6]:
        path = KERNEL_ROOT / f"fused-norm-mix-{count}"
        path.mkdir(parents=True, exist_ok=True)
        norm_mix(count).compile(path / "design.xclbin", path / "instructions.bin")
        (path / "config.json").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    dtype="float32",
                    channels=C,
                    mixes=count,
                    exact_fp32=False,
                )
            )
            + "\n"
        )
        print(path, flush=True)
