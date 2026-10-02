# SPDX-License-Identifier: Apache-2.0
"""Mix and shift update, broadcast a joined pair to stay within two core DMAs."""

import json
import os
import aie.iron as iron
from aie.iron import In, InOut, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_full import ROOT, KERNEL_ROOT, typ, external


@iron.jit
def design(x: In, old: InOut, coeff: In, mixed: Out):
    pair = ObjectFifo(typ(4096), name="pair", depth=1)
    inputs = pair.prod().join([0, 2048], obj_types=[typ(2048)] * 2)
    c = ObjectFifo(typ(2048), name="coeff", depth=1)
    o = ObjectFifo(typ(4096), name="output", depth=1)
    split = o.cons().split([0, 2048], obj_types=[typ(2048)] * 2)
    fn = external(
        "rwkv7_mix_pair_shift", "mix_fp32.cc", [typ(4096), typ(2048), typ(4096)]
    )

    def core(p, c, o, f):
        a = p.acquire(1)
        for _ in range_(6):
            b, y = c.acquire(1), o.acquire(1)
            f(a, b, y)
            c.release(1)
            o.release(1)
        p.release(1)

    worker = Worker(core, [pair.cons(), c.cons(), o.prod(), fn], stack_size=12288)

    def seq(x, p, c, o, hx, hp, hc, ho, hs):
        hx.fill(x)
        hp.fill(p)
        hc.fill(c)
        ho.drain(o, wait=True)
        hs.drain(p, tap=TAP((2048,), 0, [6, 1, 1, 2048], [0, 0, 0, 1]), wait=True)

    return Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [
                typ(2048),
                typ(2048),
                typ(12288),
                typ(12288),
                inputs[0].prod(),
                inputs[1].prod(),
                c.prod(),
                split[0].cons(),
                split[1].cons(),
            ],
        ),
        workers=[worker],
    ).resolve_program()


@iron.jit
def single(x: In, old: InOut, coeff: In, out: Out):
    pair = ObjectFifo(typ(4096), name="pair", depth=1)
    inputs = pair.prod().join([0, 2048], obj_types=[typ(2048)] * 2)
    c = ObjectFifo(typ(2048), name="coeff", depth=1)
    fo = ObjectFifo(typ(4096), name="output", depth=1)
    split = fo.cons().split([0, 2048], obj_types=[typ(2048)] * 2)
    fn = external(
        "rwkv7_mix_pair_shift", "mix_fp32.cc", [typ(4096), typ(2048), typ(4096)]
    )

    def core(p, c, o, f):
        a, b, y = p.acquire(1), c.acquire(1), o.acquire(1)
        f(a, b, y)
        p.release(1)
        c.release(1)
        o.release(1)

    worker = Worker(core, [pair.cons(), c.cons(), fo.prod(), fn], stack_size=12288)

    def seq(x, p, c, o, hx, hp, hc, ho, hs):
        hx.fill(x)
        hp.fill(p)
        hc.fill(c)
        ho.drain(o, wait=True)
        hs.drain(p, wait=True)

    return Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [typ(2048)] * 4
            + [
                inputs[0].prod(),
                inputs[1].prod(),
                c.prod(),
                split[0].cons(),
                split[1].cons(),
            ],
        ),
        workers=[worker],
    ).resolve_program()


if __name__ == "__main__":
    for name, program, meta in [
        (
            "fused-mix",
            design,
            dict(
                schema_version=3,
                dtype="float32",
                count=2048,
                mixes=6,
                state_update=True,
            ),
        ),
        (
            "fused-shift-mix",
            single,
            dict(schema_version=1, dtype="float32", count=2048, state_update=True),
        ),
    ]:
        p = KERNEL_ROOT / name
        p.mkdir(parents=True, exist_ok=True)
        program.compile(p / "design.xclbin", p / "instructions.bin")
        (p / "config.json").write_text(
            json.dumps(
                dict(meta, exact_fp32=os.environ.get("RWKV_XDNA_EXACT", "1") == "1")
            )
            + "\n"
        )
        print(p, flush=True)
