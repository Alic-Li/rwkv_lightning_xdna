# SPDX-License-Identifier: Apache-2.0
"""Offline resident decode ABIs; DMA joins/splits replace CPU packing."""

import argparse
import json
import os
from pathlib import Path
import aie.iron as iron
from aie.iron import In, Out, InOut, ObjectFifo, Worker, Runtime, Program
from rwkv7_full import ROOT, KERNEL_ROOT, typ, external


def ops_design(source):
    @iron.jit
    def design(meta: In, x: In, y: In, z: In, w: In, out: Out):
        packed, output = ObjectFifo(typ(8208), name="packed", depth=1), ObjectFifo(
            typ(2048), name="out", depth=1
        )
        inputs = packed.prod().join(
            [0, 16, 2064, 4112, 6160], obj_types=[typ(16)] + [typ(2048)] * 4
        )
        fn = external("rwkv7_ops_fp32", source, [typ(8208), typ(2048)])

        def core(i, o, f):
            a, b = i.acquire(1), o.acquire(1)
            f(a, b)
            i.release(1)
            o.release(1)

        worker = Worker(core, [packed.cons(), output.prod(), fn], stack_size=12288)

        def seq(m, x, y, z, w, o, hm, hx, hy, hz, hw, ho):
            hm.fill(m)
            hx.fill(x)
            hy.fill(y)
            hz.fill(z)
            hw.fill(w)
            ho.drain(o, wait=True)

        return Program(
            iron.get_current_device(),
            Runtime(
                seq,
                [typ(16)]
                + [typ(2048)] * 5
                + [f.prod() for f in inputs]
                + [output.cons()],
            ),
            workers=[worker],
        ).resolve_program()

    return design


ops = ops_design("ops_fp32.cc")
fast_ops = ops_design("ops_fast_fp32.cc")


@iron.jit
def recurrent(state: InOut, vectors: In, y: Out):
    # Six graph vectors occupy 2048-float slots in one BO. A strided DMA
    # gathers one 64-element head from each slot, without CPU repacking.
    from aie.helpers.taplib import TensorAccessPattern

    fs = ObjectFifo(typ(4096), name="state", depth=1)
    fp = ObjectFifo(typ(384), name="packed", depth=1)
    fo = ObjectFifo(typ(4160), name="output", depth=1)
    outputs = fo.cons().split([0, 4096], obj_types=[typ(4096), typ(64)])
    fn = external("rwkv7_wkv_fp32", "wkv7_fp32.cc", [typ(4096), typ(384), typ(4160)])

    def core(s, p, o, f):
        sv, pv, ov = s.acquire(1), p.acquire(1), o.acquire(1)
        f(sv, pv, ov)
        s.release(1)
        p.release(1)
        o.release(1)

    worker = Worker(core, [fs.cons(), fp.cons(), fo.prod(), fn], stack_size=12288)

    def seq(s, vectors, y, hs, hp, hn, hy):
        hs.fill(s)
        hp.fill(
            vectors,
            tap=TensorAccessPattern((10304,), 0, [1, 1, 6, 64], [0, 0, 2048, 1]),
        )
        hn.drain(s, wait=True)
        hy.drain(y, wait=True)

    return Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [typ(4096), typ(10304), typ(64), fs.prod(), fp.prod()]
            + [f.cons() for f in outputs],
        ),
        workers=[worker],
    ).resolve_program()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=KERNEL_ROOT)
    args = parser.parse_args()
    for name, design in [
        ("resident-ops", ops),
        ("resident-ops-fast", fast_ops),
        ("resident-decode", recurrent),
    ]:
        path = args.output / name
        path.mkdir(parents=True, exist_ok=True)
        design.compile(path / "design.xclbin", path / "instructions.bin")
        (path / "config.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "abi": name,
                    "dtype": "float32",
                    "exact_fp32": os.environ.get("RWKV_XDNA_EXACT", "1") == "1",
                }
            )
            + "\n"
        )
        print(name, flush=True)
