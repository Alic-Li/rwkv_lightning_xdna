# SPDX-License-Identifier: Apache-2.0
"""Offline-only compiler for the float32 RWKV-7 recurrent transition."""

import argparse
import json
from pathlib import Path
import numpy as np
import aie.iron as iron
from aie.iron import ExternalFunction, In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.device import NPU2
from aie.utils import set_current_device

ROOT = Path(__file__).resolve().parents[2]
set_current_device(NPU2())


@iron.jit
def design(state: In, packed: In, output: Out):
    st = np.ndarray[(4096,), np.dtype[np.float32]]
    pt = np.ndarray[(384,), np.dtype[np.float32]]
    ot = np.ndarray[(4160,), np.dtype[np.float32]]
    kernel = ExternalFunction(
        "rwkv7_wkv_fp32",
        source_file=str(ROOT / "kernels/rwkv/wkv7_fp32.cc"),
        arg_types=[st, pt, ot],
        compile_flags=["-fno-fast-math", "-ffp-contract=off"],
        stack_size_override=4096,
    )
    fs, fp, fo = (
        ObjectFifo(st, name="state", depth=1),
        ObjectFifo(pt, name="packed", depth=1),
        ObjectFifo(ot, name="output", depth=1),
    )

    def core(s, p, o, fn):
        sv, pv, ov = s.acquire(1), p.acquire(1), o.acquire(1)
        fn(sv, pv, ov)
        s.release(1)
        p.release(1)
        o.release(1)

    worker = Worker(core, [fs.cons(), fp.cons(), fo.prod(), kernel], stack_size=8192)

    def sequence(s, p, o, hs, hp, ho):
        hs.fill(s)
        hp.fill(p)
        ho.drain(o, wait=True)

    rt = Runtime(sequence, [st, pt, ot, fs.prod(), fp.prod(), fo.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/kernels/rwkv7-wkv-fp32"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    design.compile(args.output / "design.xclbin", args.output / "instructions.bin")
    (args.output / "config.json").write_text(
        json.dumps(
            dict(schema_version=1, head_size=64, dtype="float32", layout="key_value"),
            indent=2,
        )
        + "\n"
    )
    print(args.output)
