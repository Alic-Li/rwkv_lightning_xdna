# SPDX-License-Identifier: Apache-2.0
"""Eight parallel four-head stages, fixed arenas and direct strided DMA."""

import json
import os
import aie.iron as iron
from aie.iron import In, InOut, Out, ObjectFifo, Worker, Runtime, Program
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_full import ROOT, typ, external


def stage(prepare):
    @iron.jit
    def design(aux: In, rec: InOut, out: Out):
        a_size, b_size = (1024, 1024) if prepare else (1280, 1536)
        aa = [ObjectFifo(typ(a_size), name=f"a{i}", depth=1) for i in range(8)]
        bb = [ObjectFifo(typ(b_size), name=f"b{i}", depth=1) for i in range(8)]
        oo = [
            ObjectFifo(typ(512 if prepare else 1024), name=f"out{i}", depth=1)
            for i in range(8)
        ]
        rr = (
            [ObjectFifo(typ(1024), name=f"rec{i}", depth=1) for i in range(8)]
            if prepare
            else []
        )
        types = (
            [typ(a_size), typ(b_size), typ(512), typ(1024)]
            if prepare
            else [typ(a_size), typ(b_size), typ(1024)]
        )
        fn = external(
            "rwkv7_prepare_dma" if prepare else "rwkv7_finish_dma",
            "stages_fp32.cc",
            types,
            optimization="-Oz",
        )

        def pre(a, b, o, r, f):
            av, bv, ov, rv = a.acquire(1), b.acquire(1), o.acquire(1), r.acquire(1)
            f(av, bv, ov, rv)
            a.release(1)
            b.release(1)
            o.release(1)
            r.release(1)

        def post(a, b, o, f):
            av, bv, ov = a.acquire(1), b.acquire(1), o.acquire(1)
            f(av, bv, ov)
            a.release(1)
            b.release(1)
            o.release(1)

        workers = [
            Worker(
                pre if prepare else post,
                [aa[i].cons(), bb[i].cons(), oo[i].prod()]
                + ([rr[i].prod()] if prepare else [])
                + [fn],
                stack_size=12288,
            )
            for i in range(8)
        ]
        aux_size = 16384 if prepare else 10240
        out_size = 4096 if prepare else 12288

        def tap(size, offset, vectors):
            return TAP((size,), offset, [1, 1, vectors, 256], [0, 0, 2048, 1])

        def seq(aux, rec, out, ha, hb, ho, hr):
            for i in range(8):
                ha[i].fill(aux, tap=tap(aux_size, i * 256, 4 if prepare else 5))
                hb[i].fill(
                    aux if prepare else rec,
                    tap=tap(
                        aux_size if prepare else 12288,
                        (8192 if prepare else 0) + i * 256,
                        4 if prepare else 6,
                    ),
                )
                ho[i].drain(
                    out, tap=tap(out_size, i * 256, 2 if prepare else 4), wait=True
                )
                if prepare:
                    hr[i].drain(
                        rec,
                        tap=TAP(
                            (12288,), 2048 + i * 256, [1, 2, 2, 256], [0, 6144, 2048, 1]
                        ),
                        wait=True,
                    )

        return Program(
            iron.get_current_device(),
            Runtime(
                seq,
                [
                    typ(aux_size),
                    typ(12288),
                    typ(out_size),
                    [f.prod() for f in aa],
                    [f.prod() for f in bb],
                    [f.cons() for f in oo],
                    [f.cons() for f in rr],
                ],
            ),
            workers=workers,
        ).resolve_program()

    return design


if __name__ == "__main__":
    for name in ["prepare", "finish"]:
        p = ROOT / f"build/kernels/rwkv7-full/fused-{name}"
        p.mkdir(parents=True, exist_ok=True)
        stage(name == "prepare").compile(p / "design.xclbin", p / "instructions.bin")
        (p / "config.json").write_text(
            json.dumps(
                dict(
                    schema_version=2,
                    dtype="float32",
                    exact_fp32=os.environ.get("RWKV_XDNA_EXACT", "1") == "1",
                    channels=2048,
                    head_size=64,
                    stage=name,
                )
            )
            + "\n"
        )
        print(p, flush=True)
