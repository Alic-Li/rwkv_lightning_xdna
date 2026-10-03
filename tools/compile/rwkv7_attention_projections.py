# SPDX-License-Identifier: Apache-2.0
"""R/K/V and W/A/G/(V) in one 14-core BF16 program, 16 shim input streams."""

import argparse
import json
import hashlib
import shutil
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program, Buffer
from aie.dialects import arith, memref
from aie.ir import IndexType, F32Type
from aie.helpers.dialects.scf import if_, else_
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import KERNEL_ROOT, typ, external


def projections(count, batch=1, mode_reuse=False, norm_only=False):
    if mode_reuse and (batch != 1 or count != 4):
        raise ValueError("Mode reuse requires batch=1 and four rank branches")
    if norm_only and not mode_reuse:
        raise ValueError("Norm instructions require mode reuse")
    arena_stride = 61440 if batch == 2 and count == 4 else 55296
    value_stride = arena_stride if batch == 2 and count == 4 else 6144
    value_size = (batch - 1) * value_stride + 6144
    source = (
        "prefill_attention_bf16.cc" if batch == 2 else "attention_projections_bf16.cc"
    )
    if mode_reuse:
        source = "mode_attention_probe.cc"
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    rkv_weight_type = typ(2048) if mode_reuse else wt(4096)
    n_rkv = 3 * 2048 * 2048
    n_rank = count * 256 * 2048
    n_weights = n_rkv + 2 * n_rank

    @iron.jit
    def design(mixed: In, weights: In, arena: Out, value_aux: Out, rank_aux: Out):
        fn = external(
            "rwkv7_prefill_attention_rank"
            if batch == 2
            else "rwkv7_attention_rank_tile",
            source,
            [typ(2048 * batch), wt(4096), typ(128 * batch), np.int32, np.int32],
            optimization="-O3",
        )
        z = fn.object_file.bind(
            "rwkv7_prefill_attention_zero256"
            if batch == 2
            else "rwkv7_attention_zero128",
            [typ(128 * batch)],
        )
        act = fn.object_file.bind(
            "rwkv7_prefill_attention_activate"
            if batch == 2
            else "rwkv7_attention_activate128",
            [typ(128 * batch), typ(128 * batch), np.int32],
        )
        mul = fn.object_file.bind(
            "rwkv7_prefill_attention_second" if batch == 2 else "rwkv7_gemv_tile",
            [typ(256 * batch), wt(4096), typ(16 * batch)],
        )
        rz = fn.object_file.bind(
            "rwkv7_prefill_attention_zero32" if batch == 2 else "rwkv7_zero",
            [typ(16 * batch)],
        )
        rkv = fn.object_file.bind(
            "rwkv7_prefill_attention_rkv" if batch == 2 else "rwkv7_attention_rkv_tile",
            [typ(2048 * batch), rkv_weight_type, typ(16 * batch), np.int32],
        )
        rx = ObjectFifo(typ(2048 * batch), name="rkv_x", depth=1)
        kx = ObjectFifo(typ(2048 * batch), name="rank_x", depth=1)
        rw = [ObjectFifo(rkv_weight_type, name=f"rkv_w{i}", depth=2) for i in range(8)]
        ry = [ObjectFifo(typ(16 * batch), name=f"rkv_y{i}", depth=2) for i in range(8)]
        w1 = [ObjectFifo(wt(4096), name=f"first_w{i}", depth=1) for i in range(2)]
        w2 = [ObjectFifo(wt(4096), name=f"second_w{i}", depth=1) for i in range(4)]
        raw = ObjectFifo(typ(256 * batch), name="raw", depth=1)
        activated = ObjectFifo(typ(256 * batch), name="active", depth=1)
        hr = raw.prod().join([0, 128 * batch], obj_types=[typ(128 * batch)] * 2)
        ha = activated.prod().join([0, 128 * batch], obj_types=[typ(128 * batch)] * 2)
        ky = [ObjectFifo(typ(16 * batch), name=f"rank_y{i}", depth=1) for i in range(4)]

        def rkv_core(x, w, y, f, z):
            for _ in range_(3):
                xv = x.acquire(1)
                for _ in range_(16):
                    out = y.acquire(1)
                    z(out)
                    for col in range_(8):
                        wv = w.acquire(1)
                        f(xv, wv, out, col)
                        w.release(1)
                    y.release(1)
                x.release(1)

        if mode_reuse:
            normalize = fn.object_file.bind("rwkv7_mode_norm", [typ(2048)] * 4)
            copy16 = fn.object_file.bind(
                "rwkv7_mode_copy16", [typ(2048), typ(16), np.int32]
            )
            mix16 = fn.object_file.bind(
                "rwkv7_mode_mix16", [typ(2048)] * 3 + [typ(16), np.int32]
            )
            scratch = Buffer(typ(2048), name="normalized")

        def mode_rkv_core(x, w, y, f, z, norm, cp, mix, scratch, lane):
            # Reuse an existing DMA input: another control FIFO would exceed
            # the two input channels per core. Only lane 0 performs norm/mix;
            # other RKV lanes consume x/old to keep broadcast locks balanced.
            header = w.acquire(1)
            opcode = memref.LoadOp(
                header, [arith.ConstantOp(IndexType.get(), 0)]
            ).result
            is_norm = arith.CmpFOp(
                arith.CmpFPredicate.OEQ, opcode, arith.ConstantOp(F32Type.get(), 0.0)
            ).result
            w.release(1)
            with if_(is_norm) as branch:
                if lane == 0:
                    xv = x.acquire(1)
                    affine = w.acquire(2)
                    norm(xv, affine[0], affine[1], scratch)
                    w.release(2)
                    x.release(1)
                    # Stream old only after releasing x: holding both forces
                    # extra FIFO storage and exceeds local memory with scratch.
                    old = x.acquire(1)
                    for source in [scratch, old]:
                        for i in range_(128):
                            out = y.acquire(1)
                            cp(source, out, i * 16)
                            y.release(1)
                    for _ in range_(6):
                        coeff = w.acquire(1)
                        for i in range_(128):
                            out = y.acquire(1)
                            mix(scratch, old, coeff, out, i * 16)
                            y.release(1)
                        w.release(1)
                    for i in range_(128):
                        out = y.acquire(1)
                        cp(scratch, out, i * 16)
                        y.release(1)
                    x.release(1)
                else:
                    for _ in range_(2):
                        x.acquire(1)
                        x.release(1)
            with else_(branch):
                rkv_core(x, w, y, f, z)

        def first(x, w, r, a, f, z, activation):
            for p in range_(count):
                xv, rv, av = x.acquire(1), r.acquire(1), a.acquire(1)
                z(rv)
                for row in range_(8):
                    for col in range_(8):
                        wv = w.acquire(1)
                        f(xv, wv, rv, row, col)
                        w.release(1)
                activation(rv, av, p)
                x.release(1)
                r.release(1)
                a.release(1)

        def second(x, w, y, f, z):
            for _ in range_(count):
                xv = x.acquire(1)
                for _ in range_(32):
                    wv, yv = w.acquire(1), y.acquire(1)
                    z(yv)
                    f(xv, wv, yv)
                    w.release(1)
                    y.release(1)
                x.release(1)

        workers = [
            Worker(
                mode_rkv_core if mode_reuse else rkv_core,
                [rx.cons(), rw[i].cons(), ry[i].prod(), rkv, rz]
                + (
                    [normalize, copy16, mix16, scratch if i == 0 else None, i]
                    if mode_reuse
                    else []
                ),
                stack_size=16384 if mode_reuse else 12288,
                **({"dynamic_objfifo_lowering": True} if mode_reuse else {}),
            )
            for i in range(8)
        ]
        workers += [
            Worker(
                first,
                [kx.cons(), w1[i].cons(), hr[i].prod(), ha[i].prod(), fn, z, act],
                stack_size=12288,
            )
            for i in range(2)
        ]
        workers += [
            Worker(
                second,
                [activated.cons(), w2[i].cons(), ky[i].prod(), mul, rz],
                stack_size=12288,
            )
            for i in range(4)
        ]

        def seq(
            x,
            w,
            arena,
            value_aux,
            rank_aux,
            hxr,
            hxk,
            hwr,
            hyr,
            hw1,
            hw2,
            hraw,
            hact,
            hyk,
        ):
            if mode_reuse:
                # One FP32 opcode in an 8KiB header on each existing weight stream.
                for hw in hwr:
                    hw.fill(
                        w,
                        tap=TAP(
                            (n_weights + 4096,),
                            n_weights,
                            [1, 1, 1, 4096],
                            [0, 0, 0, 1],
                        ),
                    )
            for slot in [0, 2, 3]:
                hxr.fill(
                    x,
                    tap=TAP(
                        (12288 * batch,),
                        slot * 2048,
                        [batch, 1, 1, 2048],
                        [12288, 0, 0, 1],
                    ),
                )
            for slot in [1, 4, 5, 3][:count]:
                hxk.fill(
                    x,
                    tap=TAP(
                        (12288 * batch,),
                        slot * 2048,
                        [batch, 1, 1, 2048],
                        [12288, 0, 0, 1],
                    ),
                )
            for i in range(8):
                hwr[i].fill(
                    w,
                    tap=TAP(
                        (n_weights,),
                        i * 3 * 256 * 2048,
                        [1, 1, 1, 3 * 256 * 2048],
                        [0, 0, 0, 1],
                    ),
                )
                destinations = [
                    (arena, arena_stride, 13 * 2048),
                    (arena, arena_stride, 0),
                    (arena, arena_stride, 16 * 2048)
                    if count == 3
                    else (value_aux, value_stride, 0),
                ]
                for target, size, offset in destinations:
                    hyr[i].drain(
                        target,
                        tap=TAP(
                            (
                                (arena_stride * batch)
                                if target is arena
                                else value_size,
                            ),
                            offset + i * 256,
                            [16, batch, 1, 16],
                            [16, size, 0, 1],
                        ),
                        wait=True,
                    )
            for i in range(2):
                hw1[i].fill(
                    w,
                    tap=TAP(
                        (n_weights,),
                        n_rkv + i * count * 128 * 2048,
                        [1, 1, 1, count * 128 * 2048],
                        [0, 0, 0, 1],
                    ),
                )
            hraw.drain(
                rank_aux,
                tap=TAP(
                    (count * 512 * batch,),
                    0,
                    [count, 2, batch, 128],
                    [256, 128, count * 512, 1],
                ),
                wait=True,
            )
            hact.drain(
                rank_aux,
                tap=TAP(
                    (count * 512 * batch,),
                    count * 256,
                    [count, 2, batch, 128],
                    [256, 128, count * 512, 1],
                ),
                wait=True,
            )
            for i in range(4):
                hw2[i].fill(
                    w,
                    tap=TAP(
                        (n_weights,),
                        n_rkv + n_rank + i * count * 512 * 256,
                        [1, 1, 1, count * 512 * 256],
                        [0, 0, 0, 1],
                    ),
                )
                destinations = [
                    (arena, arena_stride, 4096),
                    (arena, arena_stride, 2048),
                    (arena, arena_stride, 12 * 2048),
                    (value_aux, value_stride, 2048),
                ]
                for target, size, offset in destinations[:count]:
                    hyk[i].drain(
                        target,
                        tap=TAP(
                            (
                                (arena_stride * batch)
                                if target is arena
                                else value_size,
                            ),
                            offset + i * 512,
                            [32, batch, 1, 16],
                            [16, size, 0, 1],
                        ),
                        wait=True,
                    )

        def norm_seq(
            x, p, old, pair, mixed, hxr, hxk, hwr, hyr, hw1, hw2, hraw, hact, hyk
        ):
            for hw in hwr:
                hw.fill(p, tap=TAP((18432,), 16384, [1, 1, 1, 2048], [0, 0, 0, 1]))
            hxr.fill(x)
            hxr.fill(old)
            hwr[0].fill(p, tap=TAP((18432,), 0, [1, 1, 1, 16384], [0, 0, 0, 1]))
            hyr[0].drain(pair, wait=True)
            hyr[0].drain(mixed, wait=True)
            hyr[0].drain(old, wait=True)

        return Program(
            iron.get_current_device(),
            Runtime(
                norm_seq if norm_only else seq,
                [
                    *(
                        [typ(2048), typ(18432), typ(2048), typ(4096), typ(12288)]
                        if norm_only
                        else [
                            typ(12288 * batch),
                            wt(n_weights + (4096 if mode_reuse else 0)),
                            typ(arena_stride * batch),
                            typ(value_size),
                            typ(count * 512 * batch),
                        ]
                    ),
                    rx.prod(),
                    kx.prod(),
                    [f.prod() for f in rw],
                    [f.cons() for f in ry],
                    [f.prod() for f in w1],
                    [f.prod() for f in w2],
                    raw.cons(),
                    activated.cons(),
                    [f.cons() for f in ky],
                ],
            ),
            workers=workers,
        ).resolve_program()

    return design


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--mode-reuse",
        action="store_true",
        help="Share batch-1 four-branch attention workers with norm/mix",
    )
    args = parser.parse_args()
    batch = args.batch
    if args.mode_reuse and batch != 1:
        parser.error("--mode-reuse requires --batch 1")
    for count in [3, 4]:
        path = KERNEL_ROOT / (
            f"bf16-attention-projections-{count}" + ("-b2" if batch == 2 else "")
        )
        path.mkdir(parents=True, exist_ok=True)
        reuse = args.mode_reuse and count == 4
        projections(count, batch, mode_reuse=reuse).compile(
            path / "design.xclbin", path / "instructions.bin"
        )
        (path / "config.json").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    dtype="bfloat16",
                    channels=2048,
                    branches=count,
                    rank=256,
                    cores=14,
                    exact_fp32=False,
                    **(
                        dict(mode_attention=True, mode_header_bytes=8192)
                        if reuse
                        else {}
                    ),
                    **(
                        dict(
                            batch=2,
                            arena_stride=61440 if count == 4 else 55296,
                            value_stride=61440 if count == 4 else 6144,
                        )
                        if batch == 2
                        else {}
                    ),
                )
            )
            + "\n"
        )
        print(path, flush=True)

        if reuse:
            norm = KERNEL_ROOT / "mode-norm-mix-6"
            norm.mkdir(parents=True, exist_ok=True)
            projections(4, mode_reuse=True, norm_only=True).compile(
                norm / "design.xclbin", norm / "instructions.bin"
            )
            pdi = (path / "design.prj/main.pdi").read_bytes()
            if pdi != (norm / "design.prj/main.pdi").read_bytes():
                raise RuntimeError(
                    "Norm/attention PDI mismatch; context sharing is unsafe"
                )
            shutil.copyfile(path / "design.xclbin", norm / "design.xclbin")
            (norm / "config.json").write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        dtype="float32",
                        channels=2048,
                        mixes=6,
                        exact_fp32=False,
                        mode_attention=True,
                        mode_header_bytes=8192,
                        shared_pdi_sha256=hashlib.sha256(pdi).hexdigest(),
                    )
                )
                + "\n"
            )
