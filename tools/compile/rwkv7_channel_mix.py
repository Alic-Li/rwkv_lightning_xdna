# SPDX-License-Identifier: Apache-2.0
"""Native LayerNorm/shift/mix plus BF16 FFN with on-chip intermediates."""

import json
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, InOut, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import C, H, KERNEL_ROOT, typ, external


def channel_mix_program():
    activation_type = typ
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    tile = 4096
    weight_count, stripe = (2 * C * H, C * H // 4)
    matrix_stack = 12288
    key = external(
        "rwkv7_ffn_key_tile",
        "channel_mix_bf16.cc",
        [activation_type(C), wt(tile), typ(H // 4), np.int32, np.int32],
        optimization="-O3",
        stack_size=8192,
    )
    value = key.object_file.bind(
        "rwkv7_ffn_value_tile",
        [activation_type(H), wt(tile), typ(C // 4), np.int32, np.int32],
    )
    zero = key.object_file.bind("rwkv7_ffn_zero2048", [typ(H // 4)])
    vz = key.object_file.bind("rwkv7_channel_zero512", [typ(C // 4)])
    relu = key.object_file.bind("rwkv7_ffn_activate2048", [typ(H // 4), typ(H // 4)])
    add = key.object_file.bind("rwkv7_ffn_residual", [typ(2 * C), typ(2 * C)])
    norm_in = ObjectFifo(typ(4 * C), name="norm_input", depth=1)
    norm_join = norm_in.prod().join([0, C, 2 * C, 3 * C], obj_types=[typ(C)] * 4)
    pair_norm = ObjectFifo(typ(2 * C), name="norm_pair", depth=1)
    coeff = ObjectFifo(typ(C), name="coefficient", depth=1)
    norm = external("rwkv7_norm_pair", "norm_mix_fp32.cc", [typ(4 * C), typ(2 * C)])
    mix = external("rwkv7_mix_pair", "mix_pair_fp32.cc", [typ(2 * C), typ(C), typ(C)])
    fx = ObjectFifo(typ(C), name="input", depth=1)
    w1 = [ObjectFifo(wt(tile), name=f"keyw{i}", depth=2) for i in range(4)]
    raw = ObjectFifo(typ(H), name="raw", depth=1)
    act = ObjectFifo(typ(H), name="active", depth=1)
    kr = raw.prod().join([i * (H // 4) for i in range(4)], obj_types=[typ(H // 4)] * 4)
    ka = act.prod().join([i * (H // 4) for i in range(4)], obj_types=[typ(H // 4)] * 4)
    w2 = [ObjectFifo(wt(tile), name=f"valuew{i}", depth=2) for i in range(4)]
    pair = ObjectFifo(typ(2 * C), name="residual_pair", depth=1)
    joins = pair.prod().join(
        [0, C // 4, C // 2, 3 * C // 4, C],
        obj_types=[typ(C // 4)] * 4 + [typ(C)],
    )
    vo = joins[:4]
    final = ObjectFifo(typ(2 * C), name="result", depth=1)

    def first(x, w, r, a, f, z, activation):
        xv, rv, av = (x.acquire(1), r.acquire(1), a.acquire(1))
        z(rv)
        for row in range_(H // 64):
            for col in range_(C // 256):
                wv = w.acquire(1)
                f(xv, wv, rv, row, col)
                w.release(1)
        activation(rv, av)
        x.release(1)
        r.release(1)
        a.release(1)

    def second(x, w, o, f, z):
        xv, ov = (x.acquire(1), o.acquire(1))
        z(ov)
        for row in range_(C // 64):
            for col in range_(H // 256):
                wv = w.acquire(1)
                f(xv, wv, ov, row, col)
                w.release(1)
        x.release(1)
        o.release(1)

    def finish(p, o, f):
        pv, ov = (p.acquire(1), o.acquire(1))
        f(pv, ov)
        p.release(1)
        o.release(1)

    def normalize(p, o, f):
        pv, ov = (p.acquire(1), o.acquire(1))
        f(pv, ov)
        p.release(1)
        o.release(1)

    def mixes(p, c, o, f):
        pv, cv, ov = (p.acquire(1), c.acquire(1), o.acquire(1))
        f(pv, cv, ov)
        p.release(1)
        c.release(1)
        o.release(1)

    workers = [
        Worker(
            normalize,
            [norm_in.cons(), pair_norm.prod(), norm],
            stack_size=12288,
        ),
        Worker(
            mixes,
            [pair_norm.cons(), coeff.cons(), fx.prod(), mix],
            stack_size=12288,
        ),
    ]
    workers += [
        Worker(
            first,
            [fx.cons(), w1[i].cons(), kr[i].prod(), ka[i].prod(), key, zero, relu],
            stack_size=matrix_stack,
        )
        for i in range(4)
    ]
    workers += [
        Worker(
            second,
            [act.cons(), w2[i].cons(), vo[i].prod(), value, vz],
            stack_size=matrix_stack,
        )
        for i in range(4)
    ]
    workers += [Worker(finish, [pair.cons(), final.prod(), add], stack_size=12288)]

    def seq(
        x,
        parameters,
        w,
        diag,
        out,
        hx,
        hw,
        hb,
        hold,
        hc,
        hn,
        hm,
        hw1,
        hw2,
        hr,
        ha,
        hres,
        ho,
    ):
        hx.fill(x)
        hw.fill(parameters, tap=TAP((3 * C,), 0, [1, 1, 1, C], [0, 0, 0, 1]))
        hb.fill(parameters, tap=TAP((3 * C,), C, [1, 1, 1, C], [0, 0, 0, 1]))
        hold.fill(diag, tap=TAP((3 * C + 2 * H,), 0, [1, 1, 1, C], [0, 0, 0, 1]))
        hc.fill(parameters, tap=TAP((3 * C,), 2 * C, [1, 1, 1, C], [0, 0, 0, 1]))
        hn.drain(
            diag,
            tap=TAP((3 * C + 2 * H,), 0, [1, 1, 1, 2 * C], [0, 0, 0, 1]),
            wait=True,
        )
        hm.drain(
            diag,
            tap=TAP((3 * C + 2 * H,), 2 * C, [1, 1, 1, C], [0, 0, 0, 1]),
            wait=True,
        )
        for i in range(4):
            hw1[i].fill(
                w, tap=TAP((weight_count,), i * stripe, [1, 1, 1, stripe], [0, 0, 0, 1])
            )
        for i in range(4):
            hw2[i].fill(
                w,
                tap=TAP(
                    (weight_count,),
                    weight_count // 2 + i * stripe,
                    [1, 1, 1, stripe],
                    [0, 0, 0, 1],
                ),
            )
        hr.drain(
            diag,
            tap=TAP((3 * C + 2 * H,), 3 * C, [1, 1, 1, H], [0, 0, 0, 1]),
            wait=True,
        )
        ha.drain(
            diag,
            tap=TAP((3 * C + 2 * H,), 3 * C + H, [1, 1, 1, H], [0, 0, 0, 1]),
            wait=True,
        )
        hres.fill(x)
        ho.drain(out, wait=True)

    program = Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [
                typ(C),
                typ(3 * C),
                wt(weight_count),
                typ(3 * C + 2 * H),
                typ(2 * C),
                *[f.prod() for f in norm_join],
                coeff.prod(),
                pair_norm.cons(),
                fx.cons(),
                [f.prod() for f in w1],
                [f.prod() for f in w2],
                raw.cons(),
                act.cons(),
                joins[4].prod(),
                final.cons(),
            ],
        ),
        workers=workers,
    )
    return program.resolve_program()


@iron.jit
def design(x: In, parameters: In, weights: In, diagnostic: InOut, result: Out):
    return channel_mix_program()


if __name__ == "__main__":
    path = KERNEL_ROOT / "bf16-channel-mix"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                dtype="bfloat16",
                channels=C,
                hidden=H,
                key_cores=4,
                value_cores=4,
                exact_fp32=False,
                weight_layout="row_major",
            )
        )
        + "\n"
    )
    print(path, flush=True)
