# SPDX-License-Identifier: Apache-2.0
"""Single-token W8A8 block-cyclic W1 and streaming K-major W2."""

import json
import numpy as np
import aie.iron as iron
from aie.iron import In, InOut, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import KERNEL_ROOT, typ, external


def channel_mix_program():
    place = lambda col, row: Tile(col, row)
    activation_type = lambda n: np.ndarray[(n // 256 * 320,), np.dtype[np.uint8]]
    wt = lambda n: np.ndarray[(n,), np.dtype[np.uint8]]
    tile = 4160
    weight_count, stripe = (8192 * tile, 1024 * tile)
    matrix_stack = 20480
    key = external(
        "rwkv7_ffn_key_tile",
        "channel_mix_int8.cc",
        [activation_type(2048), wt(tile), typ(512), np.int32, np.int32],
        optimization="-O3",
        stack_size=16384,
    )
    value = key.object_file.bind(
        "rwkv7_ffn_value_tile",
        [activation_type(1024), wt(tile), typ(512), np.int32, np.int32],
    )
    zero = key.object_file.bind("rwkv7_ffn_zero256", [typ(512)])
    vz = key.object_file.bind("rwkv7_channel_zero512", [typ(512)])
    relu = key.object_file.bind(
        "rwkv7_ffn_activate_quant256", [typ(512), activation_type(256)]
    )
    add = key.object_file.bind("rwkv7_ffn_residual", [typ(4096), typ(4096)])
    norm_in = ObjectFifo(typ(8192), name="norm_input", depth=1)
    norm_join = norm_in.prod().join(
        [0, 2048, 4096, 6144], tile=place(0, 1), obj_types=[typ(2048)] * 4
    )
    pair_norm = ObjectFifo(typ(4096), name="norm_pair", depth=1)
    coeff = ObjectFifo(typ(2048), name="coefficient", depth=1)
    norm = external("rwkv7_norm_pair", "norm_mix_fp32.cc", [typ(8192), typ(4096)])
    mix = external(
        "rwkv7_mix_pair", "mix_pair_fp32.cc", [typ(4096), typ(2048), typ(2048)]
    )
    fx = ObjectFifo(typ(2048), name="input", depth=1)
    fx_int8 = ObjectFifo(activation_type(2048), name="input_int8", depth=1)
    fx_half = fx_int8.cons().forward(tile=place(1, 1), name="input_int8_fanout")
    convert = external(
        "rwkv7_channel_convert2048",
        "channel_mix_convert_int8.cc",
        [typ(2048), activation_type(2048)],
        optimization="-O3",
    )
    w1 = [ObjectFifo(wt(tile), name=f"keyw{i}", depth=2) for i in range(4)]
    raw = ObjectFifo(typ(2048), name="raw_and_active", depth=1)
    kr = raw.prod().join(
        [i * 512 for i in range(4)], tile=place(2, 1), obj_types=[typ(512)] * 4
    )
    act_half = ObjectFifo(activation_type(1024), name="quantized_active", depth=1)
    ka = act_half.prod().join(
        [i * 320 for i in range(4)],
        tile=place(5, 1),
        obj_types=[activation_type(256)] * 4,
    )
    w2 = [ObjectFifo(wt(tile), name=f"valuew{i}", depth=2) for i in range(4)]
    pair = ObjectFifo(typ(4096), name="residual_pair", depth=1)
    joins = pair.prod().join(
        [0, 512, 1024, 1536, 2048],
        tile=place(7, 1),
        obj_types=[typ(512)] * 4 + [typ(2048)],
    )
    vo = joins[:4]
    final = ObjectFifo(typ(4096), name="result", depth=1)

    def first(x, w, r, a, f, z, activation):
        xv = x.acquire(1)
        for block in range_(8):
            rv, av = (r.acquire(1), a.acquire(1))
            z(rv)
            for row in range_(16):
                for col in range_(8):
                    wv = w.acquire(1)
                    f(xv, wv, rv, row, col)
                    w.release(1)
            activation(rv, av)
            r.release(1)
            a.release(1)
        x.release(1)

    def second(x, w, o, f, z):
        ov = o.acquire(1)
        z(ov)
        for block in range_(8):
            xv = x.acquire(1)
            for col in range_(4):
                for row in range_(32):
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
            tile=place(0, 2),
            stack_size=12288,
        ),
        Worker(
            mixes,
            [pair_norm.cons(), coeff.cons(), fx.prod(), mix],
            tile=place(1, 2),
            stack_size=12288,
        ),
    ]
    workers += [
        Worker(
            first,
            [fx_half.cons(), w1[i].cons(), kr[i].prod(), ka[i].prod(), key, zero, relu],
            tile=place(i, 3),
            stack_size=matrix_stack,
        )
        for i in range(4)
    ]
    workers += [
        Worker(
            second,
            [act_half.cons(), w2[i].cons(), vo[i].prod(), value, vz],
            tile=place(i + 4, 3),
            stack_size=matrix_stack,
        )
        for i in range(4)
    ]
    workers += [
        Worker(
            finish, [pair.cons(), final.prod(), add], tile=place(7, 2), stack_size=12288
        )
    ]
    workers += [
        Worker(
            normalize,
            [fx.cons(), fx_int8.prod(), convert],
            tile=place(2, 2),
            stack_size=12288,
        )
    ]

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
        hres,
        ho,
    ):
        hx.fill(x)
        hw.fill(parameters, tap=TAP((6144,), 0, [1, 1, 1, 2048], [0, 0, 0, 1]))
        hb.fill(parameters, tap=TAP((6144,), 2048, [1, 1, 1, 2048], [0, 0, 0, 1]))
        hold.fill(diag, tap=TAP((22528,), 0, [1, 1, 1, 2048], [0, 0, 0, 1]))
        hc.fill(parameters, tap=TAP((6144,), 4096, [1, 1, 1, 2048], [0, 0, 0, 1]))
        hn.drain(diag, tap=TAP((22528,), 0, [1, 1, 1, 4096], [0, 0, 0, 1]), wait=True)
        hm.drain(
            diag, tap=TAP((22528,), 4096, [1, 1, 1, 2048], [0, 0, 0, 1]), wait=True
        )
        for i in range(4):
            hw1[i].fill(
                w,
                tap=TAP(
                    (weight_count,),
                    i * (128 * tile),
                    [1, 8, 1, 128 * tile],
                    [0, 512 * tile, 0, 1],
                ),
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
            tap=TAP((22528,), 6144, [8, 4, 2, 256], [1024, 256, 8192, 1]),
            wait=True,
        )
        hres.fill(x)
        ho.drain(out, wait=True)

    program = Program(
        iron.get_current_device(),
        Runtime(
            seq,
            [
                typ(2048),
                typ(6144),
                wt(weight_count),
                typ(22528),
                typ(4096),
                *[f.prod() for f in norm_join],
                coeff.prod(),
                pair_norm.cons(),
                fx.cons(),
                [f.prod() for f in w1],
                [f.prod() for f in w2],
                raw.cons(),
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
    path = KERNEL_ROOT / "int8-channel-mix"
    if path.is_symlink():
        raise ValueError("Refusing to compile through a symlink")
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                dtype="int8",
                channels=2048,
                hidden=8192,
                key_cores=4,
                value_cores=4,
                tile_bytes=4160,
                weight_chunk_tiles=1,
                quantization="symmetric_per_output_127",
                scale="fp16_expanded_fp32",
                activation_dtype="int8",
                activation_placement="w1_stream256",
                weight_layout="w2_k_major_4",
                w1_output_partition="block_cyclic_256",
                diagnostic_layout="raw_active_block_cyclic_scatter",
                integer_reduction="row_reduce",
                activation_block=256,
                activation_packet_bytes=320,
                multiply_dtype="int8",
                partial_sum_dtype="float32",
                accumulator_dtype="int32",
                activation_conversion="dynamic_symmetric_block256_nearest_even",
            )
        )
        + "\n"
    )
    print(path, flush=True)
