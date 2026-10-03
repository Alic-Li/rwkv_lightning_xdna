# SPDX-License-Identifier: Apache-2.0
"""Experimental four-token BF16/W8A16 FFN with streamed value activations; not production."""
import argparse
import json
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import CompileTime, In, InOut, ObjectFifo, Worker, Runtime, Program
from aie.iron.runtime import TaskGroup
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import KERNEL_ROOT, typ, external

def bt(n):
    return np.ndarray[(n,), np.dtype[bfloat16]]

@iron.jit
def design(x: In, parameters: In, weights: In, fp32: InOut, bf16: InOut, *, projection_input: CompileTime[bool] = False, quantized: CompileTime[bool] = False):
    wt = (lambda n: np.ndarray[(n,), np.dtype[np.uint8]]) if quantized else bt
    tile = 4160 if quantized else 4096
    matrix_stack = 20480 if quantized else 12288
    stripe, count = 1024 * tile, 8192 * tile
    packed = ObjectFifo(typ(6144), name="norm_input", depth=1)
    inputs = packed.prod().join([0, 2048, 4096], tile=Tile(0, 1), obj_types=[typ(2048)] * 3)
    normalized = ObjectFifo(typ(2048), name="normalized", depth=1)
    params = ObjectFifo(typ(4096), name="mix_parameters", depth=1)
    joins = params.prod().join([0, 2048], tile=Tile(1, 1), obj_types=[typ(2048)] * 2)
    mix_out = ObjectFifo(bt(8192), name="mixed", depth=1)
    mx = mix_out.cons().forward(tile=Tile(2, 1), name="mixed_fanout")
    final_shift = ObjectFifo(typ(2048), name="final_shift", depth=1)
    norm = external("rwkv7_layer_norm", "norm_fp32.cc", [typ(6144), typ(2048)])
    mix = external("rwkv7_prefill_mix_tokens", "prefill_mix_bf16.cc",
                   [typ(2048), typ(4096), typ(2048), bt(8192), np.int32], optimization="-O3")
    init = mix.object_file.bind("rwkv7_prefill_shift_init", [typ(4096), typ(2048)])
    key = external("rwkv7_chunk4_key_int8" if quantized else "rwkv7_chunk4_key",
                   "chunk4_channel_int8.cc" if quantized else "chunk4_channel_bf16.cc",
                   [bt(8192), wt(tile), typ(64), np.int32], optimization="-O3",
                   stack_size=16384 if quantized else 8192)
    zero_key = key.object_file.bind("rwkv7_chunk4_zero64", [typ(64)])
    collect = key.object_file.bind("rwkv7_chunk4_collect", [typ(64), typ(8192), bt(8192), np.int32])
    value = key.object_file.bind("rwkv7_chunk4_value_int8" if quantized else "rwkv7_chunk4_value",
                                 [bt(1024), wt(tile), typ(2048), np.int32] + ([np.int32] if quantized else []))
    zero_value = key.object_file.bind("rwkv7_chunk4_zero2048", [typ(2048)])
    add = key.object_file.bind("rwkv7_chunk4_add", [typ(2048), typ(8192), typ(2048), np.int32])
    wk = [ObjectFifo(wt(tile), name=f"kw{i}", depth=2) for i in range(4)]
    wv = [ObjectFifo(wt(tile), name=f"vw{i}", depth=2) for i in range(4)]
    packets = [ObjectFifo(typ(64), name=f"packet{i}", depth=2) for i in range(4)]
    raw_out = ObjectFifo(typ(32768), name="raw", depth=1)
    act_out = ObjectFifo(bt(32768), name="activated", depth=1)
    kr = raw_out.prod().join([i * 8192 for i in range(4)], tile=Tile(3, 1), obj_types=[typ(8192)] * 4)
    ka = act_out.prod().join([i * 8192 for i in range(4)], tile=Tile(4, 1), obj_types=[bt(8192)] * 4)
    block = ObjectFifo(bt(1024), name="activation_block", depth=2)
    fanout = block.cons().forward(tile=Tile(5, 1), name="activation_fanout")
    proj = ObjectFifo(typ(8192), name="projected", depth=1)
    vp = proj.prod().join([i * 2048 for i in range(4)], tile=Tile(6, 1), obj_types=[typ(2048)] * 4)
    residual_x = ObjectFifo(typ(2048), name="residual_x", depth=1)
    final = ObjectFifo(typ(2048), name="final", depth=1)
    def normalize(p, o, f):
        for _ in range_(4):
            pv, ov = p.acquire(1), o.acquire(1)
            f(pv, ov)
            p.release(1)
            o.release(1)
    def mixes(x, p, s, o, initialize, f):
        pv, sv, ov = p.acquire(1), s.acquire(1), o.acquire(1)
        initialize(pv, sv)
        for token in range_(4):
            xv = x.acquire(1)
            f(xv, pv, sv, ov, token)
            x.release(1)
        p.release(1)
        s.release(1)
        o.release(1)
    def keys(x, w, r, f, z):
        xv = x.acquire(1)
        for row in range_(128):
            rv = r.acquire(1)
            z(rv)
            for col in range_(8):
                weights = w.acquire(1)
                f(xv, weights, rv, col)
                w.release(1)
            r.release(1)
        x.release(1)
    def collect_rows(p, r, a, f):
        rv, av = r.acquire(1), a.acquire(1)
        for row in range_(128):
            pv = p.acquire(1)
            f(pv, rv, av, row)
            p.release(1)
        r.release(1)
        a.release(1)
    def values(x, w, p, f, z):
        pv = p.acquire(1)
        z(pv)
        for row in range_(32):
            for col in range_(32):
                xv, weights = x.acquire(1), w.acquire(1)
                if quantized:
                    f(xv, weights, pv, row, col)
                else:
                    f(xv, weights, pv, row)
                x.release(1)
                w.release(1)
        p.release(1)
    def residual(x, p, o, f):
        pv = p.acquire(1)
        for token in range_(4):
            xv, ov = x.acquire(1), o.acquire(1)
            f(xv, pv, ov, token)
            x.release(1)
            o.release(1)
        p.release(1)
    workers = [
        Worker(normalize, [packed.cons(), normalized.prod(), norm], tile=Tile(0, 2), stack_size=12288),
        Worker(mixes, [normalized.cons(), params.cons(), final_shift.prod(), mix_out.prod(), init, mix],
               tile=Tile(1, 2), stack_size=12288),
    ]
    workers += [Worker(keys, [mx.cons(), wk[i].cons(), packets[i].prod(), key, zero_key],
                       tile=Tile(i, 3), stack_size=matrix_stack) for i in range(4)]
    workers += [Worker(collect_rows, [packets[i].cons(), kr[i].prod(), ka[i].prod(), collect],
                       tile=Tile(i, 4), stack_size=12288) for i in range(4)]
    workers += [Worker(values, [fanout.cons(), wv[i].cons(), vp[i].prod(), value, zero_value],
                       tile=Tile(i + 4, 3), stack_size=matrix_stack) for i in range(4)]
    workers += [Worker(residual, [residual_x.cons(), proj.cons(), final.prod(), add],
                       tile=Tile(7, 2), stack_size=12288)]
    def seq(x, p, w, f, h, hx, hw, hb, hc, hs, hkw, hvw, hm, hr, ha, hv, hres, hy, ho, hblock):
        first = TaskGroup()
        input_tap = TAP((16384,), 2048, [4, 1, 1, 2048], [4096, 0, 0, 1]) if projection_input else None
        hx.fill(x, tap=input_tap, group=first)
        for handle, offset in ((hw, 0), (hb, 2048)):
            handle.fill(p, tap=TAP((6144,), offset, [4, 1, 1, 2048], [0, 0, 0, 1]), group=first)
        hc.fill(p, tap=TAP((6144,), 4096, [1, 1, 1, 2048], [0, 0, 0, 1]), group=first)
        hs.fill(f, tap=TAP((51200,), 0, [1, 1, 1, 2048], [0, 0, 0, 1]), group=first)
        for i in range(4):
            hkw[i].fill(w, tap=TAP((count,), i * stripe, [1, 1, 1, stripe], [0, 0, 0, 1]), group=first)
        hm.drain(h, tap=TAP((40960,), 0, [1, 1, 1, 8192], [0, 0, 0, 1]), wait=True, group=first)
        hr.drain(f, tap=TAP((51200,), 2048, [4, 4, 4, 512], [2048, 8192, 512, 1]), wait=True, group=first)
        ha.drain(h, tap=TAP((40960,), 8192, [4, 4, 4, 512], [2048, 8192, 512, 1]), wait=True, group=first)
        ho.drain(f, tap=TAP((51200,), 0, [1, 1, 1, 2048], [0, 0, 0, 1]), wait=True, group=first)
        first.finish()
        second = TaskGroup()
        hres.fill(x, tap=input_tap, group=second)
        hv.drain(f, tap=TAP((51200,), 34816, [4, 4, 1, 512], [512, 2048, 0, 1]), wait=True, group=second)
        hy.drain(f, tap=TAP((51200,), 43008, [1, 1, 1, 8192], [0, 0, 0, 1]), wait=True, group=second)
        hblock.fill(h, tap=TAP((40960,), 8192, [32, 32, 4, 256], [0, 256, 8192, 1]), group=second)
        for i in range(4):
            hvw[i].fill(w, tap=TAP((count,), count // 2 + i * stripe, [1, 1, 1, stripe], [0, 0, 0, 1]), group=second)
        second.finish()
    return Program(iron.get_current_device(), Runtime(seq, [
        typ(16384 if projection_input else 8192), typ(6144), wt(count), typ(51200), bt(40960),
        *[f.prod() for f in inputs], *[f.prod() for f in joins],
        [f.prod() for f in wk], [f.prod() for f in wv],
        mix_out.cons(), raw_out.cons(), act_out.cons(), proj.cons(),
        residual_x.prod(), final.cons(), final_shift.cons(), block.prod(),
    ]), workers=workers).resolve_program()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-input", action="store_true")
    parser.add_argument("--int8", action="store_true")
    args = parser.parse_args()
    path = KERNEL_ROOT / (("int8" if args.int8 else "bf16") + "-chunk4-ffn-experiment" + ("-projection-input" if args.projection_input else ""))
    path.mkdir(parents=True, exist_ok=True)
    design.specialize(projection_input=args.projection_input, quantized=args.int8).compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(json.dumps(dict(
        schema_version=1, batch=4,
        input_layout="projection_residual_pairs" if args.projection_input else "token_major", channels=2048, hidden=8192,
        validation="experimental_requires_hardware_validation",
        weights="int8" if args.int8 else "bfloat16", activation="bfloat16", state="float32",
        fp32_arena_floats=51200, bf16_arena_elements=40960,
        fp32_offsets=dict(shift=0, raw=2048, projected=34816, output=43008),
        bf16_offsets=dict(mixed=0, activated=8192),
        value_activation_ddr_read_bytes=2097152, weight_payload_bytes=34078720 if args.int8 else 67108864,
        **(dict(tile_bytes=4160, scale="fp16_expanded_fp32") if args.int8 else {}),
    )) + "\n")
    print(path, flush=True)
