# SPDX-License-Identifier: Apache-2.0
"""Two-token BF16 ChannelMix: one submission, shared streamed weight tiles."""
import json
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import CompileTime, In, InOut, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import KERNEL_ROOT, typ, external

def bt(n):
    return np.ndarray[(n,), np.dtype[bfloat16]]

@iron.jit
def design(x: In, parameters: In, weights: In, fp32: InOut, bf16: Out,
           *, projection_input: CompileTime[bool] = False):
    packed = ObjectFifo(typ(6144), name="norm_input", depth=1)
    inputs = packed.prod().join([0, 2048, 4096], tile=Tile(0, 1), obj_types=[typ(2048)] * 3)
    normalized = ObjectFifo(typ(2048), name="normalized", depth=1)
    params = ObjectFifo(typ(4096), name="mix_parameters", depth=1)
    joins = params.prod().join([0, 2048], tile=Tile(1, 1), obj_types=[typ(2048)] * 2)
    mix_out = ObjectFifo(bt(4096), name="mixed", depth=1)
    mx = mix_out.cons().forward(tile=Tile(2, 1), name="mixed_fanout")
    final_shift = ObjectFifo(typ(2048), name="final_shift", depth=1)
    norm = external("rwkv7_layer_norm", "norm_fp32.cc", [typ(6144), typ(2048)])
    mix = external("rwkv7_prefill_mix_tokens", "prefill_mix_bf16.cc",
                   [typ(2048), typ(4096), typ(2048), bt(4096), np.int32], optimization="-O3")
    init = mix.object_file.bind("rwkv7_prefill_shift_init", [typ(4096), typ(2048)])
    key = external("rwkv7_prefill_key", "prefill_channel_bf16.cc",
                   [bt(4096), bt(4096), typ(4096), np.int32, np.int32], optimization="-O3")
    value = key.object_file.bind("rwkv7_prefill_value",
                   [bt(16384), bt(4096), typ(1024), np.int32, np.int32])
    zero_key = key.object_file.bind("rwkv7_prefill_zero4096", [typ(4096)])
    zero_value = key.object_file.bind("rwkv7_prefill_zero1024", [typ(1024)])
    relu = key.object_file.bind("rwkv7_prefill_activate4096", [typ(4096), bt(4096)])
    add = key.object_file.bind("rwkv7_prefill_add4096", [typ(4096)] * 3)
    wk = [ObjectFifo(bt(4096), name=f"kw{i}", depth=2) for i in range(4)]
    wv = [ObjectFifo(bt(4096), name=f"vw{i}", depth=2) for i in range(4)]
    raw_out = ObjectFifo(typ(16384), name="raw", depth=1)
    act_out = ObjectFifo(bt(16384), name="activated", depth=1)
    # Contiguous core stripes: [core, token, channel]. Matrix/residual kernels
    # address that layout directly; diagnostic drains scatter to token-major.
    kr = raw_out.prod().join([i * 4096 for i in range(4)], tile=Tile(3, 1),
                           obj_types=[typ(4096)] * 4)
    ka = act_out.prod().join([i * 4096 for i in range(4)], tile=Tile(4, 1),
                           obj_types=[bt(4096)] * 4)
    proj = ObjectFifo(typ(4096), name="projected", depth=1)
    vp = proj.prod().join([i * 1024 for i in range(4)], tile=Tile(6, 1),
                         obj_types=[typ(1024)] * 4)
    residual_x = ObjectFifo(typ(4096), name="residual_x", depth=1)
    final = ObjectFifo(typ(4096), name="final", depth=1)

    def normalize(p, o, f):
        for _ in range_(2):
            pv, ov = p.acquire(1), o.acquire(1)
            f(pv, ov)
            p.release(1)
            o.release(1)
    def mixes(x, p, s, o, initialize, f):
        pv, sv, ov = p.acquire(1), s.acquire(1), o.acquire(1)
        initialize(pv, sv)
        for token in range_(2):
            xv = x.acquire(1)
            f(xv, pv, sv, ov, token)
            x.release(1)
        p.release(1)
        s.release(1)
        o.release(1)
    def keys(x, w, r, a, f, z, activate):
        xv, rv, av = x.acquire(1), r.acquire(1), a.acquire(1)
        z(rv)
        for row in range_(128):
            for col in range_(8):
                weights = w.acquire(1)
                f(xv, weights, rv, row, col)
                w.release(1)
        activate(rv, av)
        x.release(1)
        r.release(1)
        a.release(1)
    def values(x, w, p, f, z):
        xv, pv = x.acquire(1), p.acquire(1)
        z(pv)
        for row in range_(32):
            for col in range_(32):
                weights = w.acquire(1)
                f(xv, weights, pv, row, col)
                w.release(1)
        x.release(1)
        p.release(1)
    def residual(x, p, o, f):
        xv, pv, ov = x.acquire(1), p.acquire(1), o.acquire(1)
        f(xv, pv, ov)
        x.release(1)
        p.release(1)
        o.release(1)
    workers = [
        Worker(normalize, [packed.cons(), normalized.prod(), norm], tile=Tile(0, 2), stack_size=12288),
        Worker(mixes, [normalized.cons(), params.cons(), final_shift.prod(), mix_out.prod(), init, mix],
               tile=Tile(1, 2), stack_size=12288),
    ]
    workers += [Worker(keys, [mx.cons(), wk[i].cons(), kr[i].prod(), ka[i].prod(), key, zero_key, relu],
                       tile=Tile(i, 3), stack_size=12288) for i in range(4)]
    workers += [Worker(values, [act_out.cons(), wv[i].cons(), vp[i].prod(), value, zero_value],
                       tile=Tile(i + 4, 3), stack_size=12288) for i in range(4)]
    workers += [Worker(residual, [residual_x.cons(), proj.cons(), final.prod(), add],
                       tile=Tile(7, 2), stack_size=12288)]
    def seq(x, p, w, f, h, hx, hw, hb, hc, hs, hkw, hvw, hm, hr, ha, hv, hres, hy, ho):
        input_tap = (TAP((8192,), 2048, [2, 1, 1, 2048], [4096, 0, 0, 1])
                     if projection_input else None)
        hx.fill(x, tap=input_tap)
        for handle, offset in ((hw, 0), (hb, 2048)):
            handle.fill(p, tap=TAP((6144,), offset, [2, 1, 1, 2048], [0, 0, 0, 1]))
        hc.fill(p, tap=TAP((6144,), 4096, [1, 1, 1, 2048], [0, 0, 0, 1]))
        hs.fill(f, tap=TAP((26624,), 0, [1, 1, 1, 2048], [0, 0, 0, 1]))
        for i in range(4):
            hkw[i].fill(w, tap=TAP((33554432,), i * 4194304, [1, 1, 1, 4194304], [0, 0, 0, 1]))
            hvw[i].fill(w, tap=TAP((33554432,), 16777216 + i * 4194304, [1, 1, 1, 4194304], [0, 0, 0, 1]))
        hm.drain(h, tap=TAP((20480,), 0, [1, 1, 1, 4096], [0, 0, 0, 1]), wait=True)
        hr.drain(f, tap=TAP((26624,), 2048, [4, 2, 4, 512], [2048, 8192, 512, 1]), wait=True)
        ha.drain(h, tap=TAP((20480,), 4096, [4, 2, 4, 512], [2048, 8192, 512, 1]), wait=True)
        hv.drain(f, tap=TAP((26624,), 18432, [4, 2, 1, 512], [512, 2048, 0, 1]), wait=True)
        hres.fill(x, tap=input_tap)
        hy.drain(f, tap=TAP((26624,), 22528, [1, 1, 1, 4096], [0, 0, 0, 1]), wait=True)
        ho.drain(f, tap=TAP((26624,), 0, [1, 1, 1, 2048], [0, 0, 0, 1]), wait=True)
    return Program(iron.get_current_device(), Runtime(seq, [
        typ(8192 if projection_input else 4096), typ(6144), bt(33554432), typ(26624), bt(20480),
        *[f.prod() for f in inputs], *[f.prod() for f in joins],
        [f.prod() for f in wk], [f.prod() for f in wv],
        mix_out.cons(), raw_out.cons(), act_out.cons(), proj.cons(),
        residual_x.prod(), final.cons(), final_shift.cons(),
    ]), workers=workers).resolve_program()

if __name__ == "__main__":
    for projection_input in (False, True):
        suffix = "-projection-input" if projection_input else ""
        path = KERNEL_ROOT / ("bf16-prefill-ffn-b2" + suffix)
        path.mkdir(parents=True, exist_ok=True)
        design.specialize(projection_input=projection_input).compile(
            path / "design.xclbin", path / "instructions.bin")
        (path / "config.json").write_text(json.dumps(dict(
            schema_version=1, batch=2, channels=2048, hidden=8192,
            weights="bfloat16", activation="bfloat16", state="float32",
            layout="token_major", runs_per_batch=1,
            input_layout="projection_residual_pairs" if projection_input else "token_major",
            fp32_arena_floats=26624, bf16_arena_elements=20480,
            fp32_offsets=dict(shift=0, raw=2048, projected=18432, output=22528),
            bf16_offsets=dict(mixed=0, activated=4096),
        )) + "\n")
        print(path, flush=True)
