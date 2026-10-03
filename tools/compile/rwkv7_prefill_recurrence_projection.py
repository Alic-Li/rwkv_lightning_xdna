# SPDX-License-Identifier: Apache-2.0
"""Fused paired value recurrence and BF16 output projection, five-BO ABI."""
import json
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import In, InOut, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import KERNEL_ROOT, typ, external

@iron.jit
def design(state: InOut, aux: InOut, first: In, weights: In, residual: In):
    wt = lambda n: np.ndarray[(n,), np.dtype[bfloat16]]
    pre = external("rwkv7_pair_value_prepare", "prefill_recurrence_prepare_fp32.cc",
                   [typ(3840), typ(4096), typ(3456), np.int32], optimization="-Oz")
    rec = external("rwkv7_pair_recurrent", "prefill_recurrence_update_fp32.cc",
                   [typ(4096), typ(3456), typ(4224)], optimization="-Os")
    post = external("rwkv7_pair_finish_projection", "prefill_recurrence_project_finish.cc",
                    [typ(3456), typ(128), typ(640), np.int32], optimization="-Os")
    post_short = post.object_file.bind("rwkv7_pair_finish_projection512", [typ(3456), typ(128), typ(512), np.int32])
    dot = external("rwkv7_recurrence_projection_tile", "prefill_recurrence_projection_bf16.cc",
                   [typ(8192), wt(4096), typ(1024), np.int32, np.int32], optimization="-O3")
    zero = dot.object_file.bind("rwkv7_prefill_output_zero", [typ(1024)])
    add = dot.object_file.bind("rwkv7_prefill_output_int8_finish", [typ(1024), typ(8192), np.int32])
    copies = [dot.object_file.bind(f"rwkv7_copy{n}", [typ(n), typ(n)]) for n in (2560, 1536)]
    sin = [ObjectFifo(typ(n * 4096), name=f"state_input_group_{i}", depth=1) for i, n in enumerate((4, 3))]
    ain = [ObjectFifo(typ(n * 3840), name=f"aux_input_group_{i}", depth=1) for i, n in enumerate((4, 3))]
    ss, aa = [], []
    for i, n in enumerate((4, 3)):
        ss += sin[i].cons().split([j * 4096 for j in range(n)], obj_types=[typ(4096)] * n)
        aa += ain[i].cons().split([j * 3840 for j in range(n)], obj_types=[typ(3840)] * n)
    firsts = ObjectFifo(typ(4096), name="first_values", depth=1)
    pp = [ObjectFifo(typ(3456), name=f"prepared_{i}", depth=1) for i in range(7)]
    oo = [ObjectFifo(typ(4224), name=f"recout_{i}", depth=1) for i in range(7)]
    ys = [f.cons().split([0, 4096], obj_types=[typ(4096), typ(128)]) for f in oo]
    mixed = ObjectFifo(typ(8192), name="projection_inputs", depth=1)
    inputs = mixed.prod().join([0, 2560, 4096], obj_types=[typ(2560), typ(1536), typ(4096)])
    groups = [ObjectFifo(typ(n), name=f"finished_group_{i}", depth=1) for i, n in enumerate((2560, 1536))]
    finish_out = groups[0].prod().join([j * 640 for j in range(4)], obj_types=[typ(640)] * 4)
    finish_out += groups[1].prod().join([j * 512 for j in range(3)], obj_types=[typ(512)] * 3)
    ws = [ObjectFifo(wt(4096), name=f"projection_weight_{i}", depth=2) for i in range(8)]
    out = [ObjectFifo(typ(1024), name=f"projection_output_{i}", depth=1) for i in range(8)]

    def prepare(a, first, p, f, heads, offset, step):
        fv = first.acquire(1)
        for h in range_(heads):
            av, pv = a.acquire(1), p.acquire(1)
            f(av, fv, pv, offset + h * step)
            a.release(1)
            p.release(1)
        first.release(1)

    def update(s, p, o, f, heads):
        for _ in range_(heads):
            sv, pv, ov = s.acquire(1), p.acquire(1), o.acquire(1)
            f(sv, pv, ov)
            s.release(1)
            p.release(1)
            o.release(1)

    def finish(p, y, out, f, heads):
        ov = out.acquire(1)
        for h in range_(heads):
            pv, yv = p.acquire(1), y.acquire(1)
            f(pv, yv, ov, h)
            p.release(1)
            y.release(1)
        out.release(1)

    def projection(x, w, y, f, z, add, core):
        xv, yv = x.acquire(1), y.acquire(1)
        z(yv)
        for row in range_(16):
            for col in range_(8):
                wv = w.acquire(1)
                f(xv, wv, yv, row, col)
                w.release(1)
        add(yv, xv, core)
        x.release(1)
        y.release(1)

    def copy(x, y, f):
        xv, yv = x.acquire(1), y.acquire(1)
        f(xv, yv)
        x.release(1)
        y.release(1)

    workers = []
    for i in range(7):
        heads, offset, step = (5, i, 4) if i < 4 else (4, 20 + i - 4, 3)
        workers += [Worker(prepare, [aa[i].cons(), firsts.cons(), pp[i].prod(), pre, heads, offset, step], stack_size=12288),
                    Worker(update, [ss[i].cons(), pp[i].cons(), oo[i].prod(), rec, heads], stack_size=12288),
                    Worker(finish, [pp[i].cons(), ys[i][1].cons(), finish_out[i].prod(), post if i < 4 else post_short, heads], stack_size=12288)]
    workers += [Worker(copy, [groups[i].cons(), inputs[i].prod(), copies[i]], stack_size=1024) for i in range(2)]
    workers += [Worker(projection, [mixed.cons(), ws[i].cons(), out[i].prod(), dot, zero, add, i], stack_size=9216) for i in range(8)]

    def seq(state, aux, first, weights, residual, hs, ha, hn, hf, hw, hr, ho):
        hf.fill(first, tap=TAP((57344,), 0, [1, 1, 2, 2048], [0, 0, 55296, 1]))
        hr.fill(residual)
        for i, (offset, heads) in enumerate(((0, 20), (20, 12))):
            hs[i].fill(state, tap=TAP((131072,), offset * 4096,
                [1, 1, 1, heads * 4096], [0, 0, 0, 1]))
            ha[i].fill(aux, tap=TAP((122880,), offset * 64,
                [heads, 2, 30, 64], [64, 61440, 2048, 1]))
        for i in range(7):
            heads, offset, step = (5, i, 4) if i < 4 else (4, 20 + i - 4, 3)
            hn[i].drain(state, tap=TAP((131072,), offset * 4096,
                [1, 1, heads, 4096], [0, 0, step * 4096, 1]), wait=True)
        for i in range(8):
            hw[i].fill(weights, tap=TAP((4194304,), i * 524288, [1, 1, 1, 524288], [0, 0, 0, 1]))
            ho[i].drain(aux, tap=TAP((122880,), 25 * 2048 + i * 256,
                [1, 2, 2, 256], [0, 61440, 2048, 1]), wait=True)

    return Program(iron.get_current_device(), Runtime(seq, [typ(131072), typ(122880),
        typ(57344), wt(4194304), typ(4096), [f.prod() for f in sin], [f.prod() for f in ain],
        [f[0].cons() for f in ys], firsts.prod(), [f.prod() for f in ws], inputs[2].prod(),
        [f.cons() for f in out]]), workers=workers).resolve_program()

if __name__ == "__main__":
    path = KERNEL_ROOT / "bf16-prefill-recurrence-projection-b2"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    (path / "config.json").write_text(json.dumps(dict(schema_version=1, batch=2,
        channels=2048, head_size=64, arena_vectors=30, lanes=7, cores=31,
        output_vectors=[25, 26], dtype="bfloat16", state_dtype="float32")) + "\n")
    print(path, flush=True)
