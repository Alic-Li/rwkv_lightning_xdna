# SPDX-License-Identifier: Apache-2.0
"""Two sequential FP32 recurrence updates with one state DMA round trip."""
import json
import numpy as np
import aie.iron as iron
from aie.iron import In, InOut, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern as TAP
from rwkv7_common import KERNEL_ROOT, typ, external


def program(with_value):
    counts = [5, 5, 5, 5, 4, 4, 4] if with_value else [4] * 8
    lanes = len(counts)
    input_size = 1920 if with_value else 1728
    stride = input_size * 32
    pre = external("rwkv7_pair_value_prepare" if with_value else "rwkv7_pair_prepare",
        "prefill_recurrence_prepare_fp32.cc",
        [typ(2 * input_size), typ(4096), typ(3456), np.int32] if with_value else
        [typ(3456), typ(3456)], optimization="-Oz")
    rec = external("rwkv7_pair_recurrent", "prefill_recurrence_update_fp32.cc",
                   [typ(4096), typ(3456), typ(4224)], optimization="-Os")
    post = external("rwkv7_pair_finish", "prefill_recurrence_finish_fp32.cc",
                    [typ(3456), typ(128), typ(3456)], optimization="-Os")
    first = ObjectFifo(typ(4096), name="first", depth=1) if with_value else None
    ss = [ObjectFifo(typ(4096), name=f"pair_state_{i}", depth=1) for i in range(lanes)]
    aa = [ObjectFifo(typ(2 * input_size), name=f"pair_aux_{i}", depth=1) for i in range(lanes)]
    pp = [ObjectFifo(typ(3456), name=f"pair_prepared_{i}", depth=1) for i in range(lanes)]
    oo = [ObjectFifo(typ(4224), name=f"pair_recout_{i}", depth=1) for i in range(lanes)]
    yy = [f.cons().split([0, 4096], obj_types=[typ(4096), typ(128)]) for f in oo]
    ff = [ObjectFifo(typ(3456), name=f"pair_final_{i}", depth=1) for i in range(lanes)]

    def prepare(a, p, f, heads):
        for _ in range_(heads):
            av, pv = a.acquire(1), p.acquire(1)
            f(av, pv)
            a.release(1)
            p.release(1)

    def prepare_value(a, first, p, f, heads, offset):
        fv = first.acquire(1)
        for h in range_(heads):
            av, pv = a.acquire(1), p.acquire(1)
            f(av, fv, pv, h + offset)
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
        for _ in range_(heads):
            pv, yv, ov = p.acquire(1), y.acquire(1), out.acquire(1)
            f(pv, yv, ov)
            p.release(1)
            y.release(1)
            out.release(1)

    workers = []
    for i in range(lanes):
        workers += [Worker(prepare_value, [aa[i].cons(), first.cons(), pp[i].prod(), pre,
                     counts[i], sum(counts[:i])], stack_size=12288) if with_value else
                    Worker(prepare, [aa[i].cons(), pp[i].prod(), pre, counts[i]], stack_size=12288),
                    Worker(update, [ss[i].cons(), pp[i].cons(), oo[i].prod(), rec, counts[i]], stack_size=12288),
                    Worker(finish, [pp[i].cons(), yy[i][1].cons(), ff[i].prod(), post, counts[i]], stack_size=12288)]

    def schedule(state, aux, hs, ha, hn, ho, first=None, hf=None):
        if with_value:
            hf.fill(first, tap=TAP((57344,), 0, [1, 1, 2, 2048], [0, 0, 55296, 1]))
        for i in range(lanes):
            offset = sum(counts[:i])
            st = TAP((131072,), offset * 4096, [1, 1, 1, counts[i] * 4096], [0, 0, 0, 1])
            hs[i].fill(state, tap=st)
            ha[i].fill(aux, tap=TAP((2 * stride,), offset * 64,
                [counts[i], 2, input_size // 64, 64], [64, stride, 2048, 1]))
            hn[i].drain(state, tap=st, wait=True)
            ho[i].drain(aux, tap=TAP((2 * stride,), offset * 64,
                [counts[i], 2, 27, 64], [64, stride, 2048, 1]), wait=True)

    def seq(state, aux, hs, ha, hn, ho):
        schedule(state, aux, hs, ha, hn, ho)

    def seq_value(state, aux, first, hs, ha, hn, ho, hf):
        schedule(state, aux, hs, ha, hn, ho, first, hf)

    return Program(iron.get_current_device(), Runtime(seq_value if with_value else seq,
        [typ(131072), typ(2 * stride), *([typ(57344)] if with_value else []),
         [f.prod() for f in ss], [f.prod() for f in aa], [f[0].cons() for f in yy],
         [f.cons() for f in ff], *([first.prod()] if with_value else [])]), workers=workers).resolve_program()

@iron.jit
def regular(state: InOut, aux: InOut):
    return program(False)

@iron.jit
def value(state: InOut, aux: InOut, first: In):
    return program(True)

if __name__ == "__main__":
    for fused, design in [(False, regular), (True, value)]:
        path = KERNEL_ROOT / ("prefill-value-recurrence-b2" if fused else "prefill-recurrence-b2")
        path.mkdir(parents=True, exist_ok=True)
        design.compile(path / "design.xclbin", path / "instructions.bin")
        (path / "config.json").write_text(json.dumps(dict(schema_version=1, batch=2,
            channels=2048, head_size=64, arena_vectors=30 if fused else 27,
            fused_value=fused, dtype="float32", lanes=7 if fused else 8,
            first_stride=55296 if fused else 0)) + "\n")
        print(path, flush=True)
