# SPDX-License-Identifier: Apache-2.0
"""Isolated probe for small producer packets regrouped into one consumer packet."""

import numpy as np
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from rwkv7_full import KERNEL_ROOT, typ, external


def group(count):
    @iron.jit
    def design(x: In, y: Out):
        small = external(
            "rwkv7_stream_copy", "stream_copy_fp32.cc", [typ(2048), typ(2048), np.int32]
        )
        # Bind a second symbol with a packet-sized type via its own wrapper below.
        big = external(
            "rwkv7_packet_copy",
            "stream_copy_fp32.cc",
            [typ(count * 2048), typ(2048), np.int32],
        )
        fi = ObjectFifo(typ(2048), name="input", depth=1)
        mid = ObjectFifo(typ(2048), name="tiles", depth=1)
        packet = mid.cons().forward(obj_type=typ(count * 2048), name="packet", depth=1)
        fo = ObjectFifo(typ(2048), name="output", depth=1)

        def producer(x, y, f):
            for _ in range_(count):
                xv, yv = x.acquire(1), y.acquire(1)
                f(xv, yv, 2048)
                x.release(1)
                y.release(1)

        def consumer(x, y, f):
            xv = x.acquire(1)
            for part in range_(count):
                yv = y.acquire(1)
                f(xv, yv, part)
                y.release(1)
            x.release(1)

        workers = [
            Worker(producer, [fi.cons(), mid.prod(), small], stack_size=12288),
            Worker(consumer, [packet.cons(), fo.prod(), big], stack_size=12288),
        ]

        def seq(x, y, hx, hy):
            hx.fill(x)
            hy.drain(y, wait=True)

        return Program(
            iron.get_current_device(),
            Runtime(seq, [typ(count * 2048), typ(count * 2048), fi.prod(), fo.cons()]),
            workers=workers,
        ).resolve_program()

    return design


if __name__ == "__main__":
    for count in [3, 4]:
        path = KERNEL_ROOT / f"stream-group-{count}"
        path.mkdir(parents=True, exist_ok=True)
        group(count).compile(path / "design.xclbin", path / "instructions.bin")
        print(path, flush=True)
