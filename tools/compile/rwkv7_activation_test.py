# SPDX-License-Identifier: Apache-2.0
"""Offline scalar/vector ReLU-square comparison; C++ dispatch only."""

import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Worker, Runtime, Program
from rwkv7_common import KERNEL_ROOT, typ, external


@iron.jit
def design(x: In, y: Out):
    fn = external(
        "rwkv7_activation_test",
        "activation_test.cc",
        [typ(2048), typ(4096)],
        optimization="-O3",
    )
    xi = ObjectFifo(typ(2048), name="input", depth=1)
    yo = ObjectFifo(typ(4096), name="output", depth=1)

    def core(x, y, fn):
        a, b = x.acquire(1), y.acquire(1)
        fn(a, b)
        x.release(1)
        y.release(1)

    worker = Worker(core, [xi.cons(), yo.prod(), fn], stack_size=12288)

    def seq(x, y, hx, hy):
        hx.fill(x)
        hy.drain(y, wait=True)

    return Program(
        iron.get_current_device(),
        Runtime(seq, [typ(2048), typ(4096), xi.prod(), yo.cons()]),
        workers=[worker],
    ).resolve_program()


if __name__ == "__main__":
    path = KERNEL_ROOT / "activation-test"
    path.mkdir(parents=True, exist_ok=True)
    design.compile(path / "design.xclbin", path / "instructions.bin")
    print(path, flush=True)
