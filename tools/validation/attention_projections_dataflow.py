# SPDX-License-Identifier: Apache-2.0
"""Shared-input RKV and 2/4-worker low-rank groups; no cross-group FIFO edges."""

import queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rkv_dataflow import Fifo
from rank_batch_dataflow import main as rank


def main():
    rng = np.random.default_rng(173)
    x = rng.integers(-4, 5, (3, 2048)).astype(np.float32) / 16
    w = rng.integers(-4, 5, (3, 2048, 2048)).astype(np.float32) / 256
    xs = [Fifo(1) for _ in range(8)]
    ws = [Fifo(2) for _ in range(8)]
    ys = [Fifo(2) for _ in range(8)]
    ack = queue.Queue()
    out = np.empty((3, 2048), np.float32)

    def broadcast():
        for p in range(3):
            for q in xs:
                q.put(x[p])
            for _ in range(8):
                ack.get(timeout=10)

    def feed(i):
        for p in range(3):
            for row in range(16):
                for col in range(8):
                    ws[i].put(
                        w[
                            p,
                            i * 256 + row * 16 : i * 256 + (row + 1) * 16,
                            col * 256 : (col + 1) * 256,
                        ]
                    )

    def core(i):
        for p in range(3):
            xv = xs[i].acquire()
            for row in range(16):
                ys[i].reserve()
                y = np.zeros(16, np.float32)
                for col in range(8):
                    v = ws[i].acquire()
                    y += (v * xv[col * 256 : (col + 1) * 256]).sum(axis=1)
                    ws[i].release()
                ys[i].publish(y)
            xs[i].release()
            ack.put(True)

    def drain(i):
        for p in range(3):
            for row in range(16):
                out[p, i * 256 + row * 16 : i * 256 + (row + 1) * 16] = ys[i].acquire()
                ys[i].release()

    with ThreadPoolExecutor(max_workers=25) as pool:
        tasks = [pool.submit(broadcast)] + [
            pool.submit(fn, i) for i in range(8) for fn in [feed, core, drain]
        ]
        for t in tasks:
            t.result(timeout=30)
    expected = np.einsum("prk,pk->pr", w.astype(np.float64), x.astype(np.float64))
    np.testing.assert_array_equal(out, expected)
    for count in [3, 4]:
        rank(count, first_cores=2, second_cores=4)
    print(
        "RKV shared resident input: all broadcast ownership, tile counts and oracle passed"
    )


if __name__ == "__main__":
    main()
