# SPDX-License-Identifier: Apache-2.0
"""Depth-one input/output ownership mock for in-place fused recurrence."""

from concurrent.futures import ThreadPoolExecutor
import queue
import numpy as np
from rkv_dataflow import Fifo


def prepare(a):
    a = a.copy()
    kk = a[0] * a[3]
    kk /= max(np.linalg.norm(kk), 1e-12)
    alpha = 1 / (1 + np.exp(-a[1] - a[5]))
    a[19], a[20], a[17], a[18] = kk, alpha, -kk, kk * alpha
    a[15] = a[0] * (1 + (alpha - 1) * a[4])
    a[14] = np.exp(-0.6065306597126334 / (1 + np.exp(-a[2] - a[6])))
    return a


def recurrent(s, a):
    a = a.copy()
    s = (
        s * a[14, :, None]
        + a[18, :, None] * (a[17] @ s)[None, :]
        + a[15, :, None] * a[16, None, :]
    )
    y = a[13] @ s
    a[8] = y
    return s, a


def finish(a):
    a = a.copy()
    y = a[8]
    a[21] = (y - y.mean()) / np.sqrt(y.var() + 64e-5) * a[9] + a[10]
    a[22] = (a[13] * a[15] * a[11]).sum() * a[16]
    a[23] = a[21] + a[22]
    a[24] = a[23] * a[12]
    return a


def head(s, a):
    s, a = recurrent(s, prepare(a))
    return s, finish(a)


def main():
    rng = np.random.default_rng(491)
    state = rng.normal(0, 0.125, (32, 64, 64))
    aux = rng.normal(0, 0.125, (27, 2048))
    expected = [head(state[h], aux[:, h * 64 : (h + 1) * 64]) for h in range(32)]
    sf = [Fifo(1) for _ in range(8)]
    af = [Fifo(1) for _ in range(8)]
    prepared = [Fifo(1) for _ in range(8)]
    out = [Fifo(1) for _ in range(8)]
    split_s = [Fifo(1) for _ in range(8)]
    split_a = [Fifo(1) for _ in range(8)]
    final = [Fifo(1) for _ in range(8)]
    ack = [queue.Queue() for _ in range(8)]

    def fill(i):
        for h in range(i * 4, i * 4 + 4):
            sf[i].put(state[h].copy())
            af[i].put(aux[:, h * 64 : (h + 1) * 64].copy())

    def pre(i):
        for _ in range(4):
            a = af[i].acquire()
            prepared[i].reserve()
            value = prepare(a)
            af[i].release()
            prepared[i].publish(value)

    def rec(i):
        for _ in range(4):
            s, a = sf[i].acquire(), prepared[i].acquire()
            out[i].reserve()
            value = recurrent(s, a)
            sf[i].release()
            prepared[i].release()
            out[i].publish(value)

    def split(i):
        for _ in range(4):
            s, a = out[i].acquire()
            split_s[i].put(s)
            split_a[i].put(a)
            ack[i].get(timeout=10)
            ack[i].get(timeout=10)
            out[i].release()

    def post(i):
        for _ in range(4):
            a = split_a[i].acquire()
            final[i].reserve()
            value = finish(a)
            split_a[i].release()
            ack[i].put(True)
            final[i].publish(value)

    def drain_s(i):
        for h in range(i * 4, i * 4 + 4):
            state[h] = split_s[i].acquire()
            split_s[i].release()
            ack[i].put(True)

    def drain_a(i):
        for h in range(i * 4, i * 4 + 4):
            aux[:, h * 64 : (h + 1) * 64] = final[i].acquire()
            final[i].release()

    with ThreadPoolExecutor(max_workers=56) as pool:
        tasks = [
            pool.submit(fn, i)
            for i in range(8)
            for fn in [fill, pre, rec, split, post, drain_s, drain_a]
        ]
        for t in tasks:
            t.result(timeout=30)
    for h, (s, a) in enumerate(expected):
        np.testing.assert_array_equal(state[h], s)
        np.testing.assert_array_equal(aux[:, h * 64 : (h + 1) * 64], a)
    print(
        "32 heads: bounded depth-one queues, per-head in-place ownership and all arena slots passed"
    )


if __name__ == "__main__":
    main()
