# SPDX-License-Identifier: Apache-2.0
"""One representative full-head stripe: resident input and depth-two tiles."""

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rkv_dataflow import Fifo


def main():
    x = np.arange(2048, dtype=np.float32) % 7 / 16
    w = np.arange(4096, dtype=np.float32).reshape(16, 256) % 5 / 256
    xf, wf, yf = Fifo(1), Fifo(2), Fifo(2)
    result = np.empty((512, 16), np.float32)

    def fill():
        xf.put(x)
        for _ in range(512 * 8):
            wf.put(w)

    def core():
        xv = xf.acquire()
        for _ in range(512):
            yf.reserve()
            out = np.zeros(16, np.float32)
            for k in range(8):
                v = wf.acquire()
                out += (v * xv[k * 256 : (k + 1) * 256]).sum(axis=1)
                wf.release()
            yf.publish(out)
        xf.release()

    def drain():
        for row in range(512):
            result[row] = yf.acquire()
            yf.release()

    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(fn) for fn in [fill, core, drain]]
        for t in tasks:
            t.result(timeout=30)
    expected = np.tile(w, (1, 8)).astype(np.float64) @ x
    np.testing.assert_array_equal(result, np.broadcast_to(expected, result.shape))
    print(
        "Head stripe: input held through 512 row tiles, 4096 depth-two weight tiles, oracle passed"
    )


if __name__ == "__main__":
    main()
