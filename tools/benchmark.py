"""Comparable ecosystem baseline; run from v0.8.0 or v0.8.1 root.

Usage: python bench_081.py. Timing depends on host; same seed and generation
count must be used for comparisons. Only deterministic state summaries matter.
"""
import hashlib
import statistics
import time
from life_lab.world import Ecosystem


def run_bench() -> None:
    for w, h in ((160,100),(300,180)):
        times=[]
        digest=''
        for _ in range(7):
            world=Ecosystem(w,h,seed=119)
            world.randomize(.15,seed=119)
            start=time.perf_counter()
            for _step in range(12):
                world.step()
            times.append((time.perf_counter()-start)/12*1000)
            state=world.cells.tobytes()+world.energy.tobytes()+world.lineage.ids.tobytes()
            digest=hashlib.sha256(state).hexdigest()[:16]
        print(f"{w}x{h}: median {statistics.median(times):.3f} ms/step; state={digest}")


if __name__ == '__main__':
    run_bench()
