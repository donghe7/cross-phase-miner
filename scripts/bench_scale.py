"""
Scale benchmark for the signal server.

Answers the questions a reviewer will ask about "1000+ crossings":
how much memory does the state cost, how fast can observations be
ingested, what does one learner update cost, how does the LRU working set
behave when it is deliberately too small, and how long does building a
city-wide snapshot take.

Run with ``python -m scripts.bench_scale`` (add ``--intersections 20000``
for a stress run).
"""

from __future__ import annotations

import argparse
import random
import resource
import statistics
import tempfile
import time
from pathlib import Path
from typing import List

from crossphase_miner.core.models import PhaseTransition, SignalColor
from server.service import SignalService


def rss_mb() -> float:
    """Resident set size in MiB (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bench(intersections: int, cache: int, db: str) -> None:
    baseline = rss_mb()

    started = time.perf_counter()
    service = SignalService(
        num_intersections=intersections,
        zone_size=min(48, intersections),
        num_hubs=8,
        speed=1.0,
        db_path=db,
    )
    build_seconds = time.perf_counter() - started
    service.learners.capacity = cache
    after_build = rss_mb()

    print(f"crossings                {intersections:,}")
    print(f"build time               {build_seconds * 1000:.0f} ms")
    print(
        f"memory for city state    {after_build - baseline:.1f} MiB "
        f"({(after_build - baseline) * 1024 / max(intersections, 1):.1f} KiB each)"
    )

    # --- ingest throughput (direct calls; HTTP adds its own overhead) ---
    now = service.clock.now()
    ids = service.zone_ids() or service.order[:20]
    batch = [(now + i * 0.5, "RED" if i % 7 else "GREEN", 0.93) for i in range(10)]
    count = 2000
    started = time.perf_counter()
    for i in range(count):
        service.ingest_batch(
            robot_id=f"robot_{i % 20:03d}",
            intersection_id=ids[i % len(ids)],
            observations=batch,
        )
    elapsed = time.perf_counter() - started
    print(
        f"ingest                   {count / elapsed:,.0f} batches/s "
        f"= {count * len(batch) / elapsed:,.0f} observations/s"
    )

    # --- learner update cost ---
    samples: List[float] = []
    T_cycle, T_red = 150.0, 119.0
    for index, iid in enumerate(ids[:8]):
        base = now + index * 1000.0
        for k in range(12):
            timestamp = base + k * T_cycle + T_red
            transition = PhaseTransition(
                intersection_id=iid,
                timestamp=timestamp,
                from_color=SignalColor.RED,
                to_color=SignalColor.GREEN,
                episode_start=timestamp - T_red,
            )
            started = time.perf_counter()
            service._learn(iid, [transition], [("day", T_red)])
            samples.append(time.perf_counter() - started)
    print(
        f"learner update           median {statistics.median(samples) * 1000:.1f} ms, "
        f"p95 {sorted(samples)[int(len(samples) * 0.95)] * 1000:.1f} ms "
        f"-> {1 / statistics.median(samples):.0f} updates/s per core"
    )

    # --- LRU behaviour with a deliberately small cache ---
    service.learners.capacity = 32
    rng = random.Random(7)
    for _ in range(600):
        iid = service.order[rng.randrange(len(service.order))]
        timestamp = now + rng.uniform(0, 3600)
        service._learn(
            iid,
            [
                PhaseTransition(
                    intersection_id=iid,
                    timestamp=timestamp,
                    from_color=SignalColor.RED,
                    to_color=SignalColor.GREEN,
                    episode_start=timestamp - 100.0,
                )
            ],
            [],
        )
    print(
        f"working set (cap 32)     {len(service.learners)} in memory, "
        f"{service.learners.evictions} evicted, "
        f"{service.learners.restores} restored from SQLite, "
        f"{service.learners.hits} hits / {service.learners.misses} misses"
    )

    # --- snapshot cost ---
    timings = []
    for _ in range(20):
        started = time.perf_counter()
        snapshot = service.snapshot()
        timings.append(time.perf_counter() - started)
    payload = len(str(snapshot["tiles"]))
    print(
        f"snapshot                 {statistics.median(timings) * 1000:.1f} ms, "
        f"~{payload / 1024:.0f} KiB of tile data "
        f"({len(snapshot['tiles']):,} crossings)"
    )
    print(f"models stored            {len(service.models):,}")
    print(f"peak memory              {rss_mb():.1f} MiB")

    service.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Signal server scale benchmark")
    parser.add_argument("--intersections", type=int, default=1200)
    parser.add_argument("--cache", type=int, default=4096)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="crossphase-bench-") as folder:
        bench(args.intersections, args.cache, str(Path(folder) / "state.sqlite3"))


if __name__ == "__main__":
    main()
