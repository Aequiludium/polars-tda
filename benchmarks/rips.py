"""Run one method per fresh process; report latency and whole-process peak RSS."""

import argparse
import json
import platform
import random
import resource
import sys
from time import perf_counter

import polars as pl
import polars_tda as tda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["exact", "sparse"], required=True)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--max-dimension", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.points < 0 or args.dimensions <= 0 or args.max_dimension < 0:
        parser.error("points/max-dimension must be nonnegative; dimensions positive")
    if sys.platform not in ("linux", "darwin"):
        parser.error("RSS conversion is supported on Linux and macOS")

    rng = random.Random(args.seed)
    frame = pl.DataFrame(
        {
            f"x{i}": [rng.random() for _ in range(args.points)]
            for i in range(args.dimensions)
        },
        schema={f"x{i}": pl.Float64 for i in range(args.dimensions)},
    )
    options = {"method": args.method, "max_dimension": args.max_dimension}
    if args.method == "sparse":
        options.update(epsilon=args.epsilon, metric_policy="unchecked")
    expression = tda.rips(*frame.columns, **options).alias("diagram")
    started = perf_counter()
    diagrams = frame.select(expression)
    diagram_seconds = perf_counter() - started
    started = perf_counter()
    diagrams.select(tda.lifetime_summary("diagram", dimension=args.max_dimension))
    feature_seconds = perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = rss * 1024 if sys.platform == "linux" else rss
    # Read nested results after measuring, so Python object conversion is excluded.
    interval_count = diagrams.select(
        pl.col("diagram").struct.field("intervals").list.len()
    ).item()
    print(
        json.dumps(
            {
                "method": args.method,
                "points": args.points,
                "dimensions": args.dimensions,
                "max_dimension": args.max_dimension,
                "epsilon": args.epsilon if args.method == "sparse" else None,
                "metric_policy": "unchecked" if args.method == "sparse" else None,
                "seed": args.seed,
                "diagram_seconds": diagram_seconds,
                "feature_seconds": feature_seconds,
                "peak_process_rss_bytes": peak_rss_bytes,
                "interval_count": interval_count,
                "python": platform.python_version(),
                "polars": pl.__version__,
                "polars_tda": tda.__version__,
                "platform": platform.platform(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
