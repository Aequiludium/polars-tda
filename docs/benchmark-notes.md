# Initial performance observations

Local release-wheel measurements on 2026-09-28. CPU: AMD EPYC 7H12; Linux x86_64,
Python 3.13.3, Polars 1.44.2, polars-tda 0.1.0; cocycle revision
`d404282745a4238636ccb83d521eeac952ff7a30`.

Uniform 3D points in [0,1), seed 42, H0/H1 over F2, no scale cutoff. Sparse uses
epsilon 0.5 and unchecked metric policy. Each row reports medians of three fresh
processes using the release build and `benchmarks/rips.py`.

| Points | Method | Diagram time (ms) | Feature time (ms) | Peak process RSS (MiB) | Intervals |
| --- | --- | --- | --- | --- | --- |
| 256 | exact | 23.6 | 0.670 | 59.4 | 373 |
| 256 | sparse | 363.9 | 0.904 | 63.9 | 370 |
| 512 | exact | 103.4 | 0.853 | 67.6 | 750 |
| 512 | sparse | 1181.6 | 0.884 | 74.3 | 747 |

For these small clouds, sparse is slower and whole-process peak RSS is higher.
Avoiding a distance cache alone does not establish an end-to-end improvement.
Exact remains the default; there is no size-triggered automatic approximation.
The different interval counts are expected because the filtrations differ.
These runs do not identify the bottleneck or establish behavior for larger,
higher-dimensional, clustered, thresholded or grouped workloads.

Diagram time includes the first native plugin load. Feature time is one lifetime
summary of H1, including diagram decoding. RSS includes Python/Polars/input data,
plugin code, native working memory and outputs. It is not incremental kernel RSS.
No statistical significance or general speedup is claimed from three runs.

To reproduce, repeat each command three times in separate processes, then repeat
with 512 points. Build and install a release wheel first.

```sh
uv run --no-sync python benchmarks/rips.py --method exact --points 256
uv run --no-sync python benchmarks/rips.py --method sparse --points 256 --epsilon 0.5
```

The next performance task should profile native preparation and reduction on
representative datasets before changing algorithm dispatch or kernel APIs.
