# polars-tda

[![CI](https://github.com/Aequiludium/polars-tda/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Aequiludium/polars-tda/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/Aequiludium/polars-tda?include_prereleases)](https://github.com/Aequiludium/polars-tda/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Status: pre-alpha](https://img.shields.io/badge/Status-pre--alpha-orange.svg)](docs/releases.md)

[Documentation](docs/README.md) · [Usage guide](docs/usage.md) ·
[Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) ·
[Issues](https://github.com/Aequiludium/polars-tda/issues)

Topology features for Polars data analysis, computed natively by the
[cocycle Rust kernel](https://github.com/Aequiludium/cocycle-rs).

Select coordinate columns and choose **exact** or **approximate** computation.
Each point cloud becomes one row of numeric features, ready for joins, plotting
or a model feature table. This is a pre-alpha package; no PyPI release is published.

See the [documentation index](docs/README.md) for user guides, API contracts and
design decisions. The [usage guide](docs/usage.md) includes numeric preparation,
grouped features and reuse of results.

## Installation

The GitHub [v0.1.0 source preview](https://github.com/Aequiludium/polars-tda/releases/tag/v0.1.0)
is a **pre-release**, not a stable API or a PyPI distribution. With Git, uv,
Python 3.10+ and rustup installed, build the pinned source locally:

```sh
git clone --branch v0.1.0 https://github.com/Aequiludium/polars-tda.git
cd polars-tda
uv sync --locked --no-install-project
uv run --no-sync maturin develop --release --locked
```

Run the examples with the resulting uv environment. Rustup selects the pinned
Rust toolchain. This preview provides source archives, not a cross-platform wheel
set; see [release status and compatibility](docs/releases.md). For contribution
work, clone `main` instead and follow [development setup](CONTRIBUTING.md#development-setup).

## Quick start

Each row is a point. Each group is a cloud:

```python
import polars as pl
import polars_tda as tda

points = pl.DataFrame(
    {
        "sample": ["square"] * 4,
        "x": [0.0, 1.0, 1.0, 0.0],
        "y": [0.0, 0.0, 1.0, 1.0],
    }
)

result = (
    points.group_by("sample")
    .agg(tda.features("x", "y").alias("topology"))
    .unnest("topology")
)
```

For a single cloud, use `points.select(tda.features("x", "y").alias("topology"))`
and unnest the result. The same expressions work with LazyFrame and streaming
execution. `method` is the only computation option:

```python
tda.features("x", "y", method="exact")  # Default
tda.features("x", "y", method="approximate")
```

Approximate computation can change feature values. It is not a speed guarantee:
exact remains the default and is faster on the small clouds measured so far.
No automatic switch, sampling choice or memory-layout option is exposed.

## What the result contains

Ten numeric fields summarize connectivity (`h0_*`) and loop structure (`h1_*`):

| Suffix | Meaning |
| --- | --- |
| finite_count | Number of intervals that end at a finite scale |
| total_persistence | Sum of their lifetimes |
| max_persistence | Longest finite lifetime |
| entropy | Entropy of normalized finite lifetimes, in nats |
| essential_count | Intervals that never end, excluded from lifetime statistics |

When there are no finite intervals, finite count and total are zero; maximum and
entropy are null. These are meaningful missing values, not computation failures.
For the square above, `h1_finite_count` is 1 and `h1_total_persistence` is
`sqrt(2) - 1`. Counts summarize intervals across scales, not at one fixed radius.
See the [feature dictionary and interpretation](docs/features.md).

Coordinates must be finite, non-null Float64. Cast other numeric columns explicitly
with normal Polars expressions. Distances use the supplied coordinate units;
no standardization is applied. Comparable columns and consistent preprocessing
across groups matter more than an internal algorithm setting.

The implementation computes persistence once per cloud and derives both feature
summaries in Rust. It does not export or reconstruct an intermediate diagram
column. Compatible coordinates are borrowed; approximation may pack coordinates.
Distance data, algorithm workspaces and outputs still allocate memory.

See the [expression contract](docs/expression-contract.md) for selector expansion,
aggregation/window behavior, naming and stable result conventions.

## Advanced workflows

The lower-level `rips`, `betti_curve` and `lifetime_summary` functions remain
available for stored diagrams and custom analyses. See [advanced APIs](docs/advanced-api.md)
and the [diagram schema](docs/result-schema.md). Their options do not expand the
analyst-facing `features(*coordinates, method=...)` interface.

## Development

Use Rust 1.95 and Python 3.10 or later. The tested compatibility target is Python
Polars 1.44.2, Rust Polars 0.55.1 (core 0.55.2), pyo3-polars 0.28.0 and PyO3 0.29.0.
The manifests and locks pin the plugin ABI. Broaden versions only after testing.

```sh
uv sync --locked --no-install-project
uv run --no-sync maturin develop --release --locked
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
cargo fmt --all -- --check
cargo clippy --locked --lib --example profile_sparse -- -D warnings
```

Build with `uv run --no-sync maturin build --release --locked --out dist`.
Install and test the resulting wheel before release. CI does not publish.

## Reproducible performance measurements

Run each method in a fresh process after installing a release build:

```sh
uv run --no-sync python benchmarks/rips.py --method exact --points 256
uv run --no-sync python benchmarks/rips.py --method sparse --points 256 --epsilon 0.5
```

The script uses identical seeded uniform clouds. JSON output records diagram and
feature latency separately, whole-process peak RSS, interval counts and versions.
RSS includes Python, Polars, input data and the plugin; it is not incremental
kernel allocation. Diagram timing includes the first native plugin load. Feature
timing includes diagram validation and descriptor work. Results are workload and
machine dependent, and the two methods compute different filtrations. Repeat
fresh-process runs for multiple sizes and representative data before drawing
performance conclusions. The benchmark uses `unchecked` for sparse, never the
cubic metric check. No benchmark dependency is added to the package.

[Initial measurements](docs/benchmark-notes.md) show that sparse is slower on the
tested 256/512-point uniform 3D clouds. Exact remains the default; evaluate sparse
on representative workloads rather than assuming a speed or memory improvement.

Native phase measurements and a tested, unapplied kernel candidate are recorded
in [Sparse profiling](docs/sparse-profiling.md). The experiment reduced native
time by about 20% on these fixtures; production dependencies remain unchanged.

## Kernel and project boundaries

`Cargo.toml` references the kernel GitHub repository with `git` and a full `rev`.
Cargo compiles that source into the plugin: no submodule or source copy is needed.
To upgrade, review a kernel commit, update `rev`, run `cargo update -p cocycle`,
and rebuild/test before committing both manifest and lockfile.

Python constructs expressions and validates options. Rust borrows coordinate
buffers, invokes the kernel, and serializes/deserializes diagram columns. All
homology and descriptor algorithms remain in cocycle; this project requires no
changes to its APIs. New kernel algorithms do not automatically become Polars
APIs; adapters are added here only when a public operation is supported.

See [design rationale and contributor boundaries](docs/design-rationale.md) and
[scope and next decisions](docs/api-discussion.md). Source and docs use English.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, project boundaries, checks and
pull requests. Use the [issue tracker](https://github.com/Aequiludium/polars-tda/issues)
for bugs, questions and proposed workflows.

## References and license

- [Polars expression plugins](https://docs.pola.rs/user-guide/plugins/expr_plugins/)
- [Maturin mixed project layout](https://www.maturin.rs/project_layout.html)

[MIT](LICENSE). The cocycle dependency is independently MIT licensed; see
[third-party notices](THIRD_PARTY_NOTICES.md).
