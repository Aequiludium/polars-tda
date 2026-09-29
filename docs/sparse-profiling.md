# Sparse Rips profiling and first optimization experiment

Date: 2026-09-28. Kernel: `d404282745a4238636ccb83d521eeac952ff7a30`.
Release builds with Rust 1.95.0 on the AMD EPYC 7H12 Linux host used by the initial
measurements. Uniform 3D coordinates, seed 42, epsilon 0.5, unchecked metric
policy, H0/H1 over F2, no cutoff. The production kernel checkout, Git dependency
and plugin dispatch remain unchanged. The candidate is an **unapplied** patch in
`benchmarks/experiments/column-entry.patch`.

## Native phases

`examples/profile_sparse.rs` borrows column-major fixture data, packs row-major
coordinates, prepares sparse Rips, and computes persistence. Its timings exclude
fixture parsing, final diagram text encoding and file IO. Peak RSS is the native
process's Linux VmHWM and includes those operations. No Python/Polars runtime is
loaded in the measured process, so these RSS values differ from plugin benchmarks.
Packing is a native baseline, not a direct measurement of Polars group gathering.

Three fresh-process runs per case, medians:

| Points | Packing (ms) | Preparation (ms) | Persistence (ms) | Total (ms) | Native peak RSS (MiB) |
| --- | --- | --- | --- | --- | --- |
| 256 | 0.006 | 7.79 | 359.08 | 367.05 | 8.35 |
| 512 | 0.013 | 32.24 | 1190.56 | 1222.82 | 20.16 |

Persistent homology accounts for approximately 97–98% of these native timings.
Coordinate packing is not the relevant bottleneck on these fixtures.

A temporary instrumented kernel counted 12,211 coboundary appends and 958,219
cofaces for 256 points. Coface generation plus sparse-column accumulation took
about 349 ms of a 380 ms instrumented persistence call. Instrumentation changes
execution cost; these numbers locate work, not an unbiased performance baseline.
All instrumentation was removed before candidate timing and testing.

## Rejected alternative: explicit expansion

The existing public sparse expansion API preserves the insertion-radius blocker.
Expanding through dimension two gave bitwise-identical persistence intervals, but
increased total time and peak memory:

| Points | Stored simplices | Expansion (ms) | Persistence (ms) | Total (ms) | Native peak RSS (MiB) |
| --- | --- | --- | --- | --- | --- |
| 256 | 326965 | 487.11 | 332.83 | 829.25 | 119.39 |
| 512 | 972878 | 2362.14 | 1390.96 | 3781.90 | 367.70 |

Do not switch the plugin to eager expansion based on these results. An ordinary
flag complex is also not a valid substitute: it discards the sparse blocker.

## Candidate: one tree lookup per coefficient update

The kernel's private `Column::add_term` first looked up a coefficient and then
searched the same BTreeMap again to insert or remove it. The candidate uses the
standard `Entry` API to update or remove through one lookup. Vacant entries still
normalize arbitrary residues and omit zeros. It adds no cache, dependency,
public API or algorithm dispatch, and does not change coface enumeration.

Five fresh-process runs per variant, alternating baseline/candidate order. Total
includes packing, preparation and persistence; fixture data matches the phases
above. The reported reductions are local measurements, not a general speedup
claim:

| Points | Baseline total (ms) | Candidate total (ms) | Reduction | Baseline / candidate RSS (MiB) |
| --- | --- | --- | --- | --- |
| 256 | 364.85 | 291.33 | 20.2% | 8.64 / 8.39 |
| 512 | 1190.14 | 954.59 | 19.8% | 20.21 / 20.40 |

RSS is effectively unchanged at this measurement granularity. Candidate and
baseline interval dimensions, multiplicities and floating endpoints matched
bit-for-bit in every run. The candidate still does not close the performance gap
with the specialized exact H1 path. No optimized plugin wheel has been shipped or
installed: these are native-kernel experiments in a temporary source copy.

Validation in the temporary patched kernel: 188 existing release tests and
doctests passed; an additional residue/cancellation regression passed for F2,
F3, F5 and the largest u32 prime. All-target/all-feature Clippy passed. The shared
Column implementation serves multiple reducers, so representative production
workloads should be benchmarked before adopting the patch upstream.

## Reproduce and continue

From polars-tda, build the diagnostic against the unchanged pinned dependency:

```sh
cargo build --release --locked --example profile_sparse
uv run --no-sync python benchmarks/profile_sparse.py --points 256
uv run --no-sync python benchmarks/profile_sparse.py --points 512
```

The Python driver generates the same column-major seeded fixture as `rips.py`,
runs each case in a fresh native process, and requires identical interval output
across modes and binaries. It prints raw measurements and medians as JSON. It
requires Linux for VmHWM, and does not require perf or GNU time.

For a kernel candidate, compile this same Rust worker against separate baseline
and patched disposable copies of the pinned kernel, using identical release
flags. Apply `benchmarks/experiments/column-entry.patch` only to the candidate.
Then pass the two worker paths:

```sh
uv run --no-sync python benchmarks/profile_sparse.py --points 512 --repeats 5 \
  --modes implicit --binary /path/to/baseline-worker \
  --compare-binary /path/to/candidate-worker
```

The next experiment should focus on repeated coface generation/accumulation in
the generic reducer, while preserving the blocker, field arithmetic, ordering,
censoring and resource contracts. Any new cache must be measured against peak
memory. A production dependency update requires a separately reviewed kernel
commit; the plugin keeps its current Git pin until that exists.
