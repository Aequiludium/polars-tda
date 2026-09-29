# Diagram schema version 2

`rips` returns one Struct per cloud. The same dtype is known at lazy planning
time, including for empty inputs and empty interval lists. The diagram-consuming
`betti_curve` and `lifetime_summary` expressions require this dtype and version.
The main `features` expression consumes coordinate columns instead. Diagram
readers also accept the original version-1 dtype
(without max_filtration_value and approximation). Version 1 and 2 columns cannot
be concatenated directly because their Struct dtypes differ. Version-2 exact and
sparse results share one dtype and can be concatenated.

| Field | Polars dtype | Meaning |
| --- | --- | --- |
| schema_version | UInt32 | 2 |
| intervals | List(Struct) | Intervals described below |
| max_dimension | UInt32 | Every dimension 0 through this value was computed |
| complete | Boolean | Whether the full filtration is known |
| through | Float64, nullable | Inclusive coverage cutoff, null when complete |
| n_points | UInt64 | Input rows, including duplicates |
| n_coordinates | UInt32 | Coordinate columns |
| coefficient | UInt32 | Prime-field characteristic |
| metric | String | `euclidean` |
| method | String | `exact` or `sparse` |
| backend_input | String | `distance_matrix`, `distance_callback`, or `points` |
| max_edge_length | Float64, nullable | Requested exact edge-length cutoff |
| max_filtration_value | Float64, nullable | Requested sparse modified-filtration cutoff |
| approximation | Struct, nullable | Sparse metadata below; null for exact |

`backend_input` records how the unchanged Rust kernel received its data. It is
provenance, not an option users must select. `metric` states the mathematical
distance actually computed by the plugin. `max_edge_length` records the request,
whereas `complete`/`through` record the kernel's actual coverage: a cutoff at or
above the input diameter can certify completeness for exact Rips. Sparse
coverage and all derived features describe the approximate filtration. Its
max_filtration_value records the modified scale cap independently of coverage.

Each interval has `dimension: UInt32`, `birth: Float64`, `death: Float64`
(nullable), and `end: String`:

| end | death | Interpretation |
| --- | --- | --- |
| finite | finite value greater than birth | Alive on `[birth, death)` |
| essential | null | Never dies in the complete filtration |
| censored | null | Alive through the global `through` cutoff, eventual death unknown |

Censored intervals include the cutoff itself, even when birth equals cutoff.
They must not be converted into finite deaths or assumed essential. Null death
alone is insufficient to distinguish these cases. Multiplicity is preserved;
zero-length finite intervals are omitted by the kernel.

An empty interval list in a computed dimension is a valid zero result. A dimension
above `max_dimension` is uncomputed and descriptors reject it, including for an
empty Betti grid. Descriptor reads validate the interval and coverage contract;
editing provenance metadata is not a way to certify a diagram's origin.

`lifetime_summary` produces a Struct with `finite_count: UInt64`,
`total_persistence: Float64`, `max_persistence: Float64` (nullable),
`entropy: Float64` (nullable), `excluded_essential_count: UInt64`, and
`excluded_censored_count: UInt64`. Entropy is Shannon entropy of normalized finite
lifetimes in nats. One finite interval has entropy zero. No finite intervals give
count/total zero and maximum/entropy null. A null input diagram gives a null Struct.


## Approximation metadata

These values come directly from the kernel's result, including when the diagram
has no intervals. Original vertex IDs are row indices within each input group.

| Field | Polars dtype | Meaning |
| --- | --- | --- |
| epsilon | Float64 | Sparsification parameter |
| metric_validation | String | checked, assumed, or unchecked |
| bound_factor | Float64, nullable | Conditional ideal-arithmetic multiplicative factor |
| bound_target | String, nullable | original_input or retained_subset, paired with factor |
| permutation | List(UInt64) | Complete deterministic farthest-point permutation |
| insertion_radii | List(Float64) | Aligned with permutation; initial null means infinity |
| retained_vertices | List(UInt64) | Compact vertex index to original row ID |
| min_insertion_radius | Float64 | Native sampling threshold, currently fixed at zero |
| covering_radius | Float64 | Maximum original-to-retained nearest distance |

The factor is present only when epsilon < 1 and metric validation is checked or
assumed. It relies on the metric hypothesis and ideal arithmetic; it does not
bound floating-point error. No bound is reported for unchecked inputs or epsilon
>= 1. An original_input target identifies zero-distance duplicates. The current
API exposes no positive-radius subsampling, but serializes the actual kernel
metadata rather than inferring a target from options.
