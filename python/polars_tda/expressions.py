"""Native Polars expressions for point clouds and persistence diagrams."""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import polars as pl
from polars.plugins import register_plugin_function

from polars_tda import _internal


def _uint32(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= 2**32 - 1:
        raise ValueError(f"{name} must fit in UInt32")
    return value


def features(
    *coordinates: str | pl.Expr,
    method: Literal["exact", "approximate"] = "exact",
) -> pl.Expr:
    """Aggregate a point cloud into ten numeric topology features.

    Use select for one cloud or group_by().agg() for one cloud per group, then
    unnest the returned Struct. Coordinates must be finite, non-null Float64.
    No scaling or normalization is applied; choose comparable coordinate units.

    Multi-column expressions/selectors expand into coordinates of one cloud.
    Strings mean column names; use features(*names) for a Python list of names.

    Each of h0 (connected components) and h1 (loops) has finite_count,
    total_persistence, max_persistence, entropy and essential_count. Counts and
    total are zero when there are no finite intervals; maximum/entropy are null.
    Essential intervals are counted separately and excluded from lifetime stats.
    Entropy uses natural logarithms. Lifetimes use Euclidean edge-length units.

    Exact is the default. Approximate uses a fixed sparse Rips preset and can
    change feature values; it is not guaranteed to be faster. The current preset
    uses epsilon 0.5 without metric certification. Both modes compute H0/H1 over
    F2 across the full filtration. Record the package version with your features
    for reproducibility; preset details are documented, not extra user options.

    Persistence is computed once per cloud. Both summaries are derived directly
    in Rust without materializing an intermediate diagram column.
    """
    if not coordinates:
        raise ValueError("features requires at least one coordinate column")
    if method not in ("exact", "approximate"):
        raise ValueError("method must be 'exact' or 'approximate'")
    return register_plugin_function(
        plugin_path=Path(_internal.__file__),
        function_name="features",
        args=list(coordinates),
        kwargs={"method": method},
        is_elementwise=False,
        returns_scalar=True,
        input_wildcard_expansion=True,
    )


def rips(
    *coordinates: str | pl.Expr,
    max_dimension: int = 1,
    max_edge_length: float | None = None,
    coefficient: int = 2,
    method: Literal["exact", "sparse"] = "exact",
    epsilon: float | None = None,
    metric_policy: Literal["check", "assume", "unchecked"] | None = None,
    max_filtration_value: float | None = None,
) -> pl.Expr:
    """Aggregate Float64 coordinates into an exact or sparse Rips diagram.

    Each input row is a point. Use select for one cloud or group_by().agg() for
    multiple clouds. Nulls and nonfinite coordinates are rejected. Cast other
    numeric dtypes explicitly. Exact computation borrows coordinate buffers.
    Sparse computation packs O(n*d) row-major values when borrowing is impossible,
    avoiding the kernel callback path's O(n*n) distance cache.

    Compute H0 through max_dimension over the prime field of characteristic
    coefficient. Scales are edge lengths (not ball radii). max_edge_length is
    inclusive; unresolved intervals at a truncated cutoff are censored.

    Returns a versioned Struct containing intervals and computation metadata.
    Full-range computation stores n*(n-1)/2 distances; a cutoff still visits all
    pairs but retains only threshold edges. Persistent homology may be expensive.

    method="sparse" requires finite epsilon > 0. Use max_filtration_value to cap
    the modified sparse filtration, rather than original edge lengths. Coverage
    and feature values describe that approximate filtration.

    Sparse metric_policy defaults to "unchecked" (no reported approximation
    bound). "assume" asserts the metric hypothesis; "check" verifies all binary64
    triangle inequalities without tolerance at O(n**3 * d) cost and may reject
    roundoff. For epsilon < 1, either permits the kernel's conditional
    ideal-arithmetic factor 1/(1-epsilon), not a floating-point error certificate.
    """
    if not coordinates:
        raise ValueError("rips requires at least one coordinate column")
    if method not in ("exact", "sparse"):
        raise ValueError("method must be 'exact' or 'sparse'")
    if method == "exact":
        if (
            epsilon is not None
            or metric_policy is not None
            or max_filtration_value is not None
        ):
            raise ValueError(
                "epsilon, metric_policy and max_filtration_value "
                "require method='sparse'"
            )
    else:
        if max_edge_length is not None:
            raise ValueError(
                "sparse Rips uses max_filtration_value, not max_edge_length"
            )
        if epsilon is None:
            raise ValueError("sparse Rips requires epsilon")
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if metric_policy is None:
            metric_policy = "unchecked"
        if metric_policy not in ("check", "assume", "unchecked"):
            raise ValueError("metric_policy must be 'check', 'assume' or 'unchecked'")
    if max_filtration_value is not None:
        max_filtration_value = float(max_filtration_value)
        if not math.isfinite(max_filtration_value) or max_filtration_value < 0:
            raise ValueError("max_filtration_value must be finite and nonnegative")
    if max_edge_length is not None:
        max_edge_length = float(max_edge_length)
        if not math.isfinite(max_edge_length) or max_edge_length < 0:
            raise ValueError("max_edge_length must be finite and nonnegative")
    return register_plugin_function(
        plugin_path=Path(_internal.__file__),
        function_name="rips",
        args=list(coordinates),
        kwargs={
            "max_dimension": _uint32(max_dimension, "max_dimension"),
            "max_edge_length": max_edge_length,
            "coefficient": _uint32(coefficient, "coefficient"),
            "method": method,
            "epsilon": epsilon,
            "metric_policy": metric_policy,
            "max_filtration_value": max_filtration_value,
        },
        is_elementwise=False,
        returns_scalar=True,
        input_wildcard_expansion=True,
    )


def betti_curve(
    diagram: str | pl.Expr,
    *,
    dimension: int,
    scales: Sequence[float],
) -> pl.Expr:
    """Return List(UInt64) live-class counts on a finite, increasing scale grid.

    Birth is inclusive and finite death exclusive. Querying an uncomputed
    dimension or a scale beyond censored coverage raises an error. A null diagram
    yields null. Stored diagrams are reused without computing homology again.
    """
    grid = [float(scale) for scale in scales]
    if any(not math.isfinite(scale) for scale in grid) or any(
        right <= left for left, right in zip(grid, grid[1:])
    ):
        raise ValueError("scales must be finite and strictly increasing")
    return register_plugin_function(
        plugin_path=Path(_internal.__file__),
        function_name="betti_curve",
        args=[diagram],
        kwargs={"dimension": _uint32(dimension, "dimension"), "scales": grid},
        is_elementwise=True,
    )


def lifetime_summary(diagram: str | pl.Expr, *, dimension: int) -> pl.Expr:
    """Return finite_count, total_persistence, max_persistence and entropy.

    Entropy uses natural logarithms. Essential and censored intervals are
    excluded and counted separately. When there are no finite intervals, total
    persistence is zero and maximum/entropy are null. Null diagrams yield null;
    uncomputed dimensions raise an error.
    """
    return register_plugin_function(
        plugin_path=Path(_internal.__file__),
        function_name="lifetime_summary",
        args=[diagram],
        kwargs={"dimension": _uint32(dimension, "dimension")},
        is_elementwise=True,
    )
