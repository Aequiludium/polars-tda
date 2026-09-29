"""Sparse construction semantics and shared diagram/descriptor integration."""

import io
import math

import polars as pl
import polars_tda as tda
import pytest
from polars.testing import assert_frame_equal


def square():
    return pl.DataFrame({"x": [0.0, 1.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0, 1.0]})


def summarize(frame):
    return frame.with_columns(
        tda.betti_curve("diagram", dimension=1, scales=[0, 1, 2]).alias("h1"),
        tda.lifetime_summary("diagram", dimension=1).alias("summary"),
    )


def test_square_sparse_pipeline():
    expr = tda.rips("x", "y", method="sparse", epsilon=0.5).alias("diagram")
    eager = summarize(square().select(expr))
    query = summarize(square().lazy().select(expr))
    assert query.collect_schema() == eager.schema
    assert_frame_equal(query.collect(), eager)
    assert_frame_equal(query.collect(engine="streaming"), eager)
    diagram = eager["diagram"][0]
    assert diagram["method"] == "sparse" and diagram["backend_input"] == "points"
    assert diagram["complete"] and diagram["through"] is None
    assert eager["h1"].to_list() == [[0, 1, 0]]
    assert eager["summary"][0]["total_persistence"] == pytest.approx(math.sqrt(2) - 1)
    assert diagram["approximation"]["metric_validation"] == "unchecked"
    assert diagram["approximation"]["bound_factor"] is None
    assert diagram["approximation"]["bound_target"] is None


@pytest.mark.parametrize(
    "policy, evidence", [("check", "checked"), ("assume", "assumed")]
)
def test_conditional_bound_and_duplicate_vertex_provenance(policy, evidence):
    frame = pl.DataFrame({"x": [0.0, 0.0, 2.0, 10.0, 5.0]})
    result = frame.select(
        tda.rips("x", method="sparse", epsilon=0.5, metric_policy=policy)
    ).item()
    assert result["n_points"] == 5
    a = result["approximation"]
    # Matches the deterministic kernel fixture, including original vertex IDs.
    assert a["permutation"] == [0, 3, 4, 2, 1]
    assert a["insertion_radii"] == [None, 10.0, 5.0, 2.0, 0.0]
    assert a["retained_vertices"] == [0, 2, 3, 4]
    assert a["covering_radius"] == 0
    assert a["metric_validation"] == evidence
    assert a["bound_factor"] == 2 and a["bound_target"] == "original_input"


@pytest.mark.parametrize("epsilon", [1.0, 1.5, 2.0])
def test_large_epsilon_has_no_bound(epsilon):
    result = (
        square()
        .select(
            tda.rips("x", "y", method="sparse", epsilon=epsilon, metric_policy="check")
        )
        .item()
    )
    assert result["approximation"]["bound_factor"] is None
    assert result["approximation"]["bound_target"] is None


def test_modified_filtration_is_not_original_edge_length():
    # With epsilon=2, the edge from point 3 to point 1 is delayed from 2 to 3.
    frame = pl.DataFrame({"x": [0.0, 1.0, 3.0]})
    sparse = frame.select(tda.rips("x", method="sparse", epsilon=2).alias("diagram"))
    deaths = [i["death"] for i in sparse.item()["intervals"] if i["end"] == "finite"]
    assert deaths == [1.0, 3.0]
    truncated = frame.select(
        tda.rips("x", method="sparse", epsilon=2, max_filtration_value=2).alias(
            "diagram"
        )
    )
    assert truncated.item()["max_edge_length"] is None
    assert truncated.item()["max_filtration_value"] == 2
    assert not truncated.item()["complete"] and truncated.item()["through"] == 2
    assert truncated.select(
        tda.betti_curve("diagram", dimension=0, scales=[2])
    ).item().to_list() == [2]
    summary = truncated.select(tda.lifetime_summary("diagram", dimension=0)).item()
    assert summary["excluded_censored_count"] == 2
    with pytest.raises(pl.exceptions.PolarsError, match="coverage|through|outside"):
        truncated.select(tda.betti_curve("diagram", dimension=0, scales=[3]))


def test_packing_chunks_offsets_and_group_local_vertex_ids():
    cloud = square()
    expr = tda.rips("x", "y", method="sparse", epsilon=0.5).alias("diagram")
    padded = pl.concat([cloud.head(1), cloud, cloud.tail(1)], rechunk=False).slice(1, 4)
    assert_frame_equal(padded.select(expr), cloud.select(expr))
    uneven = pl.DataFrame(
        [
            pl.concat([cloud["x"].head(1), cloud["x"].slice(1)], rechunk=False),
            pl.concat([cloud["y"].head(3), cloud["y"].tail(1)], rechunk=False),
        ]
    )
    assert_frame_equal(uneven.select(expr), cloud.select(expr))
    groups = (
        pl.concat(
            [
                cloud.with_columns(pl.lit("a").alias("sample")),
                cloud.select(pl.all() * 2).with_columns(pl.lit("b").alias("sample")),
            ]
        )
        .with_row_index()
        .sort(pl.col("index") % 4)
    )
    query = summarize(groups.lazy().group_by("sample").agg(expr))
    result = query.collect().sort("sample")
    assert_frame_equal(query.collect(engine="streaming").sort("sample"), result)
    for sample in ["a", "b"]:
        expected = summarize(groups.filter(pl.col("sample") == sample).select(expr))
        assert_frame_equal(
            result.filter(pl.col("sample") == sample).drop("sample"), expected
        )


def test_single_coordinate_borrow_and_multichunk_pack_agree():
    values = pl.Series("x", [99.0, 0.0, 1.0, 3.0, 99.0]).slice(1, 3)
    chunks = pl.concat([values.head(1), values.slice(1)], rechunk=False)
    expr = tda.rips("x", method="sparse", epsilon=0.5)
    assert_frame_equal(values.to_frame().select(expr), chunks.to_frame().select(expr))


@pytest.mark.parametrize("values, expected", [([], 0), ([0.0], 1), ([0.0, 0.0], 1)])
def test_empty_singleton_duplicates(values, expected):
    result = pl.DataFrame({"x": values}, schema={"x": pl.Float64}).select(
        tda.rips("x", method="sparse", epsilon=0.5).alias("diagram")
    )
    assert len(result.item()["approximation"]["retained_vertices"]) == expected
    assert result.select(
        tda.betti_curve("diagram", dimension=0, scales=[0, 1])
    ).item().to_list() == [expected, expected]


def test_explicit_metric_check_can_reject_rounded_euclidean_distances():
    # Stored 0.3 + stored 0.7 is strictly below stored 1.0 in exact arithmetic.
    frame = pl.DataFrame({"x": [0.0, 0.3, 1.0]})
    with pytest.raises(pl.exceptions.PolarsError, match="metric|triangle"):
        frame.select(tda.rips("x", method="sparse", epsilon=0.5, metric_policy="check"))
    result = frame.select(tda.rips("x", method="sparse", epsilon=0.5)).item()
    assert result["approximation"]["bound_factor"] is None


def test_mixed_methods_nulls_and_parquet():
    exact = square().select(tda.rips("x", "y").alias("diagram"))
    sparse = square().select(
        tda.rips("x", "y", method="sparse", epsilon=0.5).alias("diagram")
    )
    assert exact.schema == sparse.schema
    assert exact.item()["approximation"] is None
    frame = pl.concat(
        [exact, sparse, pl.DataFrame({"diagram": [None]}, schema=exact.schema)],
        rechunk=False,
    )
    expected = summarize(frame)
    assert expected["summary"].null_count() == 1
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    assert_frame_equal(summarize(pl.read_parquet(buffer)), expected)
    assert summarize(frame.head(0)).schema == expected.schema


def test_legacy_diagram_remains_readable():
    frame = square().select(tda.rips("x", "y").alias("diagram"))
    legacy_fields = [
        f
        for f in frame.schema["diagram"].fields
        if f.name not in ("approximation", "max_filtration_value")
    ]
    value = frame.item()
    del value["approximation"], value["max_filtration_value"]
    value["schema_version"] = 1
    legacy = pl.DataFrame(
        {"diagram": [value]}, schema={"diagram": pl.Struct(legacy_fields)}
    )
    assert_frame_equal(
        summarize(frame).drop("diagram"), summarize(legacy).drop("diagram")
    )
    buffer = io.BytesIO()
    legacy.write_parquet(buffer)
    assert_frame_equal(summarize(pl.read_parquet(buffer)), summarize(legacy))
    value["schema_version"] = 2
    mismatched = pl.DataFrame({"diagram": [value]}, schema=legacy.schema)
    with pytest.raises(pl.exceptions.PolarsError, match="version"):
        summarize(mismatched)


@pytest.mark.parametrize(
    "options",
    [
        {"method": "unknown"},
        {"epsilon": 0.5},
        {"metric_policy": "check"},
        {"max_filtration_value": 1},
        {"method": "sparse"},
        {"method": "sparse", "epsilon": 0},
        {"method": "sparse", "epsilon": -1},
        {"method": "sparse", "epsilon": float("nan")},
        {"method": "sparse", "epsilon": float("inf")},
        {"method": "sparse", "epsilon": 0.5, "metric_policy": "unknown"},
        {"method": "sparse", "epsilon": 0.5, "max_edge_length": 1},
        {"method": "sparse", "epsilon": 0.5, "max_filtration_value": -1},
        {"method": "sparse", "epsilon": 0.5, "max_filtration_value": float("inf")},
    ],
)
def test_invalid_options_fail_before_execution(options):
    with pytest.raises(ValueError):
        tda.rips("x", **options)


def test_underflowing_epsilon_propagates_kernel_error():
    with pytest.raises(pl.exceptions.PolarsError, match="underflow"):
        square().select(tda.rips("x", "y", method="sparse", epsilon=math.ulp(0.0)))
