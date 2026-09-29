"""Analyst-facing point-cloud aggregation with one computational choice."""

import math
import random

import polars as pl
import polars_tda as tda
import pytest
from polars.testing import assert_frame_equal


def square():
    return pl.DataFrame({"x": [0.0, 1.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0, 1.0]})


@pytest.mark.parametrize("method", ["exact", "approximate"])
def test_square_feature_values_schema_and_execution_modes(method):
    expr = tda.features("x", "y", method=method).alias("topology")
    expected = {
        "h0_finite_count": 3,
        "h0_total_persistence": 3.0,
        "h0_max_persistence": 1.0,
        "h0_entropy": pytest.approx(math.log(3)),
        "h0_essential_count": 1,
        "h1_finite_count": 1,
        "h1_total_persistence": pytest.approx(math.sqrt(2) - 1),
        "h1_max_persistence": pytest.approx(math.sqrt(2) - 1),
        "h1_entropy": 0.0,
        "h1_essential_count": 0,
    }
    eager = square().select(expr).unnest("topology")
    assert eager.row(0, named=True) == expected
    assert eager.width == 10
    for name, dtype in eager.schema.items():
        assert dtype == (pl.UInt64 if name.endswith("count") else pl.Float64)
    query = square().lazy().select(expr).unnest("topology")
    assert query.collect_schema() == eager.schema
    assert_frame_equal(query.collect(), eager)
    assert_frame_equal(query.collect(engine="streaming"), eager)


@pytest.mark.parametrize("method", ["exact", "approximate"])
def test_grouped_features_and_coordinate_scale(method):
    cloud = square()
    frame = (
        pl.concat(
            [
                cloud.with_columns(pl.lit("a").alias("sample")),
                cloud.select(pl.all() * 2).with_columns(pl.lit("b").alias("sample")),
            ]
        )
        .with_row_index()
        .sort(pl.col("index") % 4)
    )
    expr = tda.features("x", "y", method=method).alias("topology")
    query = frame.lazy().group_by("sample").agg(expr).unnest("topology")
    result = query.collect().sort("sample")
    assert_frame_equal(query.collect(engine="streaming").sort("sample"), result)
    for sample in ["a", "b"]:
        expected = (
            frame.filter(pl.col("sample") == sample).select(expr).unnest("topology")
        )
        assert_frame_equal(
            result.filter(pl.col("sample") == sample).drop("sample"), expected
        )
    assert result["h0_total_persistence"].to_list() == [3.0, 6.0]
    assert result["h1_total_persistence"][1] == pytest.approx(
        2 * result["h1_total_persistence"][0]
    )
    assert result["h0_entropy"][0] == pytest.approx(result["h0_entropy"][1])
    assert (
        frame.head(0).group_by("sample").agg(expr).unnest("topology").schema
        == query.collect_schema()
    )


@pytest.mark.parametrize("method", ["exact", "approximate"])
@pytest.mark.parametrize("values, essential", [([], 0), ([0.0], 1), ([0.0, 0.0], 1)])
def test_empty_and_degenerate_clouds(method, values, essential):
    result = (
        pl.DataFrame({"x": values}, schema={"x": pl.Float64})
        .select(tda.features("x", method=method))
        .item()
    )
    assert result["h0_essential_count"] == essential
    for dimension in [0, 1]:
        assert result[f"h{dimension}_finite_count"] == 0
        assert result[f"h{dimension}_total_persistence"] == 0
        assert result[f"h{dimension}_max_persistence"] is None
        assert result[f"h{dimension}_entropy"] is None


@pytest.mark.parametrize("method", ["exact", "approximate"])
def test_chunked_sliced_coordinates(method):
    cloud = square()
    expr = tda.features("x", "y", method=method)
    uneven = pl.DataFrame(
        [
            pl.concat([cloud["x"].head(1), cloud["x"].slice(1)], rechunk=False),
            pl.concat([cloud["y"].head(3), cloud["y"].tail(1)], rechunk=False),
        ]
    )
    assert_frame_equal(uneven.select(expr), cloud.select(expr))
    padded = pl.concat([cloud.head(1), cloud, cloud.tail(1)], rechunk=False)
    assert_frame_equal(padded.slice(1, 4).select(expr), cloud.select(expr))


def test_presets_agree_with_advanced_pipeline_and_approximation_is_real():
    rng = random.Random(42)
    cloud = pl.DataFrame(
        {f"x{i}": [rng.random() for _ in range(256)] for i in range(3)}
    )
    presets = []
    for method in ["exact", "approximate"]:
        result = cloud.select(tda.features(*cloud.columns, method=method)).item()
        presets.append(result)
        options = (
            {}
            if method == "exact"
            else {"method": "sparse", "epsilon": 0.5, "metric_policy": "unchecked"}
        )
        diagrams = cloud.select(tda.rips(*cloud.columns, **options).alias("diagram"))
        for dimension in [0, 1]:
            summary = diagrams.select(
                tda.lifetime_summary("diagram", dimension=dimension)
            ).item()
            for metric in [
                "finite_count",
                "total_persistence",
                "max_persistence",
                "entropy",
            ]:
                assert result[f"h{dimension}_{metric}"] == summary[metric]
            assert (
                result[f"h{dimension}_essential_count"]
                == summary["excluded_essential_count"]
            )
            assert summary["excluded_censored_count"] == 0
    assert presets[0] != presets[1]
    assert_frame_equal(
        cloud.select(tda.features(*cloud.columns)),
        cloud.select(tda.features(*cloud.columns, method="exact")),
    )


@pytest.mark.parametrize("method", ["exact", "approximate"])
@pytest.mark.parametrize(
    "value, message", [(None, "null"), (math.nan, "finite"), (math.inf, "finite")]
)
def test_invalid_coordinates(method, value, message):
    frame = pl.DataFrame({"x": [value]}, schema={"x": pl.Float64})
    with pytest.raises(pl.exceptions.PolarsError, match=message):
        frame.select(tda.features("x", method=method))


def test_explicit_cast_and_simple_options():
    frame = pl.DataFrame({"x": [0, 1]})
    with pytest.raises(pl.exceptions.PolarsError, match="Float64"):
        frame.select(tda.features("x"))
    assert (
        frame.select(tda.features(pl.col("x").cast(pl.Float64))).item()[
            "h0_total_persistence"
        ]
        == 1.0
    )
    with pytest.raises(ValueError, match="coordinate"):
        tda.features()
    with pytest.raises(ValueError, match="method"):
        tda.features("x", method="sparse")
    with pytest.raises(TypeError, match="epsilon"):
        tda.features("x", epsilon=0.5)
