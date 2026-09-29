"""Expression shape and composition contracts, independent of algorithm details."""

from datetime import datetime, timedelta

import polars as pl
import polars.selectors as cs
import polars_tda as tda
import pytest
from polars.testing import assert_frame_equal


def square():
    return pl.DataFrame({"x": [0.0, 1.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0, 1.0]})


@pytest.mark.parametrize("method", ["exact", "approximate"])
@pytest.mark.parametrize(
    "selector", [pl.col(["x", "y"]), pl.col("^[xy]$"), cs.float(), pl.all()]
)
def test_selectors_form_one_multidimensional_cloud(method, selector):
    frame = square()
    expr = tda.features(selector, method=method).alias("topology")
    expected = frame.select(tda.features("x", "y", method=method).alias("topology"))
    query = frame.lazy().select(expr)
    assert query.collect_schema() == expected.schema
    assert_frame_equal(query.collect(), expected)
    assert_frame_equal(query.collect(engine="streaming"), expected)
    grouped = frame.with_columns(pl.lit("a").alias("sample"))
    # pl.all() includes the group key too, so use an explicit coordinate selector.
    assert_frame_equal(
        grouped.group_by("sample")
        .agg(tda.features(pl.col(["x", "y"]), method=method).alias("topology"))
        .drop("sample"),
        expected,
    )
    assert expected.item()["h1_finite_count"] == 1


@pytest.mark.parametrize("options", [{}, {"method": "sparse", "epsilon": 0.5}])
def test_advanced_coordinate_aggregation_obeys_same_expansion(options):
    frame = square()
    assert_frame_equal(
        frame.select(tda.rips(pl.col(["x", "y"]), **options)),
        frame.select(tda.rips("x", "y", **options)),
    )


def test_no_matching_coordinate_is_an_error():
    with pytest.raises(pl.exceptions.PolarsError, match="input|coordinate"):
        square().select(tda.features(pl.col("^missing.*$")))


@pytest.mark.parametrize("method", ["exact", "approximate"])
def test_over_broadcast_and_with_columns_global_aggregation(method):
    frame = pl.DataFrame({"sample": ["a", "b", "a", "b"], "x": [0.0, 0.0, 1.0, 2.0]})
    expr = tda.features("x", method=method).alias("topology")
    window = frame.with_columns(expr.over("sample"))
    grouped = frame.group_by("sample").agg(expr)
    expected = frame.join(grouped, on="sample", how="left", maintain_order="left")
    assert_frame_equal(window, expected)
    assert_frame_equal(
        frame.lazy().with_columns(expr.over("sample")).collect(), expected
    )
    global_value = frame.select(expr).item()
    assert (
        frame.with_columns(expr)["topology"].to_list() == [global_value] * frame.height
    )


@pytest.mark.parametrize("method", ["exact", "approximate"])
def test_native_time_windows_define_clouds(method):
    frame = pl.DataFrame(
        {
            "t": [datetime(2026, 1, 1) + timedelta(days=i) for i in range(4)],
            "x": [0.0, 1.0, 0.0, 2.0],
        }
    )
    expr = tda.features("x", method=method).alias("topology")
    for source in [frame, frame.lazy()]:
        dynamic = source.group_by_dynamic("t", every="2d").agg(expr)
        rolling = source.rolling("t", period="2d").agg(expr)
        if isinstance(dynamic, pl.LazyFrame):
            dynamic, rolling = dynamic.collect(), rolling.collect()
        assert dynamic.unnest("topology")["h0_total_persistence"].to_list() == [
            1.0,
            2.0,
        ]
        assert rolling.unnest("topology")["h0_total_persistence"].to_list() == [
            0.0,
            1.0,
            1.0,
            2.0,
        ]


def test_feature_selection_is_standard_struct_projection():
    frame = square().select(tda.features("x", "y").alias("topology"))
    result = frame.select(
        pl.col("topology").struct.field("h1_total_persistence").alias("loop_lifetime")
    )
    assert result.item() == frame.item()["h1_total_persistence"]
    assert result.schema == {"loop_lifetime": pl.Float64}


def test_unaligned_lengths_are_rejected_without_literal_broadcast():
    with pytest.raises(pl.exceptions.PolarsError, match="length"):
        square().select(tda.features("x", pl.lit(0.0)))
