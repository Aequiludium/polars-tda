"""Mathematical fixtures and real plugin integration, using the installed wheel."""

import io
import math

import polars as pl
import polars_tda as tda
import pytest
from polars.testing import assert_frame_equal


@pytest.fixture
def square():
    return pl.DataFrame({"x": [0.0, 1.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0, 1.0]})


def features(frame):
    return frame.with_columns(
        tda.betti_curve("diagram", dimension=1, scales=[0.0, 1.0, 2.0]).alias("h1"),
        tda.lifetime_summary("diagram", dimension=1).alias("lifetimes"),
    )


def test_square_pipeline_eager_lazy_and_streaming(square):
    eager = features(square.select(tda.rips("x", "y").alias("diagram")))
    lazy = features(square.lazy().select(tda.rips("x", "y").alias("diagram")))
    assert lazy.collect_schema() == eager.schema
    assert_frame_equal(lazy.collect(), eager)
    assert_frame_equal(lazy.collect(engine="streaming"), eager)
    row = eager.row(0, named=True)
    diagram = row["diagram"]
    assert diagram["complete"] and diagram["through"] is None
    assert diagram["backend_input"] == "distance_matrix"
    assert diagram["n_points"] == 4 and diagram["n_coordinates"] == 2
    h0 = [i for i in diagram["intervals"] if i["dimension"] == 0]
    h1 = [i for i in diagram["intervals"] if i["dimension"] == 1]
    assert len(h0) == 4 and len(h1) == 1
    assert sorted(i["death"] for i in h0 if i["end"] == "finite") == [1.0] * 3
    assert sum(i["end"] == "essential" for i in h0) == 1
    assert h1[0]["birth"] == 1.0
    assert h1[0]["death"] == pytest.approx(math.sqrt(2))
    assert row["h1"] == [0, 1, 0]
    assert row["lifetimes"] == {
        "finite_count": 1,
        "total_persistence": pytest.approx(math.sqrt(2) - 1),
        "max_persistence": pytest.approx(math.sqrt(2) - 1),
        "entropy": 0.0,
        "excluded_essential_count": 0,
        "excluded_censored_count": 0,
    }


def test_grouped_noncontiguous_clouds(square):
    # Interleave a square and a scaled square to catch cross-group mixing.
    frame = (
        pl.concat(
            [
                square.with_columns(pl.lit("a").alias("sample")),
                square.select(pl.all() * 2).with_columns(pl.lit("b").alias("sample")),
            ]
        )
        .with_row_index()
        .sort(pl.col("index") % 4)
    )
    query = features(
        frame.lazy().group_by("sample").agg(tda.rips("x", "y").alias("diagram"))
    )
    result = query.collect().sort("sample")
    assert result["h1"].to_list() == [[0, 1, 0], [0, 0, 1]]
    for sample in ("a", "b"):
        expected = features(
            frame.filter(pl.col("sample") == sample).select(
                tda.rips("x", "y").alias("diagram")
            )
        )
        assert_frame_equal(
            result.filter(pl.col("sample") == sample).drop("sample"), expected
        )


def test_chunk_boundaries_and_slices(square):
    frame = pl.DataFrame(
        [
            pl.concat([square["x"].head(1), square["x"].slice(1)], rechunk=False),
            pl.concat([square["y"].head(3), square["y"].tail(1)], rechunk=False),
        ]
    )
    assert frame["x"].n_chunks() == 2
    expr = tda.rips("x", "y")
    assert_frame_equal(frame.select(expr), square.select(expr))
    padded = pl.concat([square.head(1), square, square.tail(1)], rechunk=False)
    assert_frame_equal(padded.slice(1, 4).select(expr), square.select(expr))


def test_censoring_is_not_death_and_coverage_is_enforced(square):
    result = square.select(tda.rips("x", "y", max_edge_length=1).alias("diagram"))
    diagram = result.item()
    assert diagram["backend_input"] == "distance_callback"
    assert not diagram["complete"] and diagram["through"] == 1
    h1 = [i for i in diagram["intervals"] if i["dimension"] == 1]
    assert h1 == [{"dimension": 1, "birth": 1.0, "death": None, "end": "censored"}]
    assert result.select(
        tda.betti_curve("diagram", dimension=1, scales=[0, 1])
    ).item().to_list() == [0, 1]
    summary = result.select(tda.lifetime_summary("diagram", dimension=1)).item()
    assert summary["finite_count"] == 0 and summary["excluded_censored_count"] == 1
    assert summary["total_persistence"] == 0
    assert summary["max_persistence"] is None and summary["entropy"] is None
    with pytest.raises(pl.exceptions.PolarsError, match="coverage|through|outside"):
        result.select(tda.betti_curve("diagram", dimension=1, scales=[1.1]))


def test_cutoff_above_diameter_certifies_complete(square):
    result = square.select(tda.rips("x", "y", max_edge_length=2).alias("diagram"))
    assert result.item()["complete"]
    assert (
        result.item()["intervals"]
        == square.select(tda.rips("x", "y")).item()["intervals"]
    )
    assert result.select(
        tda.betti_curve("diagram", dimension=0, scales=[100])
    ).item().to_list() == [1]


@pytest.mark.parametrize("values, count", [([], 0), ([0.0], 1), ([0.0, 0.0], 1)])
def test_degenerate_clouds(values, count):
    result = pl.DataFrame({"x": values}, schema={"x": pl.Float64}).select(
        tda.rips("x").alias("diagram")
    )
    assert len(result.item()["intervals"]) == count
    assert (
        result.select(tda.betti_curve("diagram", dimension=1, scales=[]))
        .item()
        .to_list()
        == []
    )
    assert result.select(
        tda.betti_curve("diagram", dimension=0, scales=[0, 1])
    ).item().to_list() == [count, count]
    summary = result.select(tda.lifetime_summary("diagram", dimension=0)).item()
    assert summary["excluded_essential_count"] == count


def test_uncomputed_dimension_differs_from_computed_empty(square):
    result = square.select(tda.rips("x", "y", max_dimension=0).alias("diagram"))
    for expression in [
        tda.betti_curve("diagram", dimension=1, scales=[]),
        tda.lifetime_summary("diagram", dimension=1),
    ]:
        with pytest.raises(pl.exceptions.PolarsError, match="dimension"):
            result.select(expression)


def test_nulls_empty_batches_and_parquet_roundtrip(square):
    result = square.select(tda.rips("x", "y").alias("diagram"))
    result = pl.concat(
        [result, pl.DataFrame({"diagram": [None]}, schema=result.schema), result],
        rechunk=False,
    )
    expected = features(result)
    assert expected["h1"].to_list() == [[0, 1, 0], None, [0, 1, 0]]
    assert expected["lifetimes"].null_count() == 1
    assert features(result.head(0)).schema == expected.schema
    buffer = io.BytesIO()
    result.write_parquet(buffer)
    assert_frame_equal(features(pl.read_parquet(buffer)), expected)


def test_schema_version_and_interval_validation(square):
    result = square.select(tda.rips("x", "y").alias("diagram"))
    value = result.item()
    value["schema_version"] = 999
    malformed = pl.DataFrame({"diagram": [value]}, schema=result.schema)
    with pytest.raises(pl.exceptions.PolarsError, match="version"):
        features(malformed)
    value["schema_version"] = 2
    value["intervals"][0]["birth"] = 999.0
    malformed = pl.DataFrame({"diagram": [value]}, schema=result.schema)
    with pytest.raises(pl.exceptions.PolarsError, match="birth|interval"):
        features(malformed)


@pytest.mark.parametrize(
    "values, message",
    [
        ([None], "null"),
        ([float("nan")], "finite"),
        ([float("inf")], "finite"),
        ([-1e308, 1e308], "numerical"),
    ],
)
def test_invalid_coordinates(values, message):
    frame = pl.DataFrame({"x": values}, schema={"x": pl.Float64})
    with pytest.raises(pl.exceptions.PolarsError, match=message):
        frame.select(tda.rips("x"))


def test_casts_and_prime_field(square):
    frame = pl.DataFrame({"x": [0, 1]})
    with pytest.raises(pl.exceptions.PolarsError, match="Float64"):
        frame.select(tda.rips("x"))
    assert frame.select(tda.rips(pl.col("x").cast(pl.Float64))).item()["n_points"] == 2
    result = square.select(tda.rips("x", "y", coefficient=3))
    assert result.item()["coefficient"] == 3
    with pytest.raises(pl.exceptions.PolarsError, match="prime|characteristic"):
        square.select(tda.rips("x", "y", coefficient=4))


@pytest.mark.parametrize("grid", [[1, 1], [2, 1], [float("nan")], [float("inf")]])
def test_invalid_grid(grid):
    with pytest.raises(ValueError, match="scales"):
        tda.betti_curve("diagram", dimension=0, scales=grid)


def test_invalid_options():
    with pytest.raises(ValueError):
        tda.rips()
    with pytest.raises(TypeError):
        tda.rips("x", max_dimension=True)
    with pytest.raises(ValueError):
        tda.rips("x", max_dimension=-1)
    with pytest.raises(ValueError):
        tda.rips("x", max_edge_length=float("inf"))


def test_finite_lifetime_entropy_and_death_boundary():
    frame = pl.DataFrame({"x": [0.0, 1.0, 3.0]})
    diagram = frame.select(tda.rips("x", max_dimension=0).alias("diagram"))
    curve = diagram.select(tda.betti_curve("diagram", dimension=0, scales=[0, 1, 2]))
    assert curve.item().to_list() == [3, 2, 1]
    summary = diagram.select(tda.lifetime_summary("diagram", dimension=0)).item()
    assert summary["finite_count"] == 2
    assert summary["total_persistence"] == 3.0
    assert summary["max_persistence"] == 2.0
    assert summary["entropy"] == pytest.approx(
        -math.log(1 / 3) / 3 - 2 * math.log(2 / 3) / 3
    )
    assert summary["excluded_essential_count"] == 1


def test_higher_dimension_octahedron():
    points = pl.DataFrame(
        {
            "x": [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
        }
    )
    result = points.select(
        tda.rips("x", "y", "z", max_dimension=2, coefficient=3).alias("diagram")
    )
    h2 = [i for i in result.item()["intervals"] if i["dimension"] == 2]
    assert len(h2) == 1
    assert h2[0]["birth"] == pytest.approx(math.sqrt(2))
    assert h2[0]["death"] == 2.0
    assert result.select(
        tda.betti_curve("diagram", dimension=2, scales=[1, 1.5, 2])
    ).item().to_list() == [0, 1, 0]


@pytest.mark.parametrize("values", [[], [0.0], [0.0, 0.0], [0.0, 1.0]])
def test_zero_cutoff_degenerate_clouds(values):
    result = pl.DataFrame({"x": values}, schema={"x": pl.Float64}).select(
        tda.rips("x", max_edge_length=0).alias("diagram")
    )
    assert result.select(
        tda.betti_curve("diagram", dimension=0, scales=[0])
    ).item().to_list() == [len(set(values))]


def test_composed_expressions(square):
    result = square.select(
        tda.betti_curve(tda.rips("x", "y"), dimension=1, scales=[0, 1, 2])
    )
    assert result.item().to_list() == [0, 1, 0]
