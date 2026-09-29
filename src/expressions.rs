//! Expression terminals: coordinate aggregation and reusable diagram descriptors.
use cocycle::algebra::PrimeField;
use cocycle::descriptors;
use cocycle::diagram::{Coverage, PersistenceResult};
use cocycle::filtration::{ApproximateRipsBuilder, RipsBuilder};
use cocycle::geometry::{DissimilarityMatrixView, MatrixLayout, MetricPolicy, PointCloudView};
use cocycle::persistence::PersistenceExt;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::coordinates::Coordinates;
use crate::diagram::{self, Reader, kernel_error};

#[derive(Deserialize)]
struct RipsOptions {
    max_dimension: u32,
    max_edge_length: Option<f64>,
    coefficient: u32,
    method: Method,
    epsilon: Option<f64>,
    metric_policy: Option<Policy>,
    max_filtration_value: Option<f64>,
}

#[derive(Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Method {
    Exact,
    Sparse,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum Policy {
    Check,
    Assume,
    Unchecked,
}

impl RipsOptions {
    fn validate(&self) -> PolarsResult<()> {
        match self.method {
            Method::Exact => {
                polars_ensure!(self.epsilon.is_none() && self.metric_policy.is_none() && self.max_filtration_value.is_none(), InvalidOperation: "sparse options require method='sparse'")
            }
            Method::Sparse => {
                polars_ensure!(self.max_edge_length.is_none(), InvalidOperation: "sparse Rips uses max_filtration_value, not max_edge_length");
                polars_ensure!(self.epsilon.is_some_and(|e| e.is_finite() && e > 0.), InvalidOperation: "sparse Rips requires finite positive epsilon");
                polars_ensure!(self.metric_policy.is_some(), InvalidOperation: "sparse Rips requires metric_policy");
                polars_ensure!(self.max_filtration_value.is_none_or(|v| v.is_finite() && v >= 0.), InvalidOperation: "max_filtration_value must be finite and nonnegative");
            }
        }
        Ok(())
    }
}

fn coordinate_field_name(inputs: &[Field]) -> PolarsResult<PlSmallStr> {
    polars_ensure!(!inputs.is_empty(), InvalidOperation: "analysis requires coordinates");
    for input in inputs {
        polars_ensure!(input.dtype() == &DataType::Float64, InvalidOperation: "coordinates must be Float64; cast explicitly before analysis");
    }
    Ok(inputs[0].name().clone())
}

fn compute_rips(
    inputs: &[Series],
    kwargs: &RipsOptions,
) -> PolarsResult<(PersistenceResult, &'static str)> {
    kwargs.validate()?;
    let coordinates = Coordinates::new(inputs)?;
    let field = PrimeField::new(kwargs.coefficient).map_err(kernel_error)?;
    let dimension = kwargs.max_dimension as usize;
    let (result, backend_input) = if kwargs.method == Method::Sparse {
        let values = coordinates.row_major()?;
        let points =
            PointCloudView::new(&values, coordinates.len, inputs.len()).map_err(kernel_error)?;
        let policy = match kwargs.metric_policy.expect("validated policy") {
            Policy::Check => MetricPolicy::Check,
            Policy::Assume => MetricPolicy::Assume,
            Policy::Unchecked => MetricPolicy::Unchecked,
        };
        let mut builder = ApproximateRipsBuilder::from_points(
            points,
            kwargs.epsilon.expect("validated epsilon"),
            policy,
        );
        if let Some(cap) = kwargs.max_filtration_value {
            builder = builder.max_filtration_value(cap);
        }
        (
            builder
                .persistence()
                .max_homology_dimension(dimension)
                .field(field)
                .compute()
                .map_err(kernel_error)?,
            "points",
        )
    } else if let Some(cap) = kwargs.max_edge_length {
        polars_ensure!(cap.is_finite() && cap >= 0., InvalidOperation: "max_edge_length must be finite and nonnegative");
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(coordinates.len)
            .map_err(|_| polars_err!(ComputeError: "cannot allocate point indices"))?;
        indices.extend(0..coordinates.len);
        let filtration =
            RipsBuilder::from_distance_fn(&indices, |&a, &b| coordinates.distance(a, b))
                .max_edge_length(cap)
                .prepare()
                .map_err(kernel_error)?;
        (
            filtration
                .persistence()
                .max_homology_dimension(dimension)
                .field(field)
                .compute()
                .map_err(kernel_error)?,
            "distance_callback",
        )
    } else {
        let n = coordinates.len;
        // Divide before multiplying so a representable pair count is accepted.
        let (a, b) = if n.is_multiple_of(2) {
            (n / 2, n.saturating_sub(1))
        } else {
            (n, n / 2)
        };
        let pairs = a
            .checked_mul(b)
            .ok_or_else(|| polars_err!(ComputeError: "pair count overflow"))?;
        let mut distances = Vec::new();
        distances
            .try_reserve_exact(pairs)
            .map_err(|_| polars_err!(ComputeError: "cannot allocate condensed distances"))?;
        for i in 0..n {
            for j in 0..i {
                distances.push(coordinates.distance(i, j).map_err(kernel_error)?);
            }
        }
        let matrix = DissimilarityMatrixView::new(&distances, n, MatrixLayout::LowerTriangle)
            .map_err(kernel_error)?;
        (
            RipsBuilder::from_distance_matrix(matrix)
                .persistence()
                .max_homology_dimension(dimension)
                .field(field)
                .compute()
                .map_err(kernel_error)?,
            "distance_matrix",
        )
    };
    Ok((result, backend_input))
}

fn diagram_field(inputs: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(coordinate_field_name(inputs)?, diagram::dtype()))
}

#[polars_expr(output_type_func=diagram_field)]
fn rips(inputs: &[Series], kwargs: RipsOptions) -> PolarsResult<Series> {
    let (result, backend_input) = compute_rips(inputs, &kwargs)?;
    let n_coordinates = u32::try_from(inputs.len())
        .map_err(|_| polars_err!(ComputeError: "coordinate count exceeds UInt32"))?;
    diagram::encode(
        inputs[0].name().clone(),
        &result,
        inputs[0].len(),
        n_coordinates,
        kwargs.max_dimension,
        kwargs.max_edge_length,
        backend_input,
    )
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum FeatureMethod {
    Exact,
    Approximate,
}

#[derive(Deserialize)]
struct FeatureOptions {
    method: FeatureMethod,
}

fn features_field(inputs: &[Field]) -> PolarsResult<Field> {
    let name = coordinate_field_name(inputs)?;
    let mut fields = Vec::with_capacity(10);
    for dimension in 0..=1 {
        for (metric, dtype) in [
            ("finite_count", DataType::UInt64),
            ("total_persistence", DataType::Float64),
            ("max_persistence", DataType::Float64),
            ("entropy", DataType::Float64),
            ("essential_count", DataType::UInt64),
        ] {
            fields.push(Field::new(format!("h{dimension}_{metric}").into(), dtype));
        }
    }
    Ok(Field::new(name, DataType::Struct(fields)))
}

/// Analyst preset: one computation, then both dimensions directly in Rust.
/// No intermediate Arrow diagram serialization or reconstruction is needed.
#[polars_expr(output_type_func=features_field)]
fn features(inputs: &[Series], kwargs: FeatureOptions) -> PolarsResult<Series> {
    let approximate = matches!(kwargs.method, FeatureMethod::Approximate);
    let options = RipsOptions {
        max_dimension: 1,
        coefficient: 2,
        method: if approximate {
            Method::Sparse
        } else {
            Method::Exact
        },
        epsilon: approximate.then_some(0.5),
        metric_policy: approximate.then_some(Policy::Unchecked),
        max_edge_length: None,
        max_filtration_value: None,
    };
    let (result, _) = compute_rips(inputs, &options)?;
    polars_ensure!(matches!(result.diagram().coverage(), Coverage::Complete), ComputeError: "features requires complete filtration coverage");
    let mut fields = Vec::with_capacity(10);
    for dimension in 0..=1 {
        let summary = descriptors::finite_lifetime_summary(result.diagram(), dimension)
            .map_err(kernel_error)?;
        fields.extend([
            Series::new(
                format!("h{dimension}_finite_count").into(),
                [summary.finite_count() as u64],
            ),
            Series::new(
                format!("h{dimension}_total_persistence").into(),
                [summary.total_persistence()],
            ),
            Series::new(
                format!("h{dimension}_max_persistence").into(),
                [summary.max_persistence()],
            ),
            Series::new(format!("h{dimension}_entropy").into(), [summary.entropy()]),
            Series::new(
                format!("h{dimension}_essential_count").into(),
                [summary.excluded_essential_count() as u64],
            ),
        ]);
    }
    Ok(StructChunked::from_series(inputs[0].name().clone(), 1, fields.iter())?.into_series())
}

#[derive(Deserialize)]
struct CurveOptions {
    dimension: u32,
    scales: Vec<f64>,
}
#[derive(Deserialize)]
struct SummaryOptions {
    dimension: u32,
}

fn validate_diagram_field(inputs: &[Field]) -> PolarsResult<PlSmallStr> {
    polars_ensure!(inputs.len() == 1 && diagram::is_supported_dtype(inputs[0].dtype()), InvalidOperation: "expected one polars_tda.rips diagram Struct");
    Ok(inputs[0].name().clone())
}

fn curve_field(inputs: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        validate_diagram_field(inputs)?,
        DataType::List(Box::new(DataType::UInt64)),
    ))
}

#[polars_expr(output_type_func=curve_field)]
fn betti_curve(inputs: &[Series], kwargs: CurveOptions) -> PolarsResult<Series> {
    let input = &inputs[0];
    let reader = Reader::new(input)?;
    let mut output = ListPrimitiveChunkedBuilder::<UInt64Type>::new(
        input.name().clone(),
        input.len(),
        0,
        DataType::UInt64,
    );
    for row in 0..input.len() {
        if let Some(diagram) = reader.get(row)? {
            let curve =
                descriptors::betti_curve(&diagram, kwargs.dimension as usize, &kwargs.scales)
                    .map_err(kernel_error)?;
            output.append_values_iter(curve.into_iter().map(|count| count as u64));
        } else {
            output.append_null();
        }
    }
    Ok(output.finish().into_series())
}

fn summary_field(inputs: &[Field]) -> PolarsResult<Field> {
    use DataType::*;
    let fields = [
        ("finite_count", UInt64),
        ("total_persistence", Float64),
        ("max_persistence", Float64),
        ("entropy", Float64),
        ("excluded_essential_count", UInt64),
        ("excluded_censored_count", UInt64),
    ]
    .into_iter()
    .map(|(name, dtype)| Field::new(name.into(), dtype))
    .collect();
    Ok(Field::new(validate_diagram_field(inputs)?, Struct(fields)))
}

#[polars_expr(output_type_func=summary_field)]
fn lifetime_summary(inputs: &[Series], kwargs: SummaryOptions) -> PolarsResult<Series> {
    let input = &inputs[0];
    let reader = Reader::new(input)?;
    let mut counts = Vec::with_capacity(input.len());
    let mut totals = Vec::with_capacity(input.len());
    let mut maxima = Vec::with_capacity(input.len());
    let mut entropies = Vec::with_capacity(input.len());
    let mut essentials = Vec::with_capacity(input.len());
    let mut censored = Vec::with_capacity(input.len());
    for row in 0..input.len() {
        let summary = reader
            .get(row)?
            .map(|diagram| {
                descriptors::finite_lifetime_summary(&diagram, kwargs.dimension as usize)
            })
            .transpose()
            .map_err(kernel_error)?;
        counts.push(summary.as_ref().map(|s| s.finite_count() as u64));
        totals.push(summary.as_ref().map(|s| s.total_persistence()));
        maxima.push(summary.as_ref().and_then(|s| s.max_persistence()));
        entropies.push(summary.as_ref().and_then(|s| s.entropy()));
        essentials.push(
            summary
                .as_ref()
                .map(|s| s.excluded_essential_count() as u64),
        );
        censored.push(summary.as_ref().map(|s| s.excluded_censored_count() as u64));
    }
    let fields = [
        Series::new("finite_count".into(), counts),
        Series::new("total_persistence".into(), totals),
        Series::new("max_persistence".into(), maxima),
        Series::new("entropy".into(), entropies),
        Series::new("excluded_essential_count".into(), essentials),
        Series::new("excluded_censored_count".into(), censored),
    ];
    let mut result = StructChunked::from_series(input.name().clone(), input.len(), fields.iter())?;
    result.zip_outer_validity(input.struct_()?);
    Ok(result.into_series())
}
