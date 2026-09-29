//! Native Polars expressions backed by the cocycle kernel.

mod coordinates;
mod diagram;
mod expressions;

use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

#[global_allocator]
static ALLOCATOR: PolarsAllocator = PolarsAllocator::new();

#[pymodule]
fn _internal(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))
}
