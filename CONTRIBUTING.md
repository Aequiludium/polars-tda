# Contributing to polars-tda

Use [GitHub issues](https://github.com/Aequiludium/polars-tda/issues) for bugs,
questions and proposed workflows. For changes to the public API, result schema or
approximation preset, describe the use case and compatibility impact before
investing in a large implementation. Small fixes can go straight to a pull request.

## Development setup

Install uv, Python 3.10 or later, and Rust through rustup. The repository's
`rust-toolchain.toml` selects Rust 1.95.0 and the required components. From a clone:

```sh
uv sync --locked --no-install-project
uv run --no-sync maturin develop --release --locked
uv run --no-sync pytest
```

A native build is required before importing the package. The development commands
use the local uv environment. Keep the pinned Polars and plugin ABI versions
unless the change explicitly updates and validates compatibility.

## Style and project boundaries

Use English for source comments and documentation. Format Python with Ruff and
Rust with rustfmt. Follow the existing small module layout and add dependencies
only when the operation needs them.

- Python constructs Polars expressions and validates options.
- The native adapter handles inputs, output schemas and kernel calls.
- Mathematical algorithms belong to cocycle. Adding a kernel algorithm does not
  require simultaneously adding a Polars API.
- Preserve the [expression contract](docs/expression-contract.md). State changes
  to dtype, shape, null behavior, field names or approximation settings explicitly.
- Describe allocations and repeated work accurately; native execution does not
  imply a zero-copy query or a speedup.

See the [design rationale](docs/design-rationale.md) and
[scope and proposed work](docs/api-discussion.md) before adding a new public surface.

## Validation

Run the checks relevant to the change. For code changes, the CI checks are:

```sh
cargo fmt --all -- --check
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
cargo clippy --locked --lib --example profile_sparse -- -D warnings
uv run --no-sync maturin build --release --locked --out dist
uv pip install --no-deps --reinstall dist/polars_tda-*.whl
uv run --no-sync pytest
```

Install the wheel produced by the current build; if `dist` contains multiple
versions, choose the specific new wheel instead of the wildcard. Tests must run
against the current native library, not an older installed build.

For behavior changes, test mathematical results and relevant expression contexts,
including grouped/lazy operation, empty inputs and invalid inputs where applicable.
For documentation changes, run changed examples and check their links and outputs.
Record performance measurements with input sizes, versions and build mode; follow
the [benchmark notes](docs/benchmark-notes.md) when making speed or memory claims.

## Pull requests

Create a topic branch and open a pull request against `main`. Give it a concise
title describing the resulting change. Explain the problem, the final behavior
and validation performed; link related issues when relevant. Use the supplied
template as a guide and remove sections that do not apply.

Keep generated wheels, native libraries, virtual environments and benchmark data
out of commits. Commit lockfile changes alongside intentional dependency changes.
CI builds and tests but does not publish packages. Releases are separate maintainer
work; a merged PR does not imply a PyPI release. See the
[release conventions](docs/releases.md) and record user-visible changes in
[CHANGELOG.md](CHANGELOG.md).

Be respectful, focus review on the work, and make reports reproducible without
including credentials or private datasets. Contributions are made under the
project's [MIT license](LICENSE); retain applicable third-party notices.
