# Releases and compatibility

## Current preview

[v0.1.0](https://github.com/Aequiludium/polars-tda/releases/tag/v0.1.0) is the first
GitHub source **pre-release**. The crate and Python distribution version are both
`0.1.0`; GitHub's pre-release flag communicates its preview status. The package
classifier remains Pre-Alpha, and no stable API guarantee is implied by the tag.
This is not a PyPI publication.

| Item | Current status |
| --- | --- |
| Distribution | GitHub source archives; build locally with maturin |
| Python | Package minimum 3.10; exercised locally on 3.10 and 3.13 |
| CI platform | Linux x86_64, Python 3.10 |
| Python Polars | Pinned to 1.44.2 |
| Rust | Pinned toolchain 1.95.0 |
| Native plugin | abi3-py310 build target; does not imply wheels exist for every platform |
| Windows/macOS | Not yet validated by this repository's CI |
| License | MIT, with the cocycle MIT notice retained |

See the [installation instructions](../README.md#installation),
[changelog](../CHANGELOG.md) and [CI runs](https://github.com/Aequiludium/polars-tda/actions/workflows/ci.yml).
The source preview uses the same pinned numerical implementation as the initial
CI-validated commit; display metadata and release documentation do not alter it.

## Development branch after v0.1.0

The unreleased development branch depends on the published `cocycle = "=0.1.1"`
crate from crates.io, with a checksum in `Cargo.lock`. The published polars-tda
`v0.1.0` tag and release notes still describe the original kernel Git revision
`d404282745a4238636ccb83d521eeac952ff7a30`. Kernel and plugin version numbers are
independent; adopting cocycle 0.1.1 does not publish polars-tda 0.1.1.

To evaluate this change, use the development checkout and the
[contributor build instructions](../CONTRIBUTING.md#development-setup). The tagged
installation above remains a reproducible historical source preview.

## Version and tag conventions

- Use annotated Git tags in the form `vMAJOR.MINOR.PATCH`, pointing to a reviewed
  commit. Keep the Cargo version, built Python metadata and release notes aligned.
- Mark preview GitHub releases as **Pre-release**; do not present them as stable
  releases or package-index publications.
- Record schema, dtype, null-rule and approximation-preset changes explicitly.
  An additive field can affect downstream tables and models too.
- Keep published version tags fixed. Corrections go into a new commit and, when
  needed, a new version rather than moving an existing release tag.
- A GitHub Release, a PyPI upload and a crates.io publication are separate actions.
  Current CI only builds and tests. It does not publish packages.

## Preparing a subsequent release

1. Review the final changes and update [CHANGELOG.md](../CHANGELOG.md).
2. Align package versions and lockfiles if the version changes. Build and install
   the actual release wheel, then run the [contribution checks](../CONTRIBUTING.md#validation).
3. Verify applicable CI results and documented platform support. Do not advertise
   untested platforms or attach a locally built wheel as a universal artifact.
4. Create and push an annotated version tag for the reviewed commit. Create the
   GitHub Release from that existing tag, with explicit compatibility and
   installation notes. Mark preview releases accordingly.
5. Verify the release's target commit, links, assets and status after publication.

Repository topics describe discoverability (Polars, TDA, Rust, Python); version
tags identify source snapshots. Neither is a performance or stability guarantee.
