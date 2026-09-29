"""Installed package metadata."""

from importlib.metadata import version

import polars_tda


def test_native_module_matches_distribution_version():
    assert polars_tda.__version__ == version("polars-tda")
