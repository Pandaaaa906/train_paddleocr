"""Shared CLI options and validation callbacks for prepare_traindata generators.

Each option is exposed as a callable that accepts a *default* argument and
returns a ``click.option`` decorator.  Generators compose the options they need
with their own defaults rather than relying on a single monolithic decorator.

All callbacks follow Click's ``(ctx, param, value) -> value`` signature.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import click

# ---------------------------------------------------------------------------
# Validation callbacks
# ---------------------------------------------------------------------------


def validate_split(ctx: click.Context, param: click.Parameter, value: float) -> float:
    """Reject split values outside the inclusive range [0.0, 1.0]."""
    if not 0.0 <= value <= 1.0:
        raise click.BadParameter(
            f"{param.name} must be between 0.0 and 1.0, got {value}"
        )
    return value


def validate_positive_int(
    ctx: click.Context, param: click.Parameter, value: int
) -> int:
    """Reject integers <= 0."""
    if value <= 0:
        raise click.BadParameter(f"{param.name} must be > 0, got {value}")
    return value


def validate_non_negative_int(
    ctx: click.Context, param: click.Parameter, value: int
) -> int:
    """Reject integers < 0."""
    if value < 0:
        raise click.BadParameter(f"{param.name} must be >= 0, got {value}")
    return value


def validate_min_max(
    ctx: click.Context,
    param: click.Parameter,
    value: tuple[Any, Any],
) -> tuple[Any, Any]:
    """Ensure the first element of a tuple option is <= the second element."""
    if value[0] > value[1]:
        raise click.BadParameter(
            f"{param.name} min must be <= max, got ({value[0]}, {value[1]})"
        )
    return value


# ---------------------------------------------------------------------------
# Shared option callables
# ---------------------------------------------------------------------------


def output_dir(*, default: str = "data/xxx") -> Callable[..., Any]:
    """Return a click.option for the output directory."""
    return click.option(
        "--output-dir",
        type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=str),
        default=default,
        show_default=True,
        help="Output directory for generated images and annotations.",
    )


def num_samples(*, default: int = 5000) -> Callable[..., Any]:
    """Return a click.option for the number of samples to generate."""
    return click.option(
        "--num-samples",
        type=int,
        default=default,
        show_default=True,
        callback=validate_positive_int,
        help="Number of synthetic samples to generate.",
    )


def seed(*, default: int = 42) -> Callable[..., Any]:
    """Return a click.option for the random seed."""
    return click.option(
        "--seed",
        type=int,
        default=default,
        show_default=True,
        help="Random seed for reproducibility.",
    )


def workers(*, default: int = 0) -> Callable[..., Any]:
    """Return a click.option for the number of worker processes."""
    return click.option(
        "--workers",
        type=int,
        default=default,
        show_default=True,
        callback=validate_non_negative_int,
        help="Number of worker processes (0 = auto(CPU-1)).",
    )


def split(*, default: float = 0.9) -> Callable[..., Any]:
    """Return a click.option for the train/val split ratio."""
    return click.option(
        "--split",
        type=float,
        default=default,
        show_default=True,
        callback=validate_split,
        help="Train/val split ratio (0.0–1.0).",
    )


def watermark(*, default: bool = True) -> Callable[..., Any]:
    """Return a click.boolean_flag option for watermark application."""
    return click.option(
        "--watermark/--no-watermark",
        default=default,
        show_default=True,
        help="Apply a random watermark to generated images.",
    )


# ---------------------------------------------------------------------------
# Per-generator extra options
# ---------------------------------------------------------------------------


def min_structures(*, default: int = 2) -> Callable[..., Any]:
    """Return a click.option for the minimum number of structures per image."""
    return click.option(
        "--min-structures",
        type=int,
        default=default,
        show_default=True,
        callback=validate_positive_int,
        help="Minimum number of structures per image.",
    )


def max_structures(*, default: int = 10) -> Callable[..., Any]:
    """Return a click.option for the maximum number of structures per image."""
    return click.option(
        "--max-structures",
        type=int,
        default=default,
        show_default=True,
        callback=validate_positive_int,
        help="Maximum number of structures per image.",
    )


def structure_width_range(
    *, default: tuple[int, int] = (200, 350)
) -> Callable[..., Any]:
    """Return a click.option for the structure width range (min, max)."""
    return click.option(
        "--structure-width-range",
        type=(int, int),
        default=default,
        show_default=True,
        callback=validate_min_max,
        help="Structure width range as two integers (min max).",
    )


def structure_height_range(
    *, default: tuple[int, int] = (80, 140)
) -> Callable[..., Any]:
    """Return a click.option for the structure height range (min, max)."""
    return click.option(
        "--structure-height-range",
        type=(int, int),
        default=default,
        show_default=True,
        callback=validate_min_max,
        help="Structure height range as two integers (min max).",
    )


def structure_prob(*, default: float = 0.4) -> Callable[..., Any]:
    """Return a click.option for the probability a cell contains a structure."""
    return click.option(
        "--structure-prob",
        type=float,
        default=default,
        show_default=True,
        callback=validate_split,
        help="Probability a table cell contains a chemical structure (0.0–1.0).",
    )


def min_cols(*, default: int = 1) -> Callable[..., Any]:
    """Return a click.option for the minimum number of table columns."""
    return click.option(
        "--min-cols",
        type=int,
        default=default,
        show_default=True,
        callback=validate_positive_int,
        help="Minimum number of table columns.",
    )


def max_cols(*, default: int = 5) -> Callable[..., Any]:
    """Return a click.option for the maximum number of table columns."""
    return click.option(
        "--max-cols",
        type=int,
        default=default,
        show_default=True,
        callback=validate_positive_int,
        help="Maximum number of table columns.",
    )


def min_rows(*, default: int = 1) -> Callable[..., Any]:
    """Return a click.option for the minimum number of table rows."""
    return click.option(
        "--min-rows",
        type=int,
        default=default,
        show_default=True,
        callback=validate_positive_int,
        help="Minimum number of table rows.",
    )


def max_rows(*, default: int = 15) -> Callable[..., Any]:
    """Return a click.option for the maximum number of table rows."""
    return click.option(
        "--max-rows",
        type=int,
        default=default,
        show_default=True,
        callback=validate_positive_int,
        help="Maximum number of table rows.",
    )


def cell_width_range(*, default: tuple[int, int] = (100, 300)) -> Callable[..., Any]:
    """Return a click.option for the table cell width range (min, max)."""
    return click.option(
        "--cell-width-range",
        type=(int, int),
        default=default,
        show_default=True,
        callback=validate_min_max,
        help="Cell width range as two integers (min max).",
    )


def cell_height_range(*, default: tuple[int, int] = (80, 200)) -> Callable[..., Any]:
    """Return a click.option for the table cell height range (min, max)."""
    return click.option(
        "--cell-height-range",
        type=(int, int),
        default=default,
        show_default=True,
        callback=validate_min_max,
        help="Cell height range as two integers (min max).",
    )


def datasets(
    *, default: tuple[str, ...] = ("data/dense_layout", "data/table_layout")
) -> Callable[..., Any]:
    """Return a click.option for dataset directories (multiple allowed)."""
    return click.option(
        "--datasets",
        type=click.Path(file_okay=False, dir_okay=True, exists=False, path_type=str),
        multiple=True,
        default=default,
        show_default=True,
        help="Dataset directories to merge (can be specified multiple times).",
    )
