# Data Generation CLI Refactoring Design

## Overview

Refactor the three `generate_*.py` scripts and `merge_datasets.py` to accept command-line arguments via `click`, eliminating hard-coded constants. Integrate train/val split and watermarking directly into each generator. Replace `json` with `orjson` for faster serialization.

## Goals

1. **Parameterize generators** — All constants (`NUM_SAMPLES`, `OUTPUT_DIR`, `SEED`, etc.) become CLI flags.
2. **Unified CLI interface** — Shared common options via `prepare_traindata/cli.py` decorators.
3. **Built-in train/val split** — Each generator outputs both `instance_train.json` and `instance_val.json` directly, no separate `remap_and_split.py` step.
4. **Watermark integration** — `generate_dense_layout.py` and `generate_synthetic_layout.py` gain watermark support (default on).
5. **`orjson` adoption** — All JSON I/O uses `orjson` for performance.
6. **Parameterize `merge_datasets.py`** — Accept `--datasets`, `--output-dir`, `--split`, `--seed`.

## Non-Goals

- Remove `remap_and_split.py` entirely (it still has value for legacy workflows).
- Change output COCO format semantics (bbox, segmentation, read_order remain unchanged).

## Category ID Policy

All generators output **final PaddleX category IDs** directly, using the full 25-class `CATEGORIES` list from `prepare_traindata/categories.py`:

- `generate_dense_layout.py` → outputs `image` (id=14)
- `generate_synthetic_layout.py` → outputs `image` (id=14)
- `generate_table_layout.py` → outputs `table` (id=21) + `image` (id=14)

No remapping step is required. The `categories` field in every output JSON contains all 25 PP-DocLayoutV3 classes.

## CLI Validation

All range options must be validated via `click` callbacks:

| Option | Constraint | Error Message |
|--------|-----------|---------------|
| `--split` | `0.0 <= split <= 1.0` | `"split must be between 0.0 and 1.0"` |
| `--min-structures` / `--max-structures` | `min <= max` | `"min-structures must be <= max-structures"` |
| `--min-cols` / `--max-cols` | `min <= max` | `"min-cols must be <= max-cols"` |
| `--min-rows` / `--max-rows` | `min <= max` | `"min-rows must be <= max-rows"` |
| `--structure-width-range` | `min <= max` | `"width min must be <= max"` |
| `--structure-height-range` | `min <= max` | `"height min must be <= max"` |
| `--cell-width-range` | `min <= max` | `"cell-width min must be <= max"` |
| `--cell-height-range` | `min <= max` | `"cell-height min must be <= max"` |
| `--workers` | `workers >= 0` | `"workers must be >= 0"` |
| `--num-samples` | `num_samples > 0` | `"num-samples must be > 0"` |

## Edge Cases

- `--split 0.0` → write only `instance_train.json` (all images)
- `--split 1.0` → write only `instance_train.json` (all images), no val file
- `--workers 0` → resolve to `max(1, (os.cpu_count() or 4) - 1)`

## Module Design

### `prepare_traindata/cli.py`

Shared `click` options as reusable decorators:

```python
import click

output_dir = click.option("--output-dir", "-o", default="data/xxx", type=click.Path(), help="Output directory")
num_samples = click.option("--num-samples", "-n", default=5000, type=int, help="Number of samples")
seed = click.option("--seed", default=42, type=int, help="Random seed")
workers = click.option("--workers", "-j", default=0, type=int, help="Worker processes (0 = auto)")
split = click.option("--split", default=0.9, type=float, help="Train split ratio (0 = no split)")
watermark = click.option("--watermark/--no-watermark", default=True, help="Enable watermark")

def cli_options(func):
    """Apply all common options to a click command."""
    for opt in [output_dir, num_samples, seed, workers, split, watermark]:
        func = opt(func)
    return func
```

### `generate_dense_layout.py` CLI

```bash
uv run python -m prepare_traindata.generate_dense_layout \
    --num-samples 5000 \
    --output-dir data/dense_layout \
    --seed 42 \
    --workers 0 \
    --split 0.9 \
    --watermark \
    --min-structures 2 \
    --max-structures 10 \
    --structure-width-range 200 350 \
    --structure-height-range 80 140
```

Default values match current hard-coded constants. Note: `generate_table_layout.py` previously defaulted to `split=0.0` (no val split); this is preserved.

### `generate_table_layout.py` CLI

```bash
uv run python -m prepare_traindata.generate_table_layout \
    --num-samples 2500 \
    --output-dir data/table_layout \
    --seed 42 \
    --workers 0 \
    --split 0.0 \
    --watermark \
    --structure-prob 0.4 \
    --min-cols 1 --max-cols 5 \
    --min-rows 1 --max-rows 15 \
    --cell-width-range 100 300 \
    --cell-height-range 80 200
```

### `generate_synthetic_layout.py` CLI

```bash
uv run python -m prepare_traindata.generate_synthetic_layout \
    --num-samples 5000 \
    --output-dir data/synthetic_chem \
    --seed 42 \
    --workers 0 \
    --split 0.9 \
    --watermark
```

### `merge_datasets.py` CLI

```bash
uv run python -m prepare_traindata.merge_datasets \
    --datasets data/dense_layout data/table_layout \
    --output-dir data/merged_all \
    --split 0.9 \
    --seed 42
```

## Train/Val Split Logic

All three generators follow the same split pattern:

1. Generate all `num_samples` COCO records in memory.
2. Shuffle image list with `random.Random(seed)`.
3. If `split > 0`:
   - `n_train = int(len(images) * split)`
   - Write `instance_train.json` + `instance_val.json`
4. If `split == 0`:
   - Write only `instance_train.json` (all images)

Annotations are partitioned by `image_id` into the matching split file.

## Watermark Integration

All three generators support `--watermark/--no-watermark` (default `True`):
- `generate_dense_layout.py` and `generate_synthetic_layout.py` newly gain watermark support.
- `generate_table_layout.py` already has watermark support; refactored to respect the CLI toggle.
- Watermark is applied immediately after `Image.new('RGB', ...)` and before any structure/text placement.

## `orjson` Usage

Hard dependency: `import orjson` at module top (no fallback). `orjson` is listed in `pyproject.toml` and available after `uv sync`.

Usage:
- Save: `orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()`
- Load: `orjson.loads(path.read_bytes())`

## `generate_synthetic_layout.py` Additional Requirements

The legacy generator currently lacks `read_order` and `segmentation` fields. As part of this refactor, it must add:
- `segmentation`: rectangle polygon synthesized from `bbox` (`[[x, y, x+w, y, x+w, y+h, x, y+h]]`)
- `read_order`: 0-based per image, sorted top-to-bottom, left-to-right
- Output filename changed from `instances_train.json` (plural) to `instance_train.json` (singular) to match other generators.

## `merge_datasets.py` Behavior

- Skips missing split files gracefully (prints "Skipping {path} (not found)").
- Imports must use absolute package paths: `from prepare_traindata.categories import CATEGORIES`.

## File Structure Changes

```
prepare_traindata/
├── cli.py                    # NEW — shared click options
├── categories.py             # EXISTING — shared 25-class list
├── text_vocab.py             # EXISTING — random chemistry vocabulary
├── watermark_utils.py        # EXISTING — watermark generators
├── generate_dense_layout.py  # REFACTORED — click CLI, split, watermark, orjson
├── generate_synthetic_layout.py  # REFACTORED — click CLI, split, watermark, orjson, read_order, segmentation
├── generate_table_layout.py  # REFACTORED — click CLI, split, orjson, watermark toggle
├── merge_datasets.py         # REFACTORED — click CLI, orjson, absolute imports
├── remap_and_split.py        # EXISTING — kept for backward compat
├── image.py                  # EXISTING
└── rdkit_chem.py             # EXISTING
```

## Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing ...
    "orjson>=3.10",
    "click>=8.0",
]
```

## Backward Compatibility

Default CLI values match previous hard-coded constants where applicable. Some output differences are expected due to added `read_order`, `segmentation`, corrected filenames, and watermarking.

## Testing Plan

1. Run each generator with `--num-samples 5` and verify JSON structure.
2. Verify `instance_train.json` + `instance_val.json` are created when `--split > 0`.
3. Verify `--split 0.0` outputs only `instance_train.json` (no val).
4. Verify `--split 1.0` outputs only `instance_train.json` (no val).
5. Verify `--no-watermark` produces clean backgrounds.
6. Run `merge_datasets.py` with `--datasets` and verify merged output.
7. Validate `orjson` output is valid JSON (round-trip `orjson.loads(orjson.dumps(data)) == data`).
8. Verify all generators use `CATEGORIES` from `categories.py` (25 classes).
9. Verify `generate_synthetic_layout.py` output includes `read_order` and `segmentation` fields.
10. Verify CLI validation rejects invalid ranges (e.g., `--split 1.5`, `--min-structures 10 --max-structures 2`).
