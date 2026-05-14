# Data Generation CLI Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the three `generate_*.py` scripts and `merge_datasets.py` to accept command-line arguments via `click`, integrate train/val split and watermarking, replace `json` with `orjson`, and share common code through `prepare_traindata/cli.py`.

**Architecture:** Each generator becomes a standalone `click` CLI script. Common options (output_dir, num_samples, seed, workers, split, watermark) are defined in `prepare_traindata/cli.py` as individual `click.option` callables that can be composed per-command. Each generator passes its own defaults (e.g., table_layout uses split=0.0, others use 0.9). Worker functions receive all configuration through `SampleConfig` dataclasses — no module-level globals are read inside workers. Generators output final PaddleX category IDs directly.

**Tech Stack:** Python 3.12, click, orjson, Pillow, RDKit, multiprocessing

---

## File Structure

### New Files
- `prepare_traindata/cli.py` — Shared `click` options and validation callbacks

### Modified Files
- `prepare_traindata/generate_dense_layout.py` — CLI, watermark, split, orjson, seed in SampleConfig, category_id=14
- `prepare_traindata/generate_table_layout.py` — CLI, watermark toggle, split, orjson, seed in SampleConfig
- `prepare_traindata/generate_synthetic_layout.py` — CLI, watermark, split, orjson, read_order, segmentation, CATEGORIES, seed in SampleConfig, rename output file
- `prepare_traindata/merge_datasets.py` — CLI, orjson, absolute imports, split edge cases
- `prepare_traindata/remap_and_split.py` — Keep for backward compat
- `pyproject.toml` — Add `orjson` and `click` dependencies

---

## Critical Rules

### orjson Pattern
Save: `path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))`
Load: `data = orjson.loads(path.read_bytes())`
Never use `.decode()`.

### Train/Val Split Logic
```python
if 0.0 < split < 1.0:
    n_train = int(len(images) * split)
    # write train + val
else:
    # write only instance_train.json
```

### Worker Process Safety (Windows spawn)
- All CLI-parameterized values must be passed through `SampleConfig` dataclass/NamedTuple — never read from mutated module globals inside `_generate_sample`.
- `IMAGES_DIR` must be passed inside `SampleConfig` (add `output_dir: Path` field).
- `seed` must be in `SampleConfig` so watermark RNG is deterministic per sample.
- `build_configs` must accept all range parameters as arguments.

### cli.py Design
Do NOT create a monolithic `cli_options` decorator with hardcoded defaults. Instead:
- Expose individual `click.option` callables from `cli.py`.
- Each generator composes them explicitly with its own defaults.
- Expose `validate_split` as a standalone function for reuse.

Example:
```python
from prepare_traindata.cli import output_dir, num_samples, seed, workers, split, watermark, validate_split

@click.command()
@output_dir(default="data/dense_layout")
@num_samples(default=5000)
@seed(default=42)
@workers(default=0)
@split(default=0.9)
@watermark(default=True)
def main(output_dir, num_samples, seed, workers, split, watermark):
    ...
```

---

### Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `orjson` and `click` to dependencies**

  Edit `pyproject.toml` and add `"orjson>=3.10"` and `"click>=8.0"` to the `dependencies` list.

- [ ] **Step 2: Sync dependencies**

  Run: `uv sync`
  Expected: `orjson` and `click` are installed in `.venv`.

- [ ] **Step 3: Commit**

  ```bash
  git add pyproject.toml uv.lock
  git commit -m "chore: add click and orjson dependencies"
  ```

---

### Task 2: Create Shared CLI Module

**Files:**
- Create: `prepare_traindata/cli.py`

- [ ] **Step 1: Write `prepare_traindata/cli.py`**

  Create the file with:
  - `validate_split(ctx, param, value)` callback
  - `validate_min_max(ctx, param, value)` helper for tuple ranges
  - Individual `click.option` callables that accept a `default` parameter:
    ```python
    def output_dir(default="data/xxx"):
        return click.option("--output-dir", "-o", default=default, type=click.Path(), help="Output directory")
    # ... num_samples, seed, workers, split, watermark similarly ...
    ```

- [ ] **Step 2: Verify import and defaults**

  Run:
  ```bash
  uv run python -c "
  from prepare_traindata.cli import output_dir, num_samples, seed, workers, split, watermark, validate_split
  print('Import OK')
  "
  ```
  Expected: `Import OK`

- [ ] **Step 3: Commit**

  ```bash
  git add prepare_traindata/cli.py
  git commit -m "feat: add shared click CLI options module"
  ```

---

### Task 3: Refactor `generate_dense_layout.py`

**Files:**
- Modify: `prepare_traindata/generate_dense_layout.py`

- [ ] **Step 1: Add seed to SampleConfig**

  ```python
  @dataclass(frozen=True)
  class SampleConfig:
      sample_idx: int
      seed: int
      smiles: list[str]
      sizes: list[tuple[int, int]]
      num_cols: int
      output_dir: Path
  ```

- [ ] **Step 2: Parameterize build_configs**

  Update signature to accept `min_structures`, `max_structures`, `structure_width_range`, `structure_height_range`.
  Pass these into each `SampleConfig` instead of reading module globals.

- [ ] **Step 3: Add watermark RNG in _generate_sample**

  Create `rng = random.Random(cfg.seed)` at start of `_generate_sample`.
  Apply watermark after canvas creation if enabled:
  ```python
  canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
  if use_watermark:
      canvas = watermark_utils.apply_random_watermark(canvas, rng)
  ```

- [ ] **Step 4: Fix IMAGES_DIR in worker**

  In `_generate_sample`, use `cfg.output_dir / "images"` instead of global `IMAGES_DIR`.

- [ ] **Step 5: Change category_id to 14**

  Change `CATEGORY_ID: int = 0` to `CAT_ID_IMAGE` (import from `categories.py`).
  Update annotation `category_id` and use `CATEGORIES` for output JSON.

- [ ] **Step 6: Add click CLI**

  Compose options from `cli.py` with defaults:
  - `output_dir(default="data/dense_layout")`
  - `num_samples(default=5000)`
  - `split(default=0.9)`
  - Plus per-generator options: `--min-structures`, `--max-structures`, `--structure-width-range`, `--structure-height-range`

- [ ] **Step 7: Add train/val split in main()**

  Use `0.0 < split < 1.0` check. Write `instance_train.json` + `instance_val.json` when splitting, else only `instance_train.json`.
  Use `orjson` for save: `path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))`

- [ ] **Step 8: Replace json with orjson**

  All JSON I/O in the file.

- [ ] **Step 9: Verify with quick test**

  ```bash
  uv run python -m prepare_traindata.generate_dense_layout --num-samples 5 --split 0.9 --watermark --workers 2
  ```
  Expected: 5 PNGs, train+val JSONs, no pickling errors.

- [ ] **Step 10: Commit**

  ```bash
  git add prepare_traindata/generate_dense_layout.py
  git commit -m "refactor: parameterize generate_dense_layout with click CLI, watermark, split, orjson"
  ```

---

### Task 4: Refactor `generate_table_layout.py`

**Files:**
- Modify: `prepare_traindata/generate_table_layout.py`

- [ ] **Step 1: Add seed to SampleConfig**

  Add `seed: int` and `output_dir: Path` fields.

- [ ] **Step 2: Parameterize build_configs**

  Accept `structure_prob`, `min_cols`, `max_cols`, `min_rows`, `max_rows`, `cell_width_range`, `cell_height_range`.

- [ ] **Step 3: Fix IMAGES_DIR in worker**

  Use `cfg.output_dir / "images"` in `_generate_sample`.

- [ ] **Step 4: Add watermark toggle**

  Wrap existing `apply_random_watermark` call with `if use_watermark:`.

- [ ] **Step 5: Add click CLI**

  Defaults:
  - `output_dir(default="data/table_layout")`
  - `num_samples(default=2500)`
  - `split(default=0.0)`  # Note: preserved from current behavior
  - Plus: `--structure-prob`, `--min-cols`, `--max-cols`, `--min-rows`, `--max-rows`, `--cell-width-range`, `--cell-height-range`

- [ ] **Step 6: Add train/val split**

  Same `0.0 < split < 1.0` logic.

- [ ] **Step 7: Replace json with orjson**

  All JSON I/O.

- [ ] **Step 8: Verify with quick test**

  ```bash
  uv run python -m prepare_traindata.generate_table_layout --num-samples 5 --split 0.0 --workers 2
  ```
  Expected: 5 PNGs, only `instance_train.json`, no pickling errors.

- [ ] **Step 9: Commit**

  ```bash
  git add prepare_traindata/generate_table_layout.py
  git commit -m "refactor: parameterize generate_table_layout with click CLI, watermark toggle, split, orjson"
  ```

---

### Task 5: Refactor `generate_synthetic_layout.py`

**Files:**
- Modify: `prepare_traindata/generate_synthetic_layout.py`

- [ ] **Step 1: Add seed to SampleConfig**

  Add `seed: int` and `output_dir: Path` fields.

- [ ] **Step 2: Parameterize build_configs**

  Accept `min_structures`, `max_structures`, `canvas_width_range`, `canvas_height_range`, `structure_width_range`, `structure_height_range`.

- [ ] **Step 3: Fix IMAGES_DIR in worker**

  Use `cfg.output_dir / "images"` in `_generate_sample`.

- [ ] **Step 4: Add watermark support**

  After canvas creation:
  ```python
  rng = random.Random(cfg.seed)
  canvas = Image.new("RGB", canvas_size, (255, 255, 255))
  if use_watermark:
      canvas = watermark_utils.apply_random_watermark(canvas, rng)
  ```

- [ ] **Step 5: Add read_order and segmentation**

  Store placements as `(x, y, w, h)` tuples (or NamedTuple) during random placement.
  After all placements, sort by `(y, x)` and assign `read_order = i`.
  Add `segmentation` as rectangle polygon from `bbox`.
  Add `iscrowd = 0` and `area = w * h`.

- [ ] **Step 6: Fix output filename**

  Change from `instances_train.json` to `instance_train.json`.

- [ ] **Step 7: Use CATEGORIES**

  Replace hardcoded single-category list with `CATEGORIES` from `categories.py`.

- [ ] **Step 8: Add click CLI**

  Defaults:
  - `output_dir(default="data/synthetic_chem")`
  - `num_samples(default=5000)`
  - `split(default=0.9)`

- [ ] **Step 9: Add train/val split**

  Same `0.0 < split < 1.0` logic.

- [ ] **Step 10: Replace json with orjson**

  All JSON I/O.

- [ ] **Step 11: Verify with quick test**

  ```bash
  uv run python -m prepare_traindata.generate_synthetic_layout --num-samples 5 --split 0.9 --watermark --workers 2
  ```
  Expected: 5 PNGs, train+val JSONs, annotations have `read_order`, `segmentation`, `category_id=14`, 25 categories.

- [ ] **Step 12: Commit**

  ```bash
  git add prepare_traindata/generate_synthetic_layout.py
  git commit -m "refactor: parameterize generate_synthetic_layout with click CLI, watermark, split, orjson, read_order, segmentation"
  ```

---

### Task 6: Refactor `merge_datasets.py`

**Files:**
- Modify: `prepare_traindata/merge_datasets.py`

- [ ] **Step 1: Update imports**

  Add `click`, `orjson`, `prepare_traindata.categories`.

- [ ] **Step 2: Add click CLI**

  ```python
  @click.command()
  @click.option("--datasets", "-d", multiple=True, required=True, type=click.Path(exists=True), help="Input dataset dirs")
  @click.option("--output-dir", "-o", default="data/merged_all", type=click.Path(), help="Output dir")
  @click.option("--split", default=0.9, type=float, callback=validate_split)
  @click.option("--seed", default=42, type=int)
  def main(datasets, output_dir, split, seed):
      ...
  ```

- [ ] **Step 3: Update split logic**

  Use `0.0 < split < 1.0` check. Write only `instance_train.json` when split is 0 or 1.

- [ ] **Step 4: Replace json with orjson**

  All JSON I/O.

- [ ] **Step 5: Verify with quick test**

  ```bash
  uv run python -m prepare_traindata.merge_datasets \
    -d data/dense_layout -d data/table_layout \
    -o data/merged_all_test --split 0.9 --seed 42
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add prepare_traindata/merge_datasets.py
  git commit -m "refactor: parameterize merge_datasets with click CLI, orjson, absolute imports"
  ```

---

### Task 7: Full Integration Test

**Files:**
- All modified generators and merge

- [ ] **Step 1: Regenerate all datasets (small batches)**

  ```bash
  rm -rf data/dense_layout/images/* data/dense_layout/annotations/*
  uv run python -m prepare_traindata.generate_dense_layout --num-samples 5 --split 0.9 --watermark --workers 2

  rm -rf data/table_layout/images/* data/table_layout/annotations/*
  uv run python -m prepare_traindata.generate_table_layout --num-samples 5 --split 0.0 --workers 2

  rm -rf data/synthetic_chem/images/* data/synthetic_chem/annotations/*
  uv run python -m prepare_traindata.generate_synthetic_layout --num-samples 5 --split 0.9 --watermark --workers 2
  ```

- [ ] **Step 2: Merge and validate**

  ```bash
  rm -rf data/merged_all_test/*
  uv run python -m prepare_traindata.merge_datasets \
    -d data/dense_layout -d data/table_layout -d data/synthetic_chem \
    -o data/merged_all_test --split 0.9 --seed 42
  ```

  Validate:
  ```bash
  uv run python -c "
  import orjson
  from pathlib import Path
  for split in ['train', 'val']:
      p = Path(f'data/merged_all_test/annotations/instance_{split}.json')
      data = orjson.loads(p.read_bytes())
      print(f'{split}: images={len(data[\"images\"])}, boxes={len(data[\"annotations\"])}, cats={len(data[\"categories\"])}')
      assert len(data['categories']) == 25, 'Expected 25 categories'
  print('All OK')
  "
  ```

- [ ] **Step 3: Test CLI validation and help**

  ```bash
  uv run python -m prepare_traindata.generate_dense_layout --split 1.5 2>&1 | grep "split must be"
  uv run python -m prepare_traindata.generate_dense_layout --help
  uv run python -m prepare_traindata.generate_table_layout --help
  uv run python -m prepare_traindata.generate_synthetic_layout --help
  uv run python -m prepare_traindata.merge_datasets --help
  ```

- [ ] **Step 4: Test --no-watermark**

  ```bash
  uv run python -m prepare_traindata.generate_dense_layout --num-samples 1 --no-watermark --output-dir data/test_no_wm
  ```
  Verify no exception.

- [ ] **Step 5: Commit test artifacts**

  ```bash
  git add data/merged_all_test/annotations/
  git commit -m "test: verify full generator + merge pipeline"
  ```

---

## Success Criteria

- [ ] All three generators accept `--help` and print valid options.
- [ ] All three generators produce `instance_train.json` with valid COCO format.
- [ ] `--split` between 0 and 1 (exclusive) produces both train and val.
- [ ] `--split 0` or `--split 1` produces only `instance_train.json`.
- [ ] `--watermark` and `--no-watermark` both work without error.
- [ ] `--workers 2` works on Windows (no pickling errors).
- [ ] `merge_datasets.py` accepts multiple `--datasets` and merges correctly.
- [ ] All JSON files use `orjson` and contain 25 categories.
- [ ] `generate_synthetic_layout.py` output includes `read_order` and `segmentation`.
- [ ] `generate_dense_layout.py` outputs `category_id=14` (not 0).
