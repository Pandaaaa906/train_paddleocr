# Nested Table Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5% probability nested-table samples to `generate_table_layout.py` — an outer 1-column simple table wrapping a full inner table.

**Architecture:** Double-layer rendering: inner table rendered first to a temporary image (margin-free when nested), then pasted into the bottom cell of an outer 1-column table. Inner annotations are coordinate-shifted into new dicts to preserve immutability.

**Tech Stack:** Python 3.12, Pillow, RDKit, `prepare_traindata` toolkit

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `prepare_traindata/generate_table_layout.py` | Modify | Core generator: data model, config building, rendering pipeline |

---

## Prerequisites

Read these before touching code:
- Spec: `docs/superpowers/specs/2026-05-14-nested-table-layout-design.md`
- Existing generator: `prepare_traindata/generate_table_layout.py` (study `_generate_sample`, `SampleConfig`, `build_configs`)

---

## Task 1: Extend Data Model

**Files:**
- Modify: `prepare_traindata/generate_table_layout.py:76-93`

- [ ] **Step 1: Add `nested` and `header_texts` to `SampleConfig`**

  Append to the frozen dataclass:

  ```python
  nested: bool = False
  header_texts: tuple[str, ...] = ()
  ```

  Also add module-level constant near the top:

  ```python
  OUTER_CELL_PADDING: int = 10
  NESTED_PROB: float = 0.05
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add prepare_traindata/generate_table_layout.py
  git commit -m "feat: add nested table fields to SampleConfig"
  ```

---

## Task 2: Build Nested Configs

**Files:**
- Modify: `prepare_traindata/generate_table_layout.py:96-141`

- [ ] **Step 1: Add `text_vocab` import at top of file**

  ```python
  from prepare_traindata import text_vocab
  ```

  (Check if already imported inside `_generate_sample`; if so, move it to module level.)

- [ ] **Step 2: Modify `build_configs()` to generate nested samples**

  Inside the `for idx in range(num_samples):` loop, replace the simple column/row/border generation with:

  ```python
  nested = rng.random() < NESTED_PROB
  if nested:
      num_cols = rng.randint(3, 7)
      num_rows = rng.randint(3, 10)
      outer_header_rows = rng.randint(1, 2)
      header_texts = tuple(
          text_vocab.get_random_text(rng) for _ in range(outer_header_rows)
      )
  else:
      num_cols = rng.randint(min_cols, max_cols)
      num_rows = rng.randint(min_rows, max_rows)
      header_texts = ()

  col_widths = tuple(rng.randint(*cell_width_range) for _ in range(num_cols))
  row_heights = tuple(rng.randint(*cell_height_range) for _ in range(num_rows))
  border_width = rng.randint(2, 4)
  ```

  Then append the new fields to the `SampleConfig(...)` constructor:

  ```python
  SampleConfig(
      ...,
      nested=nested,
      header_texts=header_texts,
  )
  ```

  **Note:** Do NOT pre-compute outer row heights here. The inner table's rendered size (including its own margins) is not known until `_render_inner_table` runs. Outer dimensions are derived at render time.

- [ ] **Step 3: Commit**

  ```bash
  git add prepare_traindata/generate_table_layout.py
  git commit -m "feat: build_configs generates nested samples at 5% probability"
  ```

---

## Task 3: Inner Table Rendering Refactor

**Files:**
- Modify: `prepare_traindata/generate_table_layout.py:313-531`

We need to extract the inner-table rendering so it can be called twice (once standalone, once nested). The existing `_generate_sample` body (lines 313-531) becomes the inner rendering function.

### Step 1: Extract `_render_inner_table(cfg, margin=MARGIN, apply_watermark=True)`

Create a new function `_render_inner_table(cfg, margin, apply_watermark)` that contains the **entire** existing rendering logic of `_generate_sample` — everything from the RNG creation and canvas setup through the annotation building, stopping before the `SampleResult` wrapping and file saving.

It should accept a `SampleConfig` and return a tuple:

```python
def _render_inner_table(
    cfg: SampleConfig,
    margin: int = MARGIN,
    apply_watermark: bool = True,
) -> tuple[Any, list[dict[str, Any]]]:
    """Render the inner table image and raw annotations.

    Returns (pil_image, annotations_list).
    """
    from prepare_traindata import text_vocab, watermark_utils

    rng = random.Random(cfg.seed)
    total_table_w = sum(cfg.col_widths)
    total_table_h = sum(cfg.row_heights)
    canvas_w = margin * 2 + total_table_w
    canvas_h = margin * 2 + total_table_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    if apply_watermark and cfg.use_watermark:
        canvas = watermark_utils.apply_random_watermark(canvas, rng)

    draw = ImageDraw.Draw(canvas)

    # ... rest of existing rendering logic (cells, structures, text, borders) ...

    # Build annotations (same as existing annotation-building block in _generate_sample)
    annotations: list[dict[str, Any]] = []
    # table annotation
    # structure annotations

    return canvas, annotations
```

Key changes during extraction:
- Replace **every** hard-coded `MARGIN` inside the extracted block with the `margin` parameter so nested calls can pass `margin=0`. Audit the entire function body for `MARGIN` references (canvas size, table_x, table_y, cell positioning, etc.) and replace them all.
- Replace `if cfg.use_watermark:` with `if apply_watermark and cfg.use_watermark:` so nested samples skip inner watermark
- Remove the `SampleResult` wrapping, filename generation, and `out_path.parent.mkdir(...)` — return `canvas, annotations` instead
- Keep annotation building exactly as-is

### Step 2: Rewrite `_generate_sample` to use the extracted function

```python
def _generate_sample(cfg: SampleConfig) -> SampleResult | None:
    from PIL import Image, ImageDraw
    from prepare_traindata import watermark_utils

    if not cfg.nested:
        inner_img, inner_annotations = _render_inner_table(cfg)
        # Save directly (existing behaviour)
        filename = f"table_{cfg.sample_idx:06d}.png"
        images_dir = cfg.output_dir / "images"
        out_path = images_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        inner_img.save(out_path)
        return SampleResult(
            img_id=cfg.sample_idx,
            filename=filename,
            width=inner_img.size[0],
            height=inner_img.size[1],
            annotations=inner_annotations,
        )

    # ---- NESTED PATH ----
    # 1. Render inner table WITHOUT margin and WITHOUT watermark
    inner_img, inner_raw_annotations = _render_inner_table(
        cfg, margin=0, apply_watermark=False
    )
    inner_w, inner_h = inner_img.size

    # 2. Compute outer dimensions from actual inner size + header rows
    rng = random.Random(cfg.seed)
    header_heights = tuple(rng.randint(60, 100) for _ in cfg.header_texts)
    outer_total_h = sum(header_heights) + inner_h + OUTER_CELL_PADDING * 2
    outer_margin = MARGIN
    outer_w = outer_margin * 2 + inner_w + OUTER_CELL_PADDING * 2
    outer_h = outer_margin * 2 + outer_total_h

    canvas = Image.new("RGB", (outer_w, outer_h), (255, 255, 255))
    if cfg.use_watermark:
        rng = random.Random(cfg.seed)
        canvas = watermark_utils.apply_random_watermark(canvas, rng)

    draw = ImageDraw.Draw(canvas)

    # 3. Draw outer table (no annotation for outer frame per spec)
    table_x = outer_margin
    table_y = outer_margin
    table_w = inner_w + OUTER_CELL_PADDING * 2
    table_h = outer_total_h

    # Header cells
    y_cursor = table_y
    for text, row_h in zip(cfg.header_texts, header_heights):
        font_size = max(12, row_h // 3)
        font = _load_font(font_size)
        text_color = (0, 0, 0) if rng.random() < 0.5 else (30, 30, 30)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = table_x + (table_w - tw) // 2
        ty = y_cursor + (row_h - th) // 2
        if font is not None:
            draw.text((tx, ty), text, fill=text_color, font=font)
        else:
            draw.text((tx, ty), text, fill=text_color)

        # Bottom border
        y_cursor += row_h
        draw.line(
            [(table_x, y_cursor), (table_x + table_w, y_cursor)],
            fill=(0, 0, 0),
            width=1,
        )

    # 4. Paste inner table into bottom cell
    paste_x = table_x + OUTER_CELL_PADDING
    paste_y = y_cursor + OUTER_CELL_PADDING
    canvas.paste(inner_img, (paste_x, paste_y))

    # 5. Draw full outer rectangle
    draw.rectangle(
        [table_x, table_y, table_x + table_w, table_y + table_h],
        outline=(0, 0, 0),
        width=1,
    )

    # 6. Translate inner annotations into NEW dicts (immutable)
    translated_annotations: list[dict[str, Any]] = []
    for ann in inner_raw_annotations:
        new_ann = dict(ann)
        new_ann["bbox"] = [
            ann["bbox"][0] + paste_x,
            ann["bbox"][1] + paste_y,
            ann["bbox"][2],
            ann["bbox"][3],
        ]
        old_seg = ann["segmentation"][0]
        new_seg = []
        for i in range(0, len(old_seg), 2):
            new_seg.append(old_seg[i] + paste_x)
            new_seg.append(old_seg[i + 1] + paste_y)
        new_ann["segmentation"] = [new_seg]
        translated_annotations.append(new_ann)

    # 7. Save
    filename = f"table_{cfg.sample_idx:06d}.png"
    images_dir = cfg.output_dir / "images"
    out_path = images_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    return SampleResult(
        img_id=cfg.sample_idx,
        filename=filename,
        width=outer_w,
        height=outer_h,
        annotations=translated_annotations,
    )
```

- [ ] **Step 3: Commit**

  ```bash
  git add prepare_traindata/generate_table_layout.py
  git commit -m "feat: extract _render_inner_table and implement nested path"
  ```

---

## Task 4: Patch Configs in `main()`

**Files:**
- Modify: `prepare_traindata/generate_table_layout.py:595-610`

The config patching loop at the bottom of `main()` must copy the new fields.

- [ ] **Step 1: Add new fields to the patching list**

  ```python
  configs = [
      SampleConfig(
          sample_idx=c.sample_idx,
          seed=c.seed,
          num_cols=c.num_cols,
          num_rows=c.num_rows,
          col_widths=c.col_widths,
          row_heights=c.row_heights,
          border_width=c.border_width,
          cells=c.cells,
          output_dir=output_path,
          use_watermark=watermark,
          nested=c.nested,
          header_texts=c.header_texts,
      )
      for c in configs
  ]
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add prepare_traindata/generate_table_layout.py
  git commit -m "chore: copy nested fields in config patching loop"
  ```

---

## Task 5: Verification

**Files:**
- Modify: none (read-only)

- [ ] **Step 1: Generate a small test batch**

  ```bash
  uv run python -m prepare_traindata.generate_table_layout --num-samples 20 --seed 999 --output-dir data/table_layout_test --workers 1
  ```

  With seed 999 and 20 samples, probabilistically ~1 nested sample should appear.

- [ ] **Step 2: Inspect outputs**

  ```bash
  ls data/table_layout_test/images/
  ```

  Expected: 20 PNG files.

  Open a few images to visually confirm:
  - Most samples look like normal `table_layout` (multi-column inner tables only)
  - ~1 sample has an outer 1-column frame with header text and an inner table

- [ ] **Step 3: Validate annotations**

  ```bash
  uv run python -c "
  import orjson
  from pathlib import Path
  ann = orjson.loads(Path('data/table_layout_test/annotations/instance_train.json').read_bytes())
  print(f'Images: {len(ann[\"images\"])}')
  print(f'Table boxes: {sum(1 for a in ann[\"annotations\"] if a[\"category_id\"] == 21)}')
  print(f'Image boxes: {sum(1 for a in ann[\"annotations\"] if a[\"category_id\"] == 14)}')
  # Show dimensions of the largest image (likely nested)
  sizes = sorted([(img['width'] * img['height'], img['width'], img['height']) for img in ann['images']], reverse=True)
  print('Largest images (w x h):', sizes[:3])
  "
  ```

- [ ] **Step 4: Run PaddleX check_dataset (optional but recommended)**

  Temporarily change the dataset dir in the config:

  ```bash
  # Windows (PowerShell)
  (Get-Content configs/PP-DocLayoutV3.yaml) -replace 'dataset_dir: .*', 'dataset_dir: "data/table_layout_test"' | Set-Content configs/PP-DocLayoutV3.yaml
  # macOS / Linux
  sed -i 's|dataset_dir: .*|dataset_dir: "data/table_layout_test"|' configs/PP-DocLayoutV3.yaml
  ```

  Then run validation:

  ```bash
  uv run python main.py --mode check_dataset
  ```

  Expected: `Check dataset passed !`

  Remember to revert `configs/PP-DocLayoutV3.yaml` afterwards.

- [ ] **Step 5: Commit**

  ```bash
  git add -A
  git commit -m "test: verify nested table layout generation"
  ```

---

## Task 6: Regenerate Full Dataset (Optional, Post-Merge)

Once the code is merged and verified:

- [ ] Regenerate `table_layout` dataset
- [ ] Re-run `merge_datasets.py` to update `merged_all`
- [ ] Re-run `main.py --mode check_dataset` on the merged dataset

---

## Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| `_render_inner_table` extraction breaks non-nested path | Non-nested path calls `_render_inner_table(cfg)` with defaults, identical to old code; visual verification in Task 5 confirms |
| Annotation coordinate shift is wrong | Task 5 Step 3 validates bbox counts and image dimensions; visual overlay of bbox on image confirms alignment |
| Watermark applied twice | `_render_inner_table` accepts `apply_watermark=False` for nested path; outer canvas handles watermark exclusively |
| Double margin inside outer cell | `_render_inner_table` accepts `margin=0` for nested path; outer `OUTER_CELL_PADDING` is the only margin |
| Worker spawn fails because `text_vocab` moved to module level | Test with `--workers 4` in Task 5; if spawn fails, move import back inside function |
| File grows too large (>800 lines) | Already at risk; if extraction pushes past 800, consider splitting `_render_inner_table` into a private module in a follow-up refactor |
