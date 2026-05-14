# Nested Table Layout Design Spec

## 1. Overview

Add a 5% probability to `generate_table_layout.py` to produce "nested-table" samples:
an outer 1-column, 1-3-row simple table (1 px border) where the bottom cell contains a full inner table with chemical structures and text.

The outer table itself is **not annotated** as `table` or `text`; only the inner table and its chemical structures receive standard PP-DocLayoutV3 annotations (`table` = 21, `image` = 14).

## 2. Motivation

Real-world chemical documents (e.g. 询价表 / quotation sheets) frequently contain a simple outer form/table that wraps a detailed inner table. Current `table_layout` generator never produces this layout, reducing layout diversity and hurting generalisation.

## 3. Data-Model Changes

### 3.1 `SampleConfig` (frozen dataclass)

Two new fields appended at the end:

```python
nested: bool = False
header_texts: tuple[str, ...] = ()
```

### 3.2 `build_configs()`

Inside the per-sample loop:

```python
nested = rng.random() < 0.05
if nested:
    num_cols = rng.randint(3, 7)          # inner table columns
    num_rows = rng.randint(3, 10)         # inner table rows
    outer_rows = rng.randint(2, 3)        # outer table rows (header cells + 1 inner)
    header_texts = tuple(
        text_vocab.get_random_text(rng) for _ in range(outer_rows - 1)
    )
    # outer cell width = inner total width + 2 * OUTER_CELL_PADDING
    # outer header cell height = rng.randint(60, 100)
    # outer bottom cell height = inner total height + 2 * OUTER_CELL_PADDING
else:
    # existing logic unchanged
```

The existing `num_cols` / `num_rows` continue to describe the **inner** table dimensions; `border_width` also belongs to the inner table.

## 4. Rendering Architecture

### 4.1 Double-Layer Rendering (chosen approach)

```
┌─────────────────────────────┐
│  [header text cell 1]       │  ← outer row 0
├─────────────────────────────┤
│  [header text cell 2]       │  ← outer row 1 (optional)
├─────────────────────────────┤
│  ┌─────┬─────┬─────┐       │
│  │ SM  │ txt │ SM  │       │  ← inner table
│  ├─────┼─────┼─────┤       │
│  │ txt │ SM  │ txt │       │
│  └─────┴─────┴─────┘       │
└─────────────────────────────┘
```

`OUTER_MARGIN = 20` (canvas margin)  
`OUTER_CELL_PADDING = 10`

### 4.2 Rendering Pipeline (`_generate_sample`)

**Step 1 — Inner table render**

If `cfg.nested`:
- Create a temporary `SampleConfig` identical to `cfg` but with `nested=False`
- Call the existing inner-table rendering block to produce:
  - `inner_img` (PIL Image)
  - `inner_annotations` (list of dicts)

**Step 2 — Outer canvas creation**

```
outer_w = OUTER_MARGIN * 2 + inner_total_w + OUTER_CELL_PADDING * 2
outer_h = OUTER_MARGIN * 2 + sum(outer_row_heights)
```

White background, optional watermark applied on the **outer** canvas only.

**Step 3 — Draw outer table**

- For each header cell:
  - Draw text centred (font size `max(12, row_h // 3)`, black)
  - Draw 1 px bottom border
- Draw full 1 px outer rectangle around the whole outer table

**Step 4 — Paste inner table**

```
paste_x = OUTER_MARGIN + OUTER_CELL_PADDING
paste_y = OUTER_MARGIN + sum(header_row_heights) + OUTER_CELL_PADDING
```

**Step 5 — Coordinate translation**

For every `inner_annotation`:
- `bbox[0] += paste_x`, `bbox[1] += paste_y`
- Each segmentation vertex pair `x += paste_x, y += paste_y`

`read_order` values remain unchanged.

### 4.3 Annotation Rules

| Element | Annotated? | Category | Notes |
|---------|------------|----------|-------|
| Outer table | No | — | Deliberately omitted |
| Header text cells | No | — | Deliberately omitted |
| Inner table | Yes | 21 (`table`) | Single bbox for whole inner table |
| Chemical structures inside inner table | Yes | 14 (`image`) | One per structure, centred in cell |

## 5. Visual Details

- Outer border width: **1 px** solid black
- Inner table border width: unchanged (2–4 px), creating a subtle visual hierarchy
- Outer header text colour: `(0, 0, 0)` or `(30, 30, 30)` (random)
- Watermark: applied once on the outer canvas, covering the full image
- Cell backgrounds: watermark visible when present, otherwise white

## 6. Backwards Compatibility

- `nested=False` (95% of samples) executes **exactly** the existing code path
- No changes to non-nested `SampleConfig` field ordering
- `build_configs()` signature and defaults unchanged
- CLI options unchanged; no new flags required

## 7. Open Questions / Future Work

1. Should header text cells ever be annotated as `text` (cat 22) to improve text-region detection? **Decision: No for now** — keep annotations focused on the inner table.
2. Should the outer table itself be annotated as `table`? **Decision: No** — the inner table is the semantically rich table; the outer frame is layout chrome.
