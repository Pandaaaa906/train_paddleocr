# Table Layout Border Dropout Design

## 1. Problem

Current `generate_table_layout.py` always draws a complete outer border around every table. Real documents sometimes have tables with missing outer borders (top/bottom/left/right individually).

## 2. Goal

Each outer border edge (top, bottom, left, right) has an **independent 5% chance** of not being drawn. Inner borders remain unchanged. Table `bbox`/`segmentation` stay at the full table region regardless.

**Exception**: nested tables' **outer frame** always draws all 4 edges completely.

## 3. Implementation

### 3.1 `_render_inner_table` signature change

```python
def _render_inner_table(
    cfg: SampleConfig,
    margin: int = MARGIN,
    apply_watermark: bool = True,
    outer_edges: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> tuple[Any, list[dict[str, Any]]]:
```

`outer_edges` = `(top, bottom, left, right)`.

### 3.2 SampleConfig addition

```python
@dataclass(frozen=True)
class SampleConfig:
    # ... existing fields ...
    outer_edges: tuple[bool, bool, bool, bool] = (True, True, True, True)
```

`build_configs` generates edges per sample using the same local `random.Random(seed)` already used for other config fields. The tuple is generated in the main process and passed into `SampleConfig` before worker dispatch — never inside workers.

```python
outer_edges = tuple(rng.random() >= 0.05 for _ in range(4))
```

### 3.3 Border drawing logic

Replace the single `draw.rectangle()` outer border with 4 `draw.line()` calls, each gated by its `outer_edges` flag.

**Critical**: inner borders' line endpoints must **inset by `bw`** so they never touch the outer rectangle edge. This prevents an inner line from visually "filling in" a missing outer edge.

Current inner border code draws `num_cols - 1` vertical lines and `num_rows - 1` horizontal lines (the outermost top/bottom/left/right lines are not part of the inner loop because it iterates over `col_widths[:-1]` and `row_heights[:-1]`). After the change, each inner vertical line runs from `table_y + bw` to `table_y + total_table_h - bw`, and each inner horizontal line runs from `table_x + bw` to `table_x + total_table_w - bw`.

### 3.4 Nested table outer frame

In `_generate_sample`, the nested outer frame (1px rectangle) is always drawn completely — no edge dropout applied there. Nested calls to `_render_inner_table` for the inner table use the default `outer_edges=(True, True, True, True)`; the parent-generated `outer_edges` must never be passed down.

## 4. Files Changed

- `prepare_traindata/generate_table_layout.py` (only)
