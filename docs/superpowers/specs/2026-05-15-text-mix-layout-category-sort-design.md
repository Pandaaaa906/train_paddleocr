# text_mix_layout Category & Sort Fix Design

## 1. Problem

`generate_text_mix_layout.py` currently labels all text as `text` (cat=22), and sorts annotations globally by `(y, x)`. This doesn't match PP-DocLayoutV3's class semantics.

## 2. Changes

### 2.1 Category Assignment

| Element | Current | Target | Condition |
|---------|---------|--------|-----------|
| Top centered title (`TITLE_SAMPLES`) | `text` (22) | `doc_title` (6) | Always in `pathway` / `vertical` |
| Top paragraph first line | `text` (22) | `paragraph_title` (17) | `paragraph` mode only |
| Remaining text lines | `text` (22) | `text` (22) | Unchanged |
| Structure images | `image` (14) | `image` (14) | Unchanged |

### 2.2 Read-order Sorting (`vertical` mode)

Current: all annotations sorted by `(y, x)` globally.

New: `vertical` mode sorts by `(col_idx, y, x)` — column-first, then top-to-bottom within each column. Other modes keep `(y, x)`.

## 3. Implementation (Approach A)

1. Add `category_id: int = CAT_ID_TEXT` and `col_idx: int = 0` to `TextLine`
2. Add `col_idx: int = 0` to `StructurePlacement`
3. Each layout function sets `category_id` at creation time:
   - `_layout_pathway`: title(s) → `CAT_ID_DOC_TITLE`, reagent/note → `CAT_ID_TEXT`
   - `_layout_vertical`: title → `CAT_ID_DOC_TITLE`, meta → `CAT_ID_TEXT`, structures get their column index
   - `_layout_paragraph`: first top line → `CAT_ID_PARAGRAPH_TITLE`, rest → `CAT_ID_TEXT`
4. `_generate_sample` uses `category_id` from dataclass when building annotations
5. Sort key in `_generate_sample`:
   - `vertical`: `(col_idx, y, x)`
   - others: `(y, x)`

## 4. Constants to Add

```python
CAT_ID_DOC_TITLE: int = 6
CAT_ID_PARAGRAPH_TITLE: int = 17
```

## 5. Files Changed

- `prepare_traindata/generate_text_mix_layout.py` (only)
