"""Generate synthetic table-layout training data for PP-DocLayoutV3.

Each image contains a table with random rows/columns and cell sizes.
Some cells contain chemical structure images (RDKit), others contain random text.
Background has a random watermark applied.

Output COCO JSON includes:
* category_id = 21 (table) — one per image
* category_id = 14 (image) — one per chemical structure placed in a cell
"""

from __future__ import annotations

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SMILES_PATH: Path = Path("smiles.txt")
OUTPUT_DIR: Path = Path("data/table_layout")
IMAGES_DIR: Path = OUTPUT_DIR / "images"
ANNOTATIONS_DIR: Path = OUTPUT_DIR / "annotations"

NUM_SAMPLES: int = 2_500
MARGIN: int = 20
RANDOM_SEED: int | None = 42

WORKERS: int = max(1, (os.cpu_count() or 4) - 1)

# Probability a cell contains a chemical structure (otherwise text)
STRUCTURE_PROB: float = 0.4

# Padding inside a cell when placing content
CELL_PADDING: int = 10

# Minimum margin around a resized structure inside a cell
MIN_STRUCTURE_MARGIN: int = 5

# Category IDs
CAT_TABLE: int = 21
CAT_IMAGE: int = 14

# All 25 PP-DocLayoutV3 categories (must be present in output JSON)
CATEGORIES: list[dict[str, Any]] = [
    {"id": 0, "name": "abstract"},
    {"id": 1, "name": "algorithm"},
    {"id": 2, "name": "aside_text"},
    {"id": 3, "name": "chart"},
    {"id": 4, "name": "content"},
    {"id": 5, "name": "display_formula"},
    {"id": 6, "name": "doc_title"},
    {"id": 7, "name": "figure_title"},
    {"id": 8, "name": "footer"},
    {"id": 9, "name": "footer_image"},
    {"id": 10, "name": "footnote"},
    {"id": 11, "name": "formula_number"},
    {"id": 12, "name": "header"},
    {"id": 13, "name": "header_image"},
    {"id": 14, "name": "image"},
    {"id": 15, "name": "inline_formula"},
    {"id": 16, "name": "number"},
    {"id": 17, "name": "paragraph_title"},
    {"id": 18, "name": "reference"},
    {"id": 19, "name": "reference_content"},
    {"id": 20, "name": "seal"},
    {"id": 21, "name": "table"},
    {"id": 22, "name": "text"},
    {"id": 23, "name": "table_caption"},
    {"id": 24, "name": "vision_footnote"},
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_valid_smiles(path: Path) -> list[str]:
    valid: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            mol = Chem.MolFromSmiles(line)
            if mol is not None:
                valid.append(line)
    return valid


@dataclass(frozen=True)
class CellConfig:
    use_structure: bool
    smiles: str | None


@dataclass(frozen=True)
class SampleConfig:
    sample_idx: int
    seed: int
    num_cols: int
    num_rows: int
    col_widths: tuple[int, ...]
    row_heights: tuple[int, ...]
    border_width: int
    cells: tuple[CellConfig, ...]


def build_configs(
    smiles_list: list[str],
    num_samples: int,
    seed: int | None = None,
) -> list[SampleConfig]:
    rng = random.Random(seed)
    configs: list[SampleConfig] = []
    for idx in range(num_samples):
        num_cols = rng.randint(1, 5)
        num_rows = rng.randint(1, 15)
        col_widths = tuple(rng.randint(100, 300) for _ in range(num_cols))
        row_heights = tuple(rng.randint(80, 200) for _ in range(num_rows))
        border_width = rng.randint(2, 4)

        total_cells = num_cols * num_rows
        cells: list[CellConfig] = []
        for _ in range(total_cells):
            use_structure = rng.random() < STRUCTURE_PROB
            if use_structure:
                smiles = rng.choice(smiles_list)
                cells.append(CellConfig(use_structure=True, smiles=smiles))
            else:
                cells.append(CellConfig(use_structure=False, smiles=None))

        configs.append(
            SampleConfig(
                sample_idx=idx,
                seed=rng.randint(0, 2_147_483_647),
                num_cols=num_cols,
                num_rows=num_rows,
                col_widths=col_widths,
                row_heights=row_heights,
                border_width=border_width,
                cells=tuple(cells),
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------

_WORKER_SMILES_POOL: list[str] | None = None


def _init_worker(smiles_pool: list[str] | None = None) -> None:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    global _WORKER_SMILES_POOL
    if smiles_pool is not None:
        _WORKER_SMILES_POOL = smiles_pool


def _render_structure(smiles: str, target_size: tuple[int, int]) -> Any | None:
    """Render a SMILES structure to a PIL Image, or None on failure."""
    from PIL import Image
    from prepare_traindata.image import trim
    from prepare_traindata.rdkit_chem import d_opts
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        img = Draw.MolToImage(mol, size=target_size, options=d_opts, fitImage=True)
        img = trim(img)
    except Exception:
        return None

    if img is None:
        return None

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _resize_to_fit(
    img: Any,
    max_w: int,
    max_h: int,
) -> Any:
    """Resize image proportionally to fit within max_w x max_h."""
    from PIL import Image

    w, h = img.size
    if w <= max_w and h <= max_h:
        return img
    scale = min(max_w / w, max_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


@dataclass(frozen=True)
class StructurePlacement:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class SampleResult:
    img_id: int
    filename: str
    width: int
    height: int
    annotations: list[dict[str, Any]]


def _generate_sample(cfg: SampleConfig) -> SampleResult | None:
    from PIL import Image, ImageDraw
    from prepare_traindata import text_vocab
    from prepare_traindata import watermark_utils

    rng = random.Random(cfg.seed)

    total_table_w = sum(cfg.col_widths)
    total_table_h = sum(cfg.row_heights)
    canvas_w = MARGIN * 2 + total_table_w
    canvas_h = MARGIN * 2 + total_table_h

    # White canvas, apply watermark
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas = watermark_utils.apply_random_watermark(canvas, rng)

    draw = ImageDraw.Draw(canvas)

    # Draw table cell backgrounds as white rectangles (so watermark doesn't
    # interfere with readability), then draw borders on top.
    table_x = MARGIN
    table_y = MARGIN

    # Cell backgrounds
    y_cursor = table_y
    for row_h in cfg.row_heights:
        x_cursor = table_x
        for col_w in cfg.col_widths:
            draw.rectangle(
                [x_cursor, y_cursor, x_cursor + col_w, y_cursor + row_h],
                fill=(255, 255, 255),
            )
            x_cursor += col_w
        y_cursor += row_h

    # Place contents and collect structure placements
    placements: list[StructurePlacement] = []
    cell_iter = iter(cfg.cells)
    y_cursor = table_y
    for row_idx, row_h in enumerate(cfg.row_heights):
        x_cursor = table_x
        for col_idx, col_w in enumerate(cfg.col_widths):
            cell = next(cell_iter)
            content_x = x_cursor + CELL_PADDING
            content_y = y_cursor + CELL_PADDING
            content_w = col_w - CELL_PADDING * 2
            content_h = row_h - CELL_PADDING * 2

            placed_structure = False
            if cell.use_structure and cell.smiles is not None:
                target_w = max(content_w, 1)
                target_h = max(content_h, 1)
                img = _render_structure(cell.smiles, (target_w, target_h))
                if img is not None:
                    # Ensure it fits with at least MIN_STRUCTURE_MARGIN on all sides
                    max_w = col_w - MIN_STRUCTURE_MARGIN * 2
                    max_h = row_h - MIN_STRUCTURE_MARGIN * 2
                    if img.size[0] > max_w or img.size[1] > max_h:
                        img = _resize_to_fit(img, max_w, max_h)
                    iw, ih = img.size
                    # Center inside cell
                    paste_x = x_cursor + (col_w - iw) // 2
                    paste_y = y_cursor + (row_h - ih) // 2
                    canvas.paste(img, (paste_x, paste_y))
                    placements.append(
                        StructurePlacement(paste_x, paste_y, iw, ih)
                    )
                    placed_structure = True

            if not placed_structure:
                # Draw text centered in cell
                text = text_vocab.get_random_text(rng)
                # Use default font; measure via textbbox
                text_color = rng.choice([
                    (30, 30, 30),
                    (0, 0, 0),
                    (50, 50, 50),
                ])
                bbox = draw.textbbox((0, 0), text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = x_cursor + (col_w - tw) // 2
                ty = y_cursor + (row_h - th) // 2
                draw.text((tx, ty), text, fill=text_color)

            x_cursor += col_w
        y_cursor += row_h

    # Draw borders (outer + inner)
    bw = cfg.border_width
    # Outer border
    draw.rectangle(
        [table_x, table_y, table_x + total_table_w, table_y + total_table_h],
        outline=(0, 0, 0),
        width=bw,
    )
    # Vertical inner lines
    x_cursor = table_x
    for col_w in cfg.col_widths[:-1]:
        x_cursor += col_w
        draw.line(
            [(x_cursor, table_y), (x_cursor, table_y + total_table_h)],
            fill=(0, 0, 0),
            width=bw,
        )
    # Horizontal inner lines
    y_cursor = table_y
    for row_h in cfg.row_heights[:-1]:
        y_cursor += row_h
        draw.line(
            [(table_x, y_cursor), (table_x + total_table_w, y_cursor)],
            fill=(0, 0, 0),
            width=bw,
        )

    # Save image
    filename = f"table_{cfg.sample_idx:06d}.png"
    out_path = IMAGES_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    # Build annotations
    annotations: list[dict[str, Any]] = []

    # Table annotation (read_order = 0)
    annotations.append(
        {
            "id": cfg.sample_idx * 1_000,
            "image_id": cfg.sample_idx,
            "category_id": CAT_TABLE,
            "bbox": [table_x, table_y, total_table_w, total_table_h],
            "area": total_table_w * total_table_h,
            "iscrowd": 0,
            "segmentation": [
                [
                    table_x,
                    table_y,
                    table_x + total_table_w,
                    table_y,
                    table_x + total_table_w,
                    table_y + total_table_h,
                    table_x,
                    table_y + total_table_h,
                ]
            ],
            "read_order": 0,
        }
    )

    # Structure annotations sorted top-to-bottom, left-to-right
    sorted_placements = sorted(placements, key=lambda p: (p.y, p.x))
    for order, pl in enumerate(sorted_placements, start=1):
        annotations.append(
            {
                "id": cfg.sample_idx * 1_000 + order,
                "image_id": cfg.sample_idx,
                "category_id": CAT_IMAGE,
                "bbox": [pl.x, pl.y, pl.w, pl.h],
                "area": pl.w * pl.h,
                "iscrowd": 0,
                "segmentation": [
                    [
                        pl.x,
                        pl.y,
                        pl.x + pl.w,
                        pl.y,
                        pl.x + pl.w,
                        pl.y + pl.h,
                        pl.x,
                        pl.y + pl.h,
                    ]
                ],
                "read_order": order,
            }
        )

    return SampleResult(
        img_id=cfg.sample_idx,
        filename=filename,
        width=canvas_w,
        height=canvas_h,
        annotations=annotations,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Loading SMILES …")
    smiles_list = load_valid_smiles(SMILES_PATH)
    total = len(smiles_list)
    print(f"  {total:,} valid SMILES loaded.")
    if total < 1:
        print("ERROR: Need at least 1 valid SMILES.", file=sys.stderr)
        return 1

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Building sample configurations …")
    configs = build_configs(smiles_list, NUM_SAMPLES, seed=RANDOM_SEED)

    print(f"Generating {NUM_SAMPLES:,} samples using {WORKERS} workers …")

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    completed = 0

    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=_init_worker,
        initargs=(smiles_list,),
    ) as pool:
        futures = {pool.submit(_generate_sample, cfg): cfg for cfg in configs}
        for future in as_completed(futures):
            result: SampleResult | None = future.result()
            if result is not None:
                coco_images.append(
                    {
                        "id": result.img_id,
                        "file_name": result.filename,
                        "width": result.width,
                        "height": result.height,
                    }
                )
                coco_annotations.extend(result.annotations)

            completed += 1
            if completed % 100 == 0 or completed == NUM_SAMPLES:
                print(f"  {completed:,} / {NUM_SAMPLES:,} done …")

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": CATEGORIES,
    }

    ann_path = ANNOTATIONS_DIR / "instance_train.json"
    with ann_path.open("w", encoding="utf-8") as fh:
        json.dump(coco, fh, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"  Images:      {IMAGES_DIR}")
    print(f"  Annotations: {ann_path}")
    print(f"  Total images written: {len(coco_images):,}")
    print(f"  Total boxes written:  {len(coco_annotations):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
