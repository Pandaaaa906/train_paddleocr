"""Generate synthetic table-layout training data for PP-DocLayoutV3.

Each image contains a table with random rows/columns and cell sizes.
Some cells contain chemical structure images (RDKit), others contain random text.
Background has a random watermark applied.

Output COCO JSON includes:
* category_id = 21 (table) — one per image
* category_id = 14 (image) — one per chemical structure placed in a cell
"""

from __future__ import annotations

import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import click
import orjson
from rdkit import Chem, RDLogger

from prepare_traindata.categories import CAT_ID_IMAGE, CAT_ID_TABLE, CATEGORIES
from prepare_traindata.cli import (
    cell_height_range,
    cell_width_range,
    max_cols,
    max_rows,
    min_cols,
    min_rows,
    num_samples,
    output_dir,
    seed,
    split,
    structure_prob,
    watermark,
    workers,
)

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SMILES_PATH: Path = Path("smiles.txt")
MARGIN: int = 20
CELL_PADDING: int = 10
MIN_STRUCTURE_MARGIN: int = 5

CAT_TABLE: int = CAT_ID_TABLE
CAT_IMAGE: int = CAT_ID_IMAGE


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
    output_dir: Path
    use_watermark: bool


def build_configs(
    smiles_list: list[str],
    num_samples: int,
    seed: int,
    structure_prob: float,
    min_cols: int,
    max_cols: int,
    min_rows: int,
    max_rows: int,
    cell_width_range: tuple[int, int],
    cell_height_range: tuple[int, int],
) -> list[SampleConfig]:
    rng = random.Random(seed)
    configs: list[SampleConfig] = []
    for idx in range(num_samples):
        num_cols = rng.randint(min_cols, max_cols)
        num_rows = rng.randint(min_rows, max_rows)
        col_widths = tuple(rng.randint(*cell_width_range) for _ in range(num_cols))
        row_heights = tuple(rng.randint(*cell_height_range) for _ in range(num_rows))
        border_width = rng.randint(2, 4)

        total_cells = num_cols * num_rows
        cells: list[CellConfig] = []
        for _ in range(total_cells):
            use_structure = rng.random() < structure_prob
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
                output_dir=Path("PLACEHOLDER"),
                use_watermark=True,
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


@lru_cache(maxsize=8)
def _load_font(size: int) -> Any | None:
    """Try to load a system CJK font; return None if none found."""
    from PIL import ImageFont

    candidates = [
        # Windows
        Path(r"C:/Windows/Fonts/msyh.ttc"),      # Microsoft YaHei
        Path(r"C:/Windows/Fonts/simhei.ttf"),    # SimHei
        Path(r"C:/Windows/Fonts/simsun.ttc"),    # SimSun
        Path(r"C:/Windows/Fonts/msgothic.ttc"),  # MS Gothic
        # Linux
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        # macOS
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    for cand in candidates:
        if cand.exists():
            try:
                return ImageFont.truetype(str(cand), size=size)
            except Exception:
                continue
    return None


def _break_word(
    draw: Any,
    word: str,
    font: Any | None,
    max_width: int,
) -> list[str]:
    """Break a single long word into lines that fit within max_width."""
    lines: list[str] = []
    current = ""
    for char in word:
        test = current + char
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def _wrap_text(
    draw: Any,
    text: str,
    font: Any | None,
    max_width: int,
) -> list[str]:
    """Wrap text into lines that fit within max_width."""
    if not text:
        return [""]

    # Try single line first
    bbox = draw.textbbox((0, 0), text, font=font)
    if bbox[2] - bbox[0] <= max_width:
        return [text]

    lines: list[str] = []
    current_line = ""
    words = text.split(" ")

    for word in words:
        test = word if not current_line else current_line + " " + word
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
                # Check if word itself is too long for a fresh line
                bbox_word = draw.textbbox((0, 0), word, font=font)
                if bbox_word[2] - bbox_word[0] > max_width:
                    lines.extend(_break_word(draw, word, font, max_width))
                    current_line = ""
            else:
                # Empty line but word still too long
                lines.extend(_break_word(draw, word, font, max_width))
                current_line = ""

    if current_line:
        lines.append(current_line)
    return lines


def _render_structure(smiles: str, target_size: tuple[int, int]) -> Any | None:
    """Render a SMILES structure to a PIL Image, or None on failure."""
    from PIL import Image
    from rdkit.Chem import Draw

    from prepare_traindata.image import trim
    from prepare_traindata.rdkit_chem import d_opts

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
    row: int
    col: int


@dataclass(frozen=True)
class SampleResult:
    img_id: int
    filename: str
    width: int
    height: int
    annotations: list[dict[str, Any]]


def _generate_sample(cfg: SampleConfig) -> SampleResult | None:
    from PIL import Image, ImageDraw

    from prepare_traindata import text_vocab, watermark_utils

    rng = random.Random(cfg.seed)

    total_table_w = sum(cfg.col_widths)
    total_table_h = sum(cfg.row_heights)
    canvas_w = MARGIN * 2 + total_table_w
    canvas_h = MARGIN * 2 + total_table_h

    # White canvas, apply watermark
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    if cfg.use_watermark:
        canvas = watermark_utils.apply_random_watermark(canvas, rng)

    draw = ImageDraw.Draw(canvas)

    # Draw table cell backgrounds as white rectangles only when there is no
    # watermark, so that the watermark remains visible inside cells.
    table_x = MARGIN
    table_y = MARGIN

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
                img = None
                smiles_to_try = cell.smiles
                for _ in range(3):
                    img = _render_structure(smiles_to_try, (target_w, target_h))
                    if img is not None:
                        break
                    if _WORKER_SMILES_POOL:
                        smiles_to_try = rng.choice(_WORKER_SMILES_POOL)
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
                        StructurePlacement(paste_x, paste_y, iw, ih, row_idx, col_idx)
                    )
                    placed_structure = True

            if not placed_structure:
                # Draw text centered in cell (with wrapping)
                text = text_vocab.get_random_text(rng)
                text_color = rng.choice([
                    (30, 30, 30),
                    (0, 0, 0),
                    (50, 50, 50),
                ])
                # Pick font size proportional to cell height (clamp 9–16 px)
                font_size = max(9, min(content_h // 3, 16))
                font = _load_font(font_size)
                wrapped = _wrap_text(draw, text, font, content_w)

                # Measure a single line to get line height
                if font is not None:
                    sample_bbox = draw.textbbox((0, 0), "国", font=font)
                else:
                    sample_bbox = draw.textbbox((0, 0), "A")
                line_h = sample_bbox[3] - sample_bbox[1]
                total_text_h = len(wrapped) * line_h
                # Clamp total height so it doesn't overflow the cell
                max_text_h = content_h
                if total_text_h > max_text_h and len(wrapped) > 1:
                    # Try to shrink font by one step and re-wrap
                    smaller_font = _load_font(max(9, font_size - 2))
                    wrapped = _wrap_text(draw, text, smaller_font, content_w)
                    if smaller_font is not None:
                        sample_bbox = draw.textbbox((0, 0), "国", font=smaller_font)
                    else:
                        sample_bbox = draw.textbbox((0, 0), "A")
                    line_h = sample_bbox[3] - sample_bbox[1]
                    total_text_h = len(wrapped) * line_h
                    font = smaller_font

                start_y = y_cursor + (row_h - total_text_h) // 2
                for line in wrapped:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    tw = bbox[2] - bbox[0]
                    tx = x_cursor + (col_w - tw) // 2
                    if font is not None:
                        draw.text((tx, start_y), line, fill=text_color, font=font)
                    else:
                        draw.text((tx, start_y), line, fill=text_color)
                    start_y += line_h

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
    images_dir = cfg.output_dir / "images"
    out_path = images_dir / filename
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

    # Structure annotations sorted by cell row/col (top-to-bottom, left-to-right)
    sorted_placements = sorted(placements, key=lambda p: (p.row, p.col))
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

@click.command()
@output_dir(default="data/table_layout")
@num_samples(default=2500)
@seed(default=42)
@workers(default=0)
@split(default=0.0)
@watermark(default=True)
@structure_prob(default=0.4)
@min_cols(default=1)
@max_cols(default=5)
@min_rows(default=1)
@max_rows(default=15)
@cell_width_range(default=(100, 300))
@cell_height_range(default=(80, 200))
def main(
    output_dir: str,
    num_samples: int,
    seed: int,
    workers: int,
    split: float,
    watermark: bool,
    structure_prob: float,
    min_cols: int,
    max_cols: int,
    min_rows: int,
    max_rows: int,
    cell_width_range: tuple[int, int],
    cell_height_range: tuple[int, int],
) -> int:
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    annotations_dir = output_path / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SMILES …")
    smiles_list = load_valid_smiles(SMILES_PATH)
    total = len(smiles_list)
    print(f"  {total:,} valid SMILES loaded.")
    if total < 1:
        print("ERROR: Need at least 1 valid SMILES.", file=sys.stderr)
        return 1

    print("Building sample configurations …")
    configs = build_configs(
        smiles_list=smiles_list,
        num_samples=num_samples,
        seed=seed,
        structure_prob=structure_prob,
        min_cols=min_cols,
        max_cols=max_cols,
        min_rows=min_rows,
        max_rows=max_rows,
        cell_width_range=cell_width_range,
        cell_height_range=cell_height_range,
    )

    # Patch output_dir and watermark into configs for worker safety
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
        )
        for c in configs
    ]

    if workers <= 0:
        workers = max(1, (os.cpu_count() or 4) - 1)

    print(f"Generating {num_samples:,} samples using {workers} workers …")

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    completed = 0

    with ProcessPoolExecutor(
        max_workers=workers,
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
            if completed % 100 == 0 or completed == num_samples:
                print(f"  {completed:,} / {num_samples:,} done …")

    if 0.0 < split < 1.0:
        n_train = int(len(coco_images) * split)
        train_images = coco_images[:n_train]
        val_images = coco_images[n_train:]
        train_ids = {img["id"] for img in train_images}
        val_ids = {img["id"] for img in val_images}
        train_anns = [ann for ann in coco_annotations if ann["image_id"] in train_ids]
        val_anns = [ann for ann in coco_annotations if ann["image_id"] in val_ids]

        train_path = annotations_dir / "instance_train.json"
        val_path = annotations_dir / "instance_val.json"
        train_path.write_bytes(
            orjson.dumps(
                {
                    "images": train_images,
                    "annotations": train_anns,
                    "categories": CATEGORIES,
                },
                option=orjson.OPT_INDENT_2,
            )
        )
        val_path.write_bytes(
            orjson.dumps(
                {
                    "images": val_images,
                    "annotations": val_anns,
                    "categories": CATEGORIES,
                },
                option=orjson.OPT_INDENT_2,
            )
        )
        print(f"\nDone.")
        print(f"  Images:      {images_dir}")
        print(f"  Annotations: {annotations_dir}")
        print(f"  Train images: {len(train_images):,}")
        print(f"  Val images:   {len(val_images):,}")
        print(f"  Train boxes:  {len(train_anns):,}")
        print(f"  Val boxes:    {len(val_anns):,}")
    else:
        ann_path = annotations_dir / "instance_train.json"
        ann_path.write_bytes(
            orjson.dumps(
                {
                    "images": coco_images,
                    "annotations": coco_annotations,
                    "categories": CATEGORIES,
                },
                option=orjson.OPT_INDENT_2,
            )
        )
        print(f"\nDone.")
        print(f"  Images:      {images_dir}")
        print(f"  Annotations: {ann_path}")
        print(f"  Total images written: {len(coco_images):,}")
        print(f"  Total boxes written:  {len(coco_annotations):,}")

    return 0


if __name__ == "__main__":
    main()
