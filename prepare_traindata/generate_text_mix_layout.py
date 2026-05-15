"""Generate synthetic text+image mixed-layout training data for PP-DocLayoutV3.

Simulates real chemistry documents (reaction pathways, impurity analysis,
quotation sheets) with both text blocks and chemical structure images.

Output COCO JSON includes:
* category_id = 14 (image) — chemical structures
* category_id = 22 (text)  — every text line gets its own bbox
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
from rdkit import RDLogger

from prepare_traindata import text_vocab, watermark_utils
from prepare_traindata.categories import CAT_ID_IMAGE, CAT_ID_TEXT, CATEGORIES
from prepare_traindata.cli import (
    max_cols,
    max_structures,
    min_cols,
    min_structures,
    num_samples,
    output_dir,
    seed,
    split,
    watermark,
    workers,
)
from prepare_traindata.rdkit_chem import render_smiles_random
from prepare_traindata.utils import load_valid_smiles

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SMILES_PATH: Path = Path("smiles.txt")
MARGIN: int = 20
LINE_GAP: int = 8          # gap between consecutive text lines
STRUCTURE_GAP: int = 15    # gap around structures
ARROW_LENGTH: int = 60     # horizontal arrow length
ARROW_HEAD_LEN: int = 12   # arrow head triangle size


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextLine:
    """A single text line with its bounding box."""
    text: str
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class StructurePlacement:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class SampleConfig:
    sample_idx: int
    seed: int
    mode: str  # "pathway", "vertical", "paragraph"
    smiles: tuple[str, ...]
    output_dir: Path
    use_watermark: bool
    min_cols: int
    max_cols: int


@dataclass(frozen=True)
class SampleResult:
    img_id: int
    filename: str
    width: int
    height: int
    annotations: list[dict[str, Any]]


def build_configs(
    smiles_list: list[str],
    num_samples: int,
    output_dir: Path,
    use_watermark: bool,
    min_structures: int,
    max_structures: int,
    min_cols: int,
    max_cols: int,
    seed: int | None = None,
) -> list[SampleConfig]:
    rng = random.Random(seed)
    configs: list[SampleConfig] = []
    for idx in range(num_samples):
        mode = rng.choice(["pathway", "vertical", "paragraph"])
        n_structures = rng.randint(min_structures, max_structures)
        chosen = tuple(rng.sample(smiles_list, n_structures))
        configs.append(
            SampleConfig(
                sample_idx=idx,
                seed=rng.randint(0, 2_147_483_647),
                mode=mode,
                smiles=chosen,
                output_dir=output_dir,
                use_watermark=use_watermark,
                min_cols=min_cols,
                max_cols=max_cols,
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------


def _init_worker() -> None:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


@lru_cache(maxsize=8)
def _load_font(size: int) -> Any | None:
    """Try to load a system CJK font; return None if none found."""
    from PIL import ImageFont

    candidates = [
        Path(r"C:/Windows/Fonts/msyh.ttc"),
        Path(r"C:/Windows/Fonts/simhei.ttf"),
        Path(r"C:/Windows/Fonts/simsun.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
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


def _draw_arrow(
    draw: Any,
    start: tuple[int, int],
    end: tuple[int, int],
    width: int = 2,
    color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw a vector arrow from start to end on a PIL ImageDraw."""
    import math

    draw.line([start, end], fill=color, width=width)
    # Arrow head
    x1, y1 = start
    x2, y2 = end
    angle = math.atan2(y2 - y1, x2 - x1)
    head_len = ARROW_HEAD_LEN
    left_angle = angle + math.radians(150)
    right_angle = angle - math.radians(150)
    p1 = (x2 + head_len * math.cos(left_angle), y2 + head_len * math.sin(left_angle))
    p2 = (x2 + head_len * math.cos(right_angle), y2 + head_len * math.sin(right_angle))
    draw.polygon([end, p1, p2], fill=color)


def _measure_text(draw: Any, text: str, font: Any | None) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_text_line(
    draw: Any,
    text: str,
    x: int,
    y: int,
    font: Any | None,
    color: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    """Draw a single text line and return (actual_x, actual_y, width, height).

    The returned coordinates reflect the real ink bounding-box returned by
    ``textbbox``, which may be offset from the nominal (x, y) anchor point
    because of font ascent/descent metrics.
    """
    if font is not None:
        draw.text((x, y), text, fill=color, font=font)
    else:
        draw.text((x, y), text, fill=color)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return x + bbox[0], y + bbox[1], w, h


# ---------------------------------------------------------------------------
# Layout generators
# ---------------------------------------------------------------------------


def _layout_pathway(
    canvas: Any,
    draw: Any,
    rng: random.Random,
    structures: list[Any],
    font: Any | None,
    font_small: Any | None,
) -> tuple[list[StructurePlacement], list[TextLine]]:
    """Mode A: title + horizontal structures with arrows + bottom notes."""
    text_lines: list[TextLine] = []
    placements: list[StructurePlacement] = []
    w, h = canvas.size
    text_color = (30, 30, 30)

    y_cursor = MARGIN

    # Title (1-2 lines)
    title = rng.choice(text_vocab.TITLE_SAMPLES)
    tw, th = _measure_text(draw, title, font)
    tx = (w - tw) // 2
    tx, ty, tw, th = _draw_text_line(draw, title, tx, y_cursor, font, text_color)
    text_lines.append(TextLine(title, tx, ty, tw, th))
    y_cursor += th + LINE_GAP

    # Optionally second title line
    if rng.random() < 0.3:
        subtitle = rng.choice(text_vocab.TITLE_SAMPLES)
        tw2, th2 = _measure_text(draw, subtitle, font)
        tx2 = (w - tw2) // 2
        tx2, ty2, tw2, th2 = _draw_text_line(
            draw, subtitle, tx2, y_cursor, font, text_color
        )
        text_lines.append(TextLine(subtitle, tx2, ty2, tw2, th2))
        y_cursor += th2 + LINE_GAP

    y_cursor += STRUCTURE_GAP

    # Structures in a row with arrows
    struct_h = 160
    struct_w = 220
    n = len(structures)
    total_struct_w = n * struct_w + (n - 1) * ARROW_LENGTH
    start_x = (w - total_struct_w) // 2

    # Track the actual bottom of the structure+arrow region to avoid overlap
    max_y_used = y_cursor + struct_h

    for i, img in enumerate(structures):
        # Resize to fit
        if img.size[0] > struct_w or img.size[1] > struct_h:
            scale = min(struct_w / img.size[0], struct_h / img.size[1])
            new_w = int(img.size[0] * scale)
            new_h = int(img.size[1] * scale)
            img = img.resize((new_w, new_h))
        iw, ih = img.size
        x = start_x + i * (struct_w + ARROW_LENGTH) + (struct_w - iw) // 2
        y = y_cursor + (struct_h - ih) // 2
        canvas.paste(img, (x, y), img)
        placements.append(StructurePlacement(x, y, iw, ih))

        # Arrow to next
        if i < n - 1:
            arrow_x1 = start_x + i * (struct_w + ARROW_LENGTH) + struct_w
            arrow_x2 = arrow_x1 + ARROW_LENGTH
            arrow_y = y_cursor + struct_h // 2
            _draw_arrow(draw, (arrow_x1, arrow_y), (arrow_x2, arrow_y), width=2)

            # Reagent label above/below arrow (only if it fits within arrow gap)
            if rng.random() < 0.7:
                reagent = rng.choice(text_vocab.REAGENT_SAMPLES)
                rw, rh = _measure_text(draw, reagent, font_small)
                if rw <= ARROW_LENGTH - 4:
                    label_x = arrow_x1 + (ARROW_LENGTH - rw) // 2
                    # Prefer above; if it would intrude into the title region,
                    # force below.
                    label_y_top = arrow_y - rh - 4
                    label_y_bottom = arrow_y + 8
                    if label_y_top >= y_cursor:
                        label_y = label_y_top
                    else:
                        label_y = label_y_bottom
                    lx, ly, lw, lh = _draw_text_line(
                        draw, reagent, label_x, label_y, font_small, text_color
                    )
                    text_lines.append(TextLine(reagent, lx, ly, lw, lh))
                    max_y_used = max(max_y_used, label_y + rh)

    y_cursor = max_y_used + STRUCTURE_GAP

    # Bottom notes (1-3 lines)
    n_notes = rng.randint(1, 3)
    for _ in range(n_notes):
        note = rng.choice(text_vocab.NOTE_SAMPLES)
        nw, nh = _measure_text(draw, note, font)
        nx = (w - nw) // 2
        nx, ny, nw, nh = _draw_text_line(
            draw, note, nx, y_cursor, font, text_color
        )
        text_lines.append(TextLine(note, nx, ny, nw, nh))
        y_cursor += nh + LINE_GAP

    return placements, text_lines


def _layout_vertical(
    canvas: Any,
    draw: Any,
    rng: random.Random,
    structures: list[Any],
    font: Any | None,
    font_small: Any | None,
    min_cols: int = 1,
    max_cols: int = 3,
) -> tuple[list[StructurePlacement], list[TextLine]]:
    """Mode B: title + 1-3 columns (structure + meta lines) + optional notes."""
    text_lines: list[TextLine] = []
    placements: list[StructurePlacement] = []
    w, h = canvas.size
    text_color = (30, 30, 30)

    y_cursor = MARGIN

    # Title
    title = rng.choice(text_vocab.TITLE_SAMPLES)
    tw, th = _measure_text(draw, title, font)
    tx = (w - tw) // 2
    tx, ty, tw, th = _draw_text_line(
        draw, title, tx, y_cursor, font, text_color
    )
    text_lines.append(TextLine(title, tx, ty, tw, th))
    y_cursor += th + LINE_GAP * 2

    # Columns
    n_cols = rng.randint(min_cols, max_cols)
    col_w = (w - MARGIN * 2) // n_cols
    struct_h = 140

    # Track the actual bottom of the tallest column
    max_col_bottom = y_cursor + struct_h

    for col_idx in range(n_cols):
        if col_idx >= len(structures):
            break
        img = structures[col_idx]
        cx = MARGIN + col_idx * col_w + col_w // 2

        # Structure centered in column
        if img.size[0] > col_w - 10 or img.size[1] > struct_h:
            scale = min((col_w - 10) / img.size[0], struct_h / img.size[1])
            new_w = int(img.size[0] * scale)
            new_h = int(img.size[1] * scale)
            img = img.resize((new_w, new_h))
        iw, ih = img.size
        x = cx - iw // 2
        y = y_cursor
        canvas.paste(img, (x, y), img)
        placements.append(StructurePlacement(x, y, iw, ih))

        # Meta lines below structure
        col_y = y_cursor + struct_h + LINE_GAP
        n_meta = rng.randint(2, 4)
        for _ in range(n_meta):
            meta = rng.choice(text_vocab.META_SAMPLES)
            mw, mh = _measure_text(draw, meta, font_small)
            mx = cx - mw // 2
            mx, my, mw, mh = _draw_text_line(
                draw, meta, mx, col_y, font_small, text_color
            )
            text_lines.append(TextLine(meta, mx, my, mw, mh))
            col_y += mh + LINE_GAP

        max_col_bottom = max(max_col_bottom, col_y)

    y_cursor = max_col_bottom + STRUCTURE_GAP

    # Bottom notes
    if rng.random() < 0.5:
        note = rng.choice(text_vocab.NOTE_SAMPLES)
        nw, nh = _measure_text(draw, note, font)
        nx = (w - nw) // 2
        nx, ny, nw, nh = _draw_text_line(
            draw, note, nx, y_cursor, font, text_color
        )
        text_lines.append(TextLine(note, nx, ny, nw, nh))
        y_cursor += nh + LINE_GAP

    return placements, text_lines


def _layout_paragraph(
    canvas: Any,
    draw: Any,
    rng: random.Random,
    structures: list[Any],
    font: Any | None,
    font_small: Any | None,
) -> tuple[list[StructurePlacement], list[TextLine]]:
    """Mode C: top paragraph + structures + bottom paragraph."""
    text_lines: list[TextLine] = []
    placements: list[StructurePlacement] = []
    w, h = canvas.size
    text_color = (30, 30, 30)

    y_cursor = MARGIN

    # Top paragraph (2-4 lines)
    n_top = rng.randint(2, 4)
    for _ in range(n_top):
        text = rng.choice(text_vocab.GENERIC_TEXT_LINES)
        tw, th = _measure_text(draw, text, font)
        tx = MARGIN + rng.randint(0, max(0, w - MARGIN * 2 - tw))
        tx, ty, tw, th = _draw_text_line(
            draw, text, tx, y_cursor, font, text_color
        )
        text_lines.append(TextLine(text, tx, ty, tw, th))
        y_cursor += th + LINE_GAP

    y_cursor += STRUCTURE_GAP

    # Structures: single row or 2x2 grid
    struct_h = 140
    struct_w = 200
    if len(structures) <= 3:
        # Single row
        total_w = len(structures) * struct_w + (len(structures) - 1) * STRUCTURE_GAP
        start_x = (w - total_w) // 2
        for i, img in enumerate(structures):
            if img.size[0] > struct_w or img.size[1] > struct_h:
                scale = min(struct_w / img.size[0], struct_h / img.size[1])
                img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
            iw, ih = img.size
            x = start_x + i * (struct_w + STRUCTURE_GAP) + (struct_w - iw) // 2
            y = y_cursor + (struct_h - ih) // 2
            canvas.paste(img, (x, y), img)
            placements.append(StructurePlacement(x, y, iw, ih))
        y_cursor += struct_h + STRUCTURE_GAP
    else:
        # 2x2 grid
        cols = 2
        rows = 2
        cell_w = (w - MARGIN * 2) // cols
        for idx, img in enumerate(structures[:4]):
            r, c = divmod(idx, cols)
            if img.size[0] > cell_w - 10 or img.size[1] > struct_h:
                scale = min((cell_w - 10) / img.size[0], struct_h / img.size[1])
                img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
            iw, ih = img.size
            cx = MARGIN + c * cell_w + cell_w // 2
            cy = y_cursor + r * (struct_h + STRUCTURE_GAP) + struct_h // 2
            x = cx - iw // 2
            y = cy - ih // 2
            canvas.paste(img, (x, y), img)
            placements.append(StructurePlacement(x, y, iw, ih))
        y_cursor += rows * (struct_h + STRUCTURE_GAP)

    # Bottom paragraph (2-4 lines)
    y_cursor += STRUCTURE_GAP
    n_bot = rng.randint(2, 4)
    for _ in range(n_bot):
        text = rng.choice(text_vocab.GENERIC_TEXT_LINES)
        tw, th = _measure_text(draw, text, font)
        tx = MARGIN + rng.randint(0, max(0, w - MARGIN * 2 - tw))
        tx, ty, tw, th = _draw_text_line(
            draw, text, tx, y_cursor, font, text_color
        )
        text_lines.append(TextLine(text, tx, ty, tw, th))
        y_cursor += th + LINE_GAP

    return placements, text_lines


def _max_content_bottom(
    placements: list[StructurePlacement],
    text_lines: list[TextLine],
) -> int:
    """Return the largest y + h among all drawn elements."""
    bottoms: list[int] = []
    for pl in placements:
        bottoms.append(pl.y + pl.h)
    for tl in text_lines:
        bottoms.append(tl.y + tl.h)
    return max(bottoms) if bottoms else 0


# ---------------------------------------------------------------------------
# Sample generator
# ---------------------------------------------------------------------------


def _generate_sample(cfg: SampleConfig) -> SampleResult | None:
    from PIL import Image, ImageDraw

    rng = random.Random(cfg.seed)

    n_smiles = len(cfg.smiles)

    # Canvas size varies by mode
    if cfg.mode == "pathway":
        canvas_w = max(
            600,
            n_smiles * 220 + max(0, n_smiles - 1) * ARROW_LENGTH + MARGIN * 2,
        )
        canvas_h = 400
    elif cfg.mode == "vertical":
        canvas_w = max(400, cfg.max_cols * 220 + MARGIN * 2)
        canvas_h = 500
    else:
        # paragraph: single row (<=3) or 2x2 grid (4)
        canvas_w = max(
            400,
            min(n_smiles, 3) * 220
            + max(0, min(n_smiles, 3) - 1) * STRUCTURE_GAP
            + MARGIN * 2,
        )
        canvas_h = 500

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    if cfg.use_watermark:
        canvas = watermark_utils.apply_random_watermark(canvas, rng)

    draw = ImageDraw.Draw(canvas)

    # Font sizes
    font_size = max(12, min(canvas_h // 25, 16))
    font_small_size = max(10, font_size - 2)
    font = _load_font(font_size)
    font_small = _load_font(font_small_size)

    # Render structures
    structures: list[Any] = []
    for i, s in enumerate(cfg.smiles):
        struct_rng = random.Random(cfg.seed + i + (hash(s) & 0xFFFFFFFF))
        img = render_smiles_random(s, (220, 160), struct_rng)
        if img is not None:
            structures.append(img)

    if not structures:
        return None

    # Dispatch layout
    if cfg.mode == "pathway":
        placements, text_lines = _layout_pathway(
            canvas, draw, rng, structures, font, font_small
        )
    elif cfg.mode == "vertical":
        placements, text_lines = _layout_vertical(
            canvas,
            draw,
            rng,
            structures,
            font,
            font_small,
            cfg.min_cols,
            cfg.max_cols,
        )
    else:
        placements, text_lines = _layout_paragraph(
            canvas, draw, rng, structures, font, font_small
        )

    # Extend canvas if content overflows
    max_bottom = _max_content_bottom(placements, text_lines)
    if max_bottom > canvas_h - MARGIN:
        new_h = max_bottom + MARGIN
        new_canvas = Image.new("RGB", (canvas_w, new_h), (255, 255, 255))
        new_canvas.paste(canvas, (0, 0))
        canvas = new_canvas
        canvas_h = new_h

    # Save
    filename = f"textmix_{cfg.sample_idx:06d}.png"
    images_dir = cfg.output_dir / "images"
    out_path = images_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    # Build annotations sorted by y then x for read_order
    raw_anns: list[tuple[int, int, dict]] = []  # (y, x, ann)

    for pl in placements:
        raw_anns.append(
            (
                pl.y,
                pl.x,
                {
                    "id": cfg.sample_idx * 1_000 + len(raw_anns),
                    "image_id": cfg.sample_idx,
                    "category_id": CAT_ID_IMAGE,
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
                },
            )
        )

    for tl in text_lines:
        raw_anns.append(
            (
                tl.y,
                tl.x,
                {
                    "id": cfg.sample_idx * 1_000 + len(raw_anns),
                    "image_id": cfg.sample_idx,
                    "category_id": CAT_ID_TEXT,
                    "bbox": [tl.x, tl.y, tl.w, tl.h],
                    "area": tl.w * tl.h,
                    "iscrowd": 0,
                    "segmentation": [
                        [
                            tl.x,
                            tl.y,
                            tl.x + tl.w,
                            tl.y,
                            tl.x + tl.w,
                            tl.y + tl.h,
                            tl.x,
                            tl.y + tl.h,
                        ]
                    ],
                },
            )
        )

    raw_anns.sort(key=lambda t: (t[0], t[1]))
    annotations = []
    for order, (_, _, ann) in enumerate(raw_anns):
        ann["read_order"] = order
        annotations.append(ann)

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
@output_dir(default="data/text_mix_layout")
@num_samples(default=1000)
@seed(default=42)
@workers(default=8)
@split(default=0.8)
@watermark(default=True)
@min_structures(default=2)
@max_structures(default=5)
@min_cols(default=1)
@max_cols(default=3)
def main(
    output_dir: str,
    num_samples: int,
    seed: int,
    workers: int,
    split: float,
    watermark: bool,
    min_structures: int,
    max_structures: int,
    min_cols: int,
    max_cols: int,
) -> int:
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    annotations_dir = output_path / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    if min_structures > max_structures:
        print(
            "ERROR: --min-structures must be <= --max-structures.",
            file=sys.stderr,
        )
        return 1
    if min_cols > max_cols:
        print("ERROR: --min-cols must be <= --max-cols.", file=sys.stderr)
        return 1

    print("Loading SMILES …")
    smiles_list = load_valid_smiles(SMILES_PATH)
    total = len(smiles_list)
    print(f"  {total:,} valid SMILES loaded.")
    if total < max_structures:
        print(
            f"ERROR: Need at least {max_structures} valid SMILES.",
            file=sys.stderr,
        )
        return 1

    print("Building sample configurations …")
    configs = build_configs(
        smiles_list=smiles_list,
        num_samples=num_samples,
        output_dir=output_path,
        use_watermark=watermark,
        min_structures=min_structures,
        max_structures=max_structures,
        min_cols=min_cols,
        max_cols=max_cols,
        seed=seed,
    )

    if workers <= 0:
        workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Generating {num_samples:,} samples using {workers} workers …")

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
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

    # Train/val split
    rng_split = random.Random(seed)
    rng_split.shuffle(coco_images)

    if 0.0 < split < 1.0:
        n_train = int(len(coco_images) * split)
        train_images = coco_images[:n_train]
        val_images = coco_images[n_train:]

        train_ids = {img["id"] for img in train_images}
        val_ids = {img["id"] for img in val_images}

        train_annotations = [
            ann for ann in coco_annotations if ann["image_id"] in train_ids
        ]
        val_annotations = [
            ann for ann in coco_annotations if ann["image_id"] in val_ids
        ]

        train_coco = {
            "images": train_images,
            "annotations": train_annotations,
            "categories": CATEGORIES,
        }
        val_coco = {
            "images": val_images,
            "annotations": val_annotations,
            "categories": CATEGORIES,
        }

        train_path = annotations_dir / "instance_train.json"
        val_path = annotations_dir / "instance_val.json"
        train_path.write_bytes(orjson.dumps(train_coco, option=orjson.OPT_INDENT_2))
        val_path.write_bytes(orjson.dumps(val_coco, option=orjson.OPT_INDENT_2))

        print(f"\nDone.")
        print(f"  Images:      {images_dir}")
        print(f"  Annotations: {annotations_dir}")
        print(f"  Train images: {len(train_images):,}")
        print(f"  Val images:   {len(val_images):,}")
        print(f"  Train boxes:  {len(train_annotations):,}")
        print(f"  Val boxes:    {len(val_annotations):,}")
    else:
        coco = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": CATEGORIES,
        }
        ann_path = annotations_dir / "instance_train.json"
        ann_path.write_bytes(orjson.dumps(coco, option=orjson.OPT_INDENT_2))

        print(f"\nDone.")
        print(f"  Images:      {images_dir}")
        print(f"  Annotations: {ann_path}")
        print(f"  Total images written: {len(coco_images):,}")
        print(f"  Total boxes written:  {len(coco_annotations):,}")

    return 0


if __name__ == "__main__":
    main()
