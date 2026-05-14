"""Generate dense vertically-stacked synthetic training data for PP-DocLayoutV3.

Structures are arranged tightly in 1 or 2 columns (randomly chosen) rather than
scattered randomly.  This better mimics real chemistry documents where multiple
structures appear in compact blocks.

Output is written directly in PaddleX-compatible COCO format with:
* category_id = 14 (image class)
* segmentation masks (rectangle polygons)
* read_order (top-to-bottom, left-to-right)
"""

from __future__ import annotations

import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import click
import orjson
from rdkit import Chem, RDLogger

from prepare_traindata import watermark_utils
from prepare_traindata.categories import CAT_ID_IMAGE, CATEGORIES
from prepare_traindata.cli import (
    max_structures,
    min_structures,
    num_samples,
    output_dir,
    seed,
    split,
    structure_height_range,
    structure_width_range,
    watermark,
    workers,
)

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SMILES_PATH: Path = Path("smiles.txt")

CANVAS_MARGIN: int = 15
COL_GAP: int = 10        # horizontal gap between columns
ROW_GAP: int = 5         # vertical gap between structures in a column


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
class SampleConfig:
    sample_idx: int
    seed: int
    smiles: list[str]
    sizes: list[tuple[int, int]]
    num_cols: int
    output_dir: Path
    use_watermark: bool


def build_configs(
    smiles_list: list[str],
    num_samples: int,
    min_structures: int,
    max_structures: int,
    structure_width_range: tuple[int, int],
    structure_height_range: tuple[int, int],
    output_dir: Path,
    use_watermark: bool,
    seed: int | None = None,
) -> list[SampleConfig]:
    rng = random.Random(seed)
    configs: list[SampleConfig] = []
    for idx in range(num_samples):
        n = rng.randint(min_structures, max_structures)
        chosen = rng.sample(smiles_list, n)
        sizes = [
            (
                rng.randint(*structure_width_range),
                rng.randint(*structure_height_range),
            )
            for _ in range(n)
        ]
        num_cols = rng.choice([1, 2])
        sample_seed = rng.randint(0, 2**31 - 1)
        configs.append(
            SampleConfig(
                sample_idx=idx,
                seed=sample_seed,
                smiles=chosen,
                sizes=sizes,
                num_cols=num_cols,
                output_dir=output_dir,
                use_watermark=use_watermark,
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Rendering helpers (executed inside worker processes)
# ---------------------------------------------------------------------------

def _init_worker() -> None:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _render_one(
    smiles: str,
    size: tuple[int, int],
    rng: random.Random,
) -> Any | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    from prepare_traindata.rdkit_chem import render_mol_random
    try:
        img = render_mol_random(mol, size, rng)
    except Exception:
        return None
    return img


class Placement(NamedTuple):
    img: object  # PIL Image
    x: int
    y: int
    row_idx: int
    col_idx: int


class SampleResult(NamedTuple):
    img_id: int
    filename: str
    width: int
    height: int
    annotations: list[dict]


def _generate_sample(cfg: SampleConfig) -> SampleResult | None:
    from PIL import Image
    rng = random.Random(cfg.seed)
    images_dir = cfg.output_dir / "images"

    images: list[object] = []
    for i, (s, size) in enumerate(zip(cfg.smiles, cfg.sizes)):
        struct_rng = random.Random(cfg.seed + i + (hash(s) & 0xFFFFFFFF))
        img = _render_one(s, size, struct_rng)
        if img is not None:
            images.append(img)
        # Limit to max_structures based on config length, but we already
        # sampled exactly the right number; this is just a safety valve.
        if len(images) >= len(cfg.smiles):
            break

    if len(images) < 2:
        return None

    num_cols = cfg.num_cols
    # Distribute structures into columns round-robin
    cols: list[list[object]] = [[] for _ in range(num_cols)]
    for i, img in enumerate(images):
        cols[i % num_cols].append(img)

    # Calculate column widths and total height per column
    col_widths = [max(img.size[0] for img in col) if col else 0 for col in cols]
    col_heights = [
        sum(img.size[1] for img in col) + ROW_GAP * max(0, len(col) - 1)
        for col in cols
    ]

    canvas_w = sum(col_widths) + COL_GAP * (num_cols - 1) + CANVAS_MARGIN * 2
    canvas_h = max(col_heights) + CANVAS_MARGIN * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    if cfg.use_watermark:
        canvas = watermark_utils.apply_random_watermark(canvas, rng)

    placements: list[Placement] = []

    x_offset = CANVAS_MARGIN
    for col_idx, col in enumerate(cols):
        y_offset = CANVAS_MARGIN
        for row_idx, img in enumerate(col):
            # Center each image horizontally within its column
            img_w, img_h = img.size
            x = x_offset + (col_widths[col_idx] - img_w) // 2
            y = y_offset
            canvas.paste(img, (x, y))
            placements.append(Placement(img, x, y, row_idx, col_idx))
            y_offset += img_h + ROW_GAP
        x_offset += col_widths[col_idx] + COL_GAP

    if not placements:
        return None

    filename = f"dense_{cfg.sample_idx:06d}.png"
    out_path = images_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    # Build annotations with bbox, segmentation, read_order
    # Sort placements top-to-bottom, left-to-right for read_order
    sorted_placements = sorted(placements, key=lambda p: (p.col_idx, p.row_idx))

    annotations = []
    for order, pl in enumerate(sorted_placements):
        w, h = pl.img.size
        x, y = pl.x, pl.y
        bbox = [x, y, w, h]
        segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        annotations.append(
            {
                "id": cfg.sample_idx * 1_000 + order,
                "image_id": cfg.sample_idx,
                "category_id": CAT_ID_IMAGE,
                "bbox": bbox,
                "area": w * h,
                "iscrowd": 0,
                "segmentation": segmentation,
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
@output_dir(default="data/dense_layout")
@num_samples(default=2500)
@seed(default=42)
@workers(default=0)
@split(default=0.9)
@watermark(default=True)
@min_structures(default=2)
@max_structures(default=10)
@structure_width_range(default=(200, 350))
@structure_height_range(default=(80, 140))
def main(
    output_dir: str,
    num_samples: int,
    seed: int,
    workers: int,
    split: float,
    watermark: bool,
    min_structures: int,
    max_structures: int,
    structure_width_range: tuple[int, int],
    structure_height_range: tuple[int, int],
) -> int:
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    annotations_dir = output_path / "annotations"

    print("Loading SMILES …")
    smiles_list = load_valid_smiles(SMILES_PATH)
    total = len(smiles_list)
    print(f"  {total:,} valid SMILES loaded.")
    if total < max_structures:
        print(
            f"ERROR: Need at least {max_structures} valid SMILES, got {total}.",
            file=sys.stderr,
        )
        return 1

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    print("Building sample configurations …")
    configs = build_configs(
        smiles_list=smiles_list,
        num_samples=num_samples,
        min_structures=min_structures,
        max_structures=max_structures,
        structure_width_range=structure_width_range,
        structure_height_range=structure_height_range,
        output_dir=output_path,
        use_watermark=watermark,
        seed=seed,
    )

    if workers <= 0:
        workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Generating {num_samples:,} samples using {workers} workers …")

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
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
