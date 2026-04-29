"""Generate dense vertically-stacked synthetic training data for PP-DocLayoutV3.

Structures are arranged tightly in 1 or 2 columns (randomly chosen) rather than
scattered randomly.  This better mimics real chemistry documents where multiple
structures appear in compact blocks.

Output is written directly in PaddleX-compatible COCO format with:
* category_id = 0
* segmentation masks (rectangle polygons)
* read_order (top-to-bottom, left-to-right)
"""

from __future__ import annotations

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from prepare_traindata.categories import CATEGORIES
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SMILES_PATH: Path = Path("smiles.txt")
OUTPUT_DIR: Path = Path("data/dense_layout")
IMAGES_DIR: Path = OUTPUT_DIR / "images"
ANNOTATIONS_DIR: Path = OUTPUT_DIR / "annotations"

NUM_SAMPLES: int = 2500
MIN_STRUCTURES: int = 2
MAX_STRUCTURES: int = 10

STRUCTURE_WIDTH_RANGE: tuple[int, int] = (200, 350)
STRUCTURE_HEIGHT_RANGE: tuple[int, int] = (80, 140)

CANVAS_MARGIN: int = 15
COL_GAP: int = 10        # horizontal gap between columns
ROW_GAP: int = 5         # vertical gap between structures in a column
RANDOM_SEED: int | None = 42

CATEGORY_ID: int = 0


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


class SampleConfig(NamedTuple):
    sample_idx: int
    smiles: list[str]
    sizes: list[tuple[int, int]]
    num_cols: int


def build_configs(
    smiles_list: list[str],
    num_samples: int,
    seed: int | None = None,
) -> list[SampleConfig]:
    rng = random.Random(seed)
    configs: list[SampleConfig] = []
    for idx in range(num_samples):
        n = rng.randint(MIN_STRUCTURES, MAX_STRUCTURES)
        chosen = rng.sample(smiles_list, n)
        sizes = [
            (
                rng.randint(*STRUCTURE_WIDTH_RANGE),
                rng.randint(*STRUCTURE_HEIGHT_RANGE),
            )
            for _ in range(n)
        ]
        num_cols = rng.choice([1, 2])
        configs.append(SampleConfig(idx, chosen, sizes, num_cols))
    return configs


# ---------------------------------------------------------------------------
# Rendering helpers (executed inside worker processes)
# ---------------------------------------------------------------------------

def _init_worker():
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _render_one(
    smiles: str,
    size: tuple[int, int],
    opts,
) -> object | None:
    from PIL import Image
    from prepare_traindata.image import trim
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        img = Draw.MolToImage(mol, size=size, options=opts, fitImage=True)
        img = trim(img)
    except Exception:
        return None

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
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
    from prepare_traindata.rdkit_chem import d_opts

    images: list[object] = []
    for s, size in zip(cfg.smiles, cfg.sizes):
        img = _render_one(s, size, d_opts)
        if img is not None:
            images.append(img)
        if len(images) >= MAX_STRUCTURES:
            break

    if len(images) < MIN_STRUCTURES:
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
    out_path = IMAGES_DIR / filename
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
                "category_id": CATEGORY_ID,
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

def main() -> int:
    print("Loading SMILES …")
    smiles_list = load_valid_smiles(SMILES_PATH)
    total = len(smiles_list)
    print(f"  {total:,} valid SMILES loaded.")
    if total < MAX_STRUCTURES:
        print(
            f"ERROR: Need at least {MAX_STRUCTURES} valid SMILES, got {total}.",
            file=sys.stderr,
        )
        return 1

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Building sample configurations …")
    configs = build_configs(smiles_list, NUM_SAMPLES, seed=RANDOM_SEED)

    workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Generating {NUM_SAMPLES:,} samples using {workers} workers …")

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
    main()
