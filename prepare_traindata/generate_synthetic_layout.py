"""Generate synthetic training data for PP-DocLayoutV3 fine-tuning.

Each output image contains 2-10 non-overlapping chemical structure images
randomly placed on a white canvas. Annotations are emitted in COCO format
with category_id=14 (``image`` in PP-DocLayoutV3's label space).

The script uses process-based parallelism to saturate CPU cores and speed up
the generation of thousands of samples.
"""

from __future__ import annotations

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SMILES_PATH: Path = Path("smiles.txt")
OUTPUT_DIR: Path = Path("data/synthetic_chem")
IMAGES_DIR: Path = OUTPUT_DIR / "images"
ANNOTATIONS_DIR: Path = OUTPUT_DIR / "annotations"

NUM_SAMPLES: int = 5_000
MIN_STRUCTURES: int = 2
MAX_STRUCTURES: int = 10

CANVAS_WIDTH_RANGE: tuple[int, int] = (800, 1_400)
CANVAS_HEIGHT_RANGE: tuple[int, int] = (1_000, 1_800)
STRUCTURE_WIDTH_RANGE: tuple[int, int] = (200, 300)
STRUCTURE_HEIGHT_RANGE: tuple[int, int] = (80, 120)

CANVAS_MARGIN: int = 20
MAX_PLACEMENT_ATTEMPTS: int = 100
RANDOM_SEED: int | None = 42

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_valid_smiles(path: Path) -> list[str]:
    """Return SMILES strings that RDKit can parse."""
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
    canvas_size: tuple[int, int]


def build_configs(
    smiles_list: list[str],
    num_samples: int,
    seed: int | None = None,
) -> list[SampleConfig]:
    """Build a deterministic list of sample configurations."""
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
        canvas = (
            rng.randint(*CANVAS_WIDTH_RANGE),
            rng.randint(*CANVAS_HEIGHT_RANGE),
        )
        configs.append(SampleConfig(idx, chosen, sizes, canvas))
    return configs


# ---------------------------------------------------------------------------
# Rendering helpers (executed inside worker processes)
# ---------------------------------------------------------------------------

def _init_worker():
    """Executed once per worker process to set up imports."""
    # Ensure project root is on path so ``prepare_traindata`` imports work.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _render_one(
    smiles: str,
    size: tuple[int, int],
    opts,
) -> object | None:
    """Render a single SMILES to a trimmed RGB PIL Image."""
    from PIL import Image

    from prepare_traindata.image import trim

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import Draw

        img = Draw.MolToImage(mol, size=size, options=opts, fitImage=True)
        img = trim(img)
    except Exception:  # noqa: BLE001
        return None

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


class SampleResult(NamedTuple):
    img_id: int
    filename: str
    width: int
    height: int
    annotations: list[dict]


def _generate_sample(cfg: SampleConfig) -> SampleResult | None:
    """Generate one composite image and its annotations."""
    from PIL import Image

    from prepare_traindata.rdkit_chem import d_opts

    images: list[Image.Image] = []
    for s, size in zip(cfg.smiles, cfg.sizes):
        img = _render_one(s, size, d_opts)
        if img is not None:
            images.append(img)
        if len(images) >= MAX_STRUCTURES:
            break

    if len(images) < MIN_STRUCTURES:
        return None

    canvas_w, canvas_h = cfg.canvas_size
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    bboxes: list[dict] = []
    placed: list[tuple[int, int, int, int]] = []

    for img in images:
        w, h = img.size
        max_x = canvas_w - w - CANVAS_MARGIN
        max_y = canvas_h - h - CANVAS_MARGIN
        if max_x < CANVAS_MARGIN or max_y < CANVAS_MARGIN:
            continue

        placed_ok = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            x = random.randint(CANVAS_MARGIN, max_x)
            y = random.randint(CANVAS_MARGIN, max_y)
            cand = (x, y, x + w, y + h)
            if not any(_boxes_overlap(cand, pb) for pb in placed):
                canvas.paste(img, (x, y))
                placed.append(cand)
                bboxes.append({"bbox": [x, y, w, h], "category_id": 14})
                placed_ok = True
                break
        if not placed_ok:
            pass

    if not bboxes:
        return None

    filename = f"chem_{cfg.sample_idx:06d}.png"
    out_path = IMAGES_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    annotations = []
    for local_idx, box in enumerate(bboxes):
        x, y, w, h = box["bbox"]
        annotations.append(
            {
                "id": cfg.sample_idx * 1_000 + local_idx,
                "image_id": cfg.sample_idx,
                "category_id": box["category_id"],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
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
        "categories": [
            {
                "id": 14,
                "name": "image",
                "supercategory": "layout",
            }
        ],
    }

    ann_path = ANNOTATIONS_DIR / "instances_train.json"
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
