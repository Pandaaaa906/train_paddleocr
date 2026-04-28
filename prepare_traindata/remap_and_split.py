"""Remap COCO category IDs and create train/val split for PaddleX layout analysis.

PP-DocLayoutV3 is registered as a layout-analysis (instance-segmentation) model in
PaddleX.  The COCO annotations must contain:

* ``segmentation`` – polygon mask for each instance (we synthesise a rectangle from bbox)
* ``read_order``   – non-negative integer, 0-based continuous per image
* ``category_id``  – remap synthetic single-class output (0) back to original image class (14)

The output filenames follow PaddleX convention:
``instance_train.json`` and ``instance_val.json``.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from prepare_traindata.categories import CATEGORIES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR: Path = Path("data/table_layout")
ANNOTATIONS_DIR: Path = DATASET_DIR / "annotations"
SOURCE_ANNO: Path = ANNOTATIONS_DIR / "instance_train.json"

TRAIN_ANNO: Path = ANNOTATIONS_DIR / "instance_train.json"
VAL_ANNO: Path = ANNOTATIONS_DIR / "instance_val.json"

TRAIN_SPLIT: float = 0.9
RANDOM_SEED: int = 42

OLD_CATEGORY_ID: int = 0
NEW_CATEGORY_ID: int = 14


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_coco(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    """Convert COCO bbox [x, y, w, h] to a rectangular polygon."""
    x, y, w, h = bbox
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def remap_and_enhance(coco: dict[str, Any]) -> dict[str, Any]:
    """Remap category_id, add segmentation + read_order."""
    # Group annotations by image
    img_to_anns: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    new_annotations: list[dict] = []
    for img_id, anns in img_to_anns.items():
        # Remap category_id
        for ann in anns:
            if ann.get("category_id") == OLD_CATEGORY_ID:
                ann["category_id"] = NEW_CATEGORY_ID
            # Add segmentation from bbox
            ann["segmentation"] = bbox_to_polygon(ann["bbox"])
        # Assign read_order
        new_annotations.extend(anns)

    coco["annotations"] = new_annotations
    coco["categories"] = CATEGORIES
    return coco


def split_images(images: list[dict], split_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = images.copy()
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * split_ratio)
    return shuffled[:n_train], shuffled[n_train:]


def build_split_coco(
    coco: dict[str, Any],
    image_ids: set[int],
    category: dict[str, Any],
) -> dict[str, Any]:
    images = [img for img in coco["images"] if img["id"] in image_ids]
    annotations = [
        ann for ann in coco["annotations"] if ann["image_id"] in image_ids
    ]
    return {
        "images": images,
        "annotations": annotations,
        "categories": [category],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not SOURCE_ANNO.exists():
        print(f"ERROR: Source annotation not found: {SOURCE_ANNO}", file=sys.stderr)
        return 1

    print(f"Loading COCO annotations from {SOURCE_ANNO} …")
    coco = load_coco(SOURCE_ANNO)

    original_count = len(coco.get("images", []))
    original_boxes = len(coco.get("annotations", []))
    print(f"  Images: {original_count:,}")
    print(f"  Boxes:  {original_boxes:,}")

    # Remap categories, add segmentation + read_order
    print("Remapping category_id 0 → 14, adding segmentation masks and read_order …")
    coco = remap_and_enhance(coco)

    # Train/val split
    print(f"Splitting {TRAIN_SPLIT:.0%} train / {1 - TRAIN_SPLIT:.0%} val (seed={RANDOM_SEED}) …")
    train_images, val_images = split_images(coco["images"], TRAIN_SPLIT, RANDOM_SEED)
    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}

    category = coco["categories"][0]
    train_coco = build_split_coco(coco, train_ids, category)
    val_coco = build_split_coco(coco, val_ids, category)

    # Save
    save_coco(TRAIN_ANNO, train_coco)
    save_coco(VAL_ANNO, val_coco)

    print(f"\nSaved:")
    print(f"  Train: {TRAIN_ANNO}  ({len(train_coco['images']):,} images, {len(train_coco['annotations']):,} boxes)")
    print(f"  Val:   {VAL_ANNO}  ({len(val_coco['images']):,} images, {len(val_coco['annotations']):,} boxes)")
    return 0


if __name__ == "__main__":
    main()
