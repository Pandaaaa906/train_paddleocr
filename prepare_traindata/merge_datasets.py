"""Merge two COCO datasets into a single unified dataset for PaddleX training.

Handles ID remapping to avoid conflicts between source datasets (which may reuse
the same image_id ranges).  Outputs train/val split directly in PaddleX-compatible
format with globally unique IDs and continuous read_order per image.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASETS: list[Path] = [
    Path("data/synthetic_chem"),
    Path("data/dense_chem"),
]
OUTPUT_DIR: Path = Path("data/merged_chem")
IMAGES_DIR: Path = OUTPUT_DIR / "images"
ANNOTATIONS_DIR: Path = OUTPUT_DIR / "annotations"

TRAIN_SPLIT: float = 0.9
RANDOM_SEED: int = 42

CATEGORY_ID: int = 0


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


def renumber_read_order(annotations: list[dict]) -> list[dict]:
    """Re-assign read_order per image so it is 0-based continuous."""
    from collections import defaultdict

    img_anns: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        img_anns[ann["image_id"]].append(ann)

    for anns in img_anns.values():
        sorted_anns = sorted(anns, key=lambda a: (a["bbox"][1], a["bbox"][0]))
        for i, ann in enumerate(sorted_anns):
            ann["read_order"] = i

    return annotations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    all_images: list[dict] = []
    all_annotations: list[dict] = []
    img_id_offset = 0
    ann_id_offset = 0

    for dataset_dir in DATASETS:
        train_path = dataset_dir / "annotations" / "instance_train.json"
        val_path = dataset_dir / "annotations" / "instance_val.json"
        src_images_dir = dataset_dir / "images"

        for split_path in [train_path, val_path]:
            if not split_path.exists():
                print(f"WARNING: {split_path} not found, skipping.", file=sys.stderr)
                continue

            coco = load_coco(split_path)
            print(
                f"Loaded {split_path}: {len(coco['images'])} images, "
                f"{len(coco['annotations'])} boxes"
            )

            # Build local ID -> new ID mapping for this split
            local_to_new_img_id: dict[int, int] = {}
            for img in coco["images"]:
                local_to_new_img_id[img["id"]] = img_id_offset
                img["id"] = img_id_offset
                img_id_offset += 1

                # Copy image file
                src_img = src_images_dir / img["file_name"]
                if src_img.exists():
                    dst_img = IMAGES_DIR / img["file_name"]
                    dst_img.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_img, dst_img)

            for ann in coco["annotations"]:
                ann["id"] = ann_id_offset
                ann["image_id"] = local_to_new_img_id[ann["image_id"]]
                ann_id_offset += 1

            all_images.extend(coco["images"])
            all_annotations.extend(coco["annotations"])

    if not all_images:
        print("ERROR: No images found.", file=sys.stderr)
        return 1

    # Re-assign read_order per image (defensive, ensures continuity)
    all_annotations = renumber_read_order(all_annotations)

    # Shuffle images and split train/val
    import random

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_images)

    n_train = int(len(all_images) * TRAIN_SPLIT)
    train_images = all_images[:n_train]
    val_images = all_images[n_train:]

    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}

    train_anns = [a for a in all_annotations if a["image_id"] in train_ids]
    val_anns = [a for a in all_annotations if a["image_id"] in val_ids]

    category = {
        "id": CATEGORY_ID,
        "name": "image",
        "supercategory": "layout",
    }

    save_coco(
        ANNOTATIONS_DIR / "instance_train.json",
        {"images": train_images, "annotations": train_anns, "categories": [category]},
    )
    save_coco(
        ANNOTATIONS_DIR / "instance_val.json",
        {"images": val_images, "annotations": val_anns, "categories": [category]},
    )

    print(f"\nMerged dataset saved to {OUTPUT_DIR}")
    print(f"  Train: {len(train_images)} images, {len(train_anns)} boxes")
    print(f"  Val:   {len(val_images)} images, {len(val_anns)} boxes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
