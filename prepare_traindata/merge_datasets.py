"""Merge multiple COCO datasets into a single unified dataset for PaddleX training.

Handles ID remapping to avoid conflicts between source datasets (which may reuse
the same image_id ranges).  Outputs train/val split directly in PaddleX-compatible
format with globally unique IDs, continuous read_order per image, and all 25
PP-DocLayoutV3 categories.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASETS: list[Path] = [
    Path("data/dense_chem"),
    Path("data/table_layout"),
]
OUTPUT_DIR: Path = Path("data/merged_all")
IMAGES_DIR: Path = OUTPUT_DIR / "images"
ANNOTATIONS_DIR: Path = OUTPUT_DIR / "annotations"

TRAIN_SPLIT: float = 0.9
RANDOM_SEED: int = 42

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
    {"id": 23, "name": "vertical_text"},
    {"id": 24, "name": "vision_footnote"},
]


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
                print(f"  Skipping {split_path} (not found)")
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
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_images)

    n_train = int(len(all_images) * TRAIN_SPLIT)
    train_images = all_images[:n_train]
    val_images = all_images[n_train:]

    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}

    train_anns = [a for a in all_annotations if a["image_id"] in train_ids]
    val_anns = [a for a in all_annotations if a["image_id"] in val_ids]

    save_coco(
        ANNOTATIONS_DIR / "instance_train.json",
        {"images": train_images, "annotations": train_anns, "categories": CATEGORIES},
    )
    save_coco(
        ANNOTATIONS_DIR / "instance_val.json",
        {"images": val_images, "annotations": val_anns, "categories": CATEGORIES},
    )

    print(f"\nMerged dataset saved to {OUTPUT_DIR}")
    print(f"  Train: {len(train_images)} images, {len(train_anns)} boxes")
    print(f"  Val:   {len(val_images)} images, {len(val_anns)} boxes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
