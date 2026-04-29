"""Merge multiple COCO datasets into a single unified dataset for PaddleX training.

Handles ID remapping to avoid conflicts between source datasets (which may reuse
the same image_id ranges).  Outputs train/val split directly in PaddleX-compatible
format with globally unique IDs, continuous read_order per image, and all 25
PP-DocLayoutV3 categories.
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import Any

import click
import orjson

from prepare_traindata.categories import CATEGORIES
from prepare_traindata.cli import datasets, output_dir, seed, split

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_coco(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return orjson.loads(fh.read())


def save_coco(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@datasets()
@output_dir(default="data/merged_all")
@split(default=0.9)
@seed(default=42)
def main(
    datasets: tuple[str, ...],
    output_dir: str,
    split: float,
    seed: int,
) -> int:
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    annotations_dir = output_path / "annotations"

    all_images: list[dict] = []
    all_annotations: list[dict] = []
    img_id_offset = 0
    ann_id_offset = 0

    for dataset_dir in datasets:
        dataset_path = Path(dataset_dir)
        train_path = dataset_path / "annotations" / "instance_train.json"
        val_path = dataset_path / "annotations" / "instance_val.json"
        src_images_dir = dataset_path / "images"

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
                    dst_img = images_dir / img["file_name"]
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

    # Shuffle images and split train/val
    rng = random.Random(seed)
    rng.shuffle(all_images)

    if 0.0 < split < 1.0:
        n_train = int(len(all_images) * split)
        train_images = all_images[:n_train]
        val_images = all_images[n_train:]

        train_ids = {img["id"] for img in train_images}
        val_ids = {img["id"] for img in val_images}

        train_anns = [a for a in all_annotations if a["image_id"] in train_ids]
        val_anns = [a for a in all_annotations if a["image_id"] in val_ids]

        save_coco(
            annotations_dir / "instance_train.json",
            {
                "images": train_images,
                "annotations": train_anns,
                "categories": CATEGORIES,
            },
        )
        save_coco(
            annotations_dir / "instance_val.json",
            {
                "images": val_images,
                "annotations": val_anns,
                "categories": CATEGORIES,
            },
        )
        print(f"\nMerged dataset saved to {output_path}")
        print(f"  Train: {len(train_images)} images, {len(train_anns)} boxes")
        print(f"  Val:   {len(val_images)} images, {len(val_anns)} boxes")
    else:
        save_coco(
            annotations_dir / "instance_train.json",
            {
                "images": all_images,
                "annotations": all_annotations,
                "categories": CATEGORIES,
            },
        )
        print(f"\nMerged dataset saved to {output_path}")
        print(f"  Total: {len(all_images)} images, {len(all_annotations)} boxes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
