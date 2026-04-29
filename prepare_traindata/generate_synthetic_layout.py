"""Generate synthetic training data for PP-DocLayoutV3 fine-tuning.

Each output image contains 2-10 non-overlapping chemical structure images
randomly placed on a white canvas. Annotations are emitted in COCO format
with category_id=14 (``image`` in PP-DocLayoutV3's label space).

The script uses process-based parallelism to saturate CPU cores and speed up
the generation of thousands of samples.
"""

from __future__ import annotations

import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import click
import orjson
from rdkit import Chem, RDLogger

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

CANVAS_WIDTH_RANGE: tuple[int, int] = (800, 1_400)
CANVAS_HEIGHT_RANGE: tuple[int, int] = (1_000, 1_800)

CANVAS_MARGIN: int = 20
MAX_PLACEMENT_ATTEMPTS: int = 100

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


class Placement(NamedTuple):
    x: int
    y: int
    w: int
    h: int


class SampleConfig(NamedTuple):
    sample_idx: int
    smiles: list[str]
    sizes: list[tuple[int, int]]
    canvas_size: tuple[int, int]
    seed: int
    output_dir: Path
    use_watermark: bool


def build_configs(
    smiles_list: list[str],
    num_samples: int,
    seed: int,
    output_dir: Path,
    use_watermark: bool,
    *,
    min_structures: int = 2,
    max_structures: int = 10,
    canvas_width_range: tuple[int, int] = (800, 1_400),
    canvas_height_range: tuple[int, int] = (1_000, 1_800),
    structure_width_range: tuple[int, int] = (200, 300),
    structure_height_range: tuple[int, int] = (80, 120),
) -> list[SampleConfig]:
    """Build a deterministic list of sample configurations."""
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
        canvas = (
            rng.randint(*canvas_width_range),
            rng.randint(*canvas_height_range),
        )
        configs.append(
            SampleConfig(
                idx,
                chosen,
                sizes,
                canvas,
                seed=seed + idx,
                output_dir=output_dir,
                use_watermark=use_watermark,
            )
        )
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

    from prepare_traindata import watermark_utils
    from prepare_traindata.rdkit_chem import d_opts

    rng = random.Random(cfg.seed)
    images: list[Image.Image] = []
    for s, size in zip(cfg.smiles, cfg.sizes):
        img = _render_one(s, size, d_opts)
        if img is not None:
            images.append(img)
        if len(images) >= len(cfg.smiles):
            break

    if len(images) < 2:
        return None

    canvas_w, canvas_h = cfg.canvas_size
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    if cfg.use_watermark:
        canvas = watermark_utils.apply_random_watermark(canvas, rng)

    placements: list[Placement] = []
    placed: list[tuple[int, int, int, int]] = []

    for img in images:
        w, h = img.size
        max_x = canvas_w - w - CANVAS_MARGIN
        max_y = canvas_h - h - CANVAS_MARGIN
        if max_x < CANVAS_MARGIN or max_y < CANVAS_MARGIN:
            continue

        placed_ok = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            x = rng.randint(CANVAS_MARGIN, max_x)
            y = rng.randint(CANVAS_MARGIN, max_y)
            cand = (x, y, x + w, y + h)
            if not any(_boxes_overlap(cand, pb) for pb in placed):
                canvas.paste(img, (x, y))
                placed.append(cand)
                placements.append(Placement(x, y, w, h))
                placed_ok = True
                break
        if not placed_ok:
            pass

    if not placements:
        return None

    filename = f"chem_{cfg.sample_idx:06d}.png"
    images_dir = cfg.output_dir / "images"
    out_path = images_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    # Sort by (y, x) for read_order
    placements.sort(key=lambda p: (p.y, p.x))

    annotations = []
    for read_order, pl in enumerate(placements):
        x, y, w, h = pl
        annotations.append(
            {
                "id": cfg.sample_idx * 1_000 + read_order,
                "image_id": cfg.sample_idx,
                "category_id": CAT_ID_IMAGE,
                "bbox": [x, y, w, h],
                "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
                "area": w * h,
                "iscrowd": 0,
                "read_order": read_order,
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
@output_dir(default="data/synthetic_chem")
@num_samples(default=5000)
@seed(default=42)
@workers(default=0)
@split(default=0.9)
@watermark(default=True)
@min_structures(default=2)
@max_structures(default=10)
@click.option(
    "--canvas-width-range",
    type=(int, int),
    default=(800, 1_400),
    show_default=True,
    callback=lambda ctx, param, value: value if value[0] <= value[1] else click.BadParameter(f"{param.name} min must be <= max"),
    help="Canvas width range as two integers (min max).",
)
@click.option(
    "--canvas-height-range",
    type=(int, int),
    default=(1_000, 1_800),
    show_default=True,
    callback=lambda ctx, param, value: value if value[0] <= value[1] else click.BadParameter(f"{param.name} min must be <= max"),
    help="Canvas height range as two integers (min max).",
)
@structure_width_range(default=(200, 300))
@structure_height_range(default=(80, 120))
def main(
    output_dir: str,
    num_samples: int,
    seed: int,
    workers: int,
    split: float,
    watermark: bool,
    min_structures: int,
    max_structures: int,
    canvas_width_range: tuple[int, int],
    canvas_height_range: tuple[int, int],
    structure_width_range: tuple[int, int],
    structure_height_range: tuple[int, int],
) -> int:
    print("Loading SMILES …")
    smiles_list = _load_valid_smiles(SMILES_PATH)
    total = len(smiles_list)
    print(f"  {total:,} valid SMILES loaded.")
    if total < max_structures:
        print(
            f"ERROR: Need at least {max_structures} valid SMILES, got {total}.",
            file=sys.stderr,
        )
        return 1

    out_path = Path(output_dir)
    images_dir = out_path / "images"
    annotations_dir = out_path / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    print("Building sample configurations …")
    configs = build_configs(
        smiles_list,
        num_samples,
        seed=seed,
        output_dir=out_path,
        use_watermark=watermark,
        min_structures=min_structures,
        max_structures=max_structures,
        canvas_width_range=canvas_width_range,
        canvas_height_range=canvas_height_range,
        structure_width_range=structure_width_range,
        structure_height_range=structure_height_range,
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

    coco_images.sort(key=lambda img: img["id"])
    coco_annotations.sort(key=lambda ann: ann["id"])

    def _write_coco(images: list[dict], annotations: list[dict], path: Path) -> None:
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": CATEGORIES,
        }
        path.write_bytes(orjson.dumps(coco, option=orjson.OPT_INDENT_2))

    if 0.0 < split < 1.0:
        n_train = int(len(coco_images) * split)
        train_images = coco_images[:n_train]
        val_images = coco_images[n_train:]
        train_ids = {img["id"] for img in train_images}
        val_ids = {img["id"] for img in val_images}
        train_annotations = [ann for ann in coco_annotations if ann["image_id"] in train_ids]
        val_annotations = [ann for ann in coco_annotations if ann["image_id"] in val_ids]
        _write_coco(train_images, train_annotations, annotations_dir / "instance_train.json")
        _write_coco(val_images, val_annotations, annotations_dir / "instance_val.json")
        print(f"  Split: {len(train_images)} train, {len(val_images)} val")
    else:
        _write_coco(coco_images, coco_annotations, annotations_dir / "instance_train.json")

    print("\nDone.")
    print(f"  Images:      {images_dir}")
    print(f"  Annotations: {annotations_dir}")
    print(f"  Total images written: {len(coco_images):,}")
    print(f"  Total boxes written:  {len(coco_annotations):,}")
    return 0


def _load_valid_smiles(path: Path) -> list[str]:
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


if __name__ == "__main__":
    sys.exit(main())
