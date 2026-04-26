# train_paddleocr

## Project Goal
Fine-tune **PP-DocLayoutV3** to accurately detect and crop individual chemical structures from documents/images that contain multiple structures.

The pre-trained model tends to merge adjacent chemical structures into one large `image` region. We are improving its ability to separate them into distinct bounding boxes.

## Tech Stack
- **Python 3.12**
- **Package Manager**: `uv`
- **Chemistry**: RDKit (structure rendering)
- **Vision/OCR**: PaddleOCR / PaddleX / PaddlePaddle
- **Data Format**: COCO JSON

## Project Structure

```
.
├── smiles.txt                          # Source SMILES strings (~15K lines)
├── prepare_traindata/
│   ├── __init__.py
│   ├── image.py                        # `trim(im, margin=5)` — crop whitespace
│   ├── rdkit_chem.py                   # RDKit rendering helpers (`molecule_to_img`)
│   └── generate_synthetic_layout.py    # Synthetic data generator (see below)
├── data/synthetic_chem/
│   ├── images/                         # 5,000 composite PNGs (2-10 structures each)
│   └── annotations/
│       └── instances_train.json        # COCO format, category_id=14 (`image`)
└── CLAUDE.md                           # This file
```

## Synthetic Data Generation

**Script**: `prepare_traindata/generate_synthetic_layout.py`

- Reads valid SMILES from `smiles.txt`
- Renders each structure via RDKit (`molecule_to_img`) and trims whitespace to ~5px margin
- Places 2–10 structures randomly on a white canvas (800×1000 – 1400×1800 px)
- Prevents overlap via random placement with collision detection
- Outputs COCO-format annotations with `category_id=14` (maps to PP-DocLayoutV3 `image`)
- Uses multiprocessing (`ProcessPoolExecutor`) for speed

**Current Dataset Stats**
- Images: 5,000
- Annotations: ~30,110 boxes (~6 per image)
- Category: `image` only

## Fine-tuning Plan (Pending)

### Option A: Single-class fine-tuning (recommended)
- Treat the task as a single-class detector (`image` / chemical structure)
- Change dataset `category_id` from `14` → `0`
- Set model `num_classes=1`
- Fastest, most focused. Loses ability to detect text/tables/etc.

### Option B: 25-class full fine-tuning
- Keep all PP-DocLayoutV3 categories
- Requires pseudo-labeling the other 24 classes on synthetic images (or mixing real document data)
- Preserves general layout analysis capability

### Required Next Steps
1. Confirm training hardware (GPU / CPU / CUDA version)
2. Install `paddlepaddle-gpu` (or `paddlepaddle`) + `paddlex`
3. Prepare PaddleX config YAML (dataset path, `num_classes`, epochs, batch size)
4. Run training (`python main.py -c config.yaml ...` or `paddlex` CLI)
5. Export inference model for use with PaddleOCR-VL pipeline

## Dependencies (pyproject.toml)
- `rdkit`
- `pillow`
- `tqdm`
- *(pending)* `paddlepaddle-gpu` / `paddlex`

## Key References
- PP-DocLayoutV3 has 25 fixed layout categories; `image` = ID 14
- COCO bbox format: `[x, y, width, height]`
- Official model weights available on Hugging Face & Paddle model zoo
