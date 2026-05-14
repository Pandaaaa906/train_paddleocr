# RDKit 结构式渲染多样性 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为所有训练数据生成器引入 RDKit 结构式随机渲染（背景色、灰度主体、原子高亮、comicMode），每张结构式独立随机。

**Architecture:** 在 `rdkit_chem.py` 中新增 `build_random_draw_options` 和 `render_mol_random` 两个纯函数；三个 generator 各自修改结构式渲染入口，为**每个结构式独立派生 `random.Random`**（`cfg.seed + i + hash(smiles)`），确保即使部分结构式渲染失败，后续结构式的随机样式仍保持可复现。现有 `d_opts` 和 `molecule_to_img` 完全保留。

**Tech Stack:** Python 3.12, RDKit, Pillow, ruff

---

## File Structure

| 文件 | 动作 | 说明 |
|------|------|------|
| `prepare_traindata/rdkit_chem.py` | 修改 | 新增 `BACKGROUNDS`、`HIGHLIGHT_COLORS`、`build_random_draw_options`、`render_mol_random` |
| `prepare_traindata/generate_dense_layout.py` | 修改 | `_render_one` 改为内部创建随机 `opts`；`_generate_sample` 中派生 `rng` |
| `prepare_traindata/generate_table_layout.py` | 修改 | `_render_structure` 签名改为 `(smiles, target_size, rng)`，调用 `render_mol_random` |
| `prepare_traindata/generate_text_mix_layout.py` | 修改 | `_render_structure` 签名同上，调用 `render_mol_random` |

---

### Task 1: Core random rendering in `rdkit_chem.py`

**Files:**
- Modify: `prepare_traindata/rdkit_chem.py`

- [ ] **Step 1: Add constants and imports**

在 `d_opts` 定义之后、`molecule_to_img` 之前插入：

```python
import random

BACKGROUNDS: list[tuple[float, float, float, float]] = [
    (1.0, 1.0, 1.0, 0.0),    # transparent 15%
    (1.0, 1.0, 1.0, 1.0),    # white      40%
    (0.98, 0.96, 0.94, 1.0),  # beige      25%
    (0.94, 0.97, 1.0, 1.0),   # light blue 12%
    (0.95, 0.95, 0.95, 1.0),  # light gray  8%
]

BACKGROUND_WEIGHTS: list[int] = [15, 40, 25, 12, 8]

HIGHLIGHT_COLORS: list[tuple[float, float, float]] = [
    (1.0, 1.0, 0.0),   # yellow
    (0.5, 1.0, 0.5),   # light green
    (1.0, 0.7, 0.2),   # light orange
]
```

- [ ] **Step 2: Add `build_random_draw_options`**

```python
def build_random_draw_options(rng: random.Random) -> rdMolDraw2D.MolDrawOptions:
    """Return a MolDrawOptions with randomized styling."""
    opts = rdMolDraw2D.MolDrawOptions()

    # 1. Background
    bg = rng.choices(BACKGROUNDS, weights=BACKGROUND_WEIGHTS)[0]
    opts.setBackgroundColour(bg)

    # 2. Atom colour (60% gray ~ black)
    gray = rng.uniform(0.0, 0.4)
    opts.setAtomPalette({k: (gray, gray, gray) for k in range(119)})

    # 3. ComicMode
    opts.comicMode = rng.random() < 0.30

    # 4. Fixed options
    opts.baseFontSize = 0.4
    opts.padding = 0.01
    opts.addStereoAnnotation = True
    opts.centreMoleculesBeforeDrawing = True
    opts.explicitMethyl = True
    opts.atomLabelDeuteriumTritium = True
    opts.simplifiedStereoGroupLabel = True

    return opts
```

- [ ] **Step 3: Add `render_mol_random`**

```python
def render_mol_random(
    mol,
    size: tuple[int, int],
    rng: random.Random,
) -> Image.Image | None:
    """Render a molecule with randomized styling. Returns None on failure."""
    opts = build_random_draw_options(rng)

    highlight_atoms = None
    highlight_color = None

    if rng.random() < 0.30:
        heavy_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        if heavy_atoms:
            n = min(rng.randint(1, 3), len(heavy_atoms))
            highlight_atoms = rng.sample(heavy_atoms, n)
            highlight_color = rng.choice(HIGHLIGHT_COLORS)

    try:
        img = Draw.MolToImage(
            mol,
            size=size,
            options=opts,
            fitImage=True,
            highlightAtoms=highlight_atoms,
            highlightColor=highlight_color,
        )
        img = trim(img)
    except Exception:
        return None

    if img is None:
        return None

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img
```

- [ ] **Step 4: Run ruff check**

```bash
ruff check prepare_traindata/rdkit_chem.py --select I,E,W
```
Expected: All checks passed

- [ ] **Step 5: Commit**

```bash
git add prepare_traindata/rdkit_chem.py
git commit -m "feat: add random RDKit rendering options and render_mol_random"
```

---

### Task 2: Update `generate_dense_layout.py`

**Files:**
- Modify: `prepare_traindata/generate_dense_layout.py`

- [ ] **Step 1: Remove `d_opts` from worker initializer**

Current initializer passes `d_opts` via `initargs`. Since options are now per-structure random, remove `d_opts` from `initargs` and remove the global `_D_OPTS` in worker.

Find:
```python
_D_OPTS = None

def _init_worker(d_opts):
    global _D_OPTS
    _D_OPTS = d_opts
    ...
```

Change to:
```python
def _init_worker() -> None:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
```

Find `ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(d_opts,))` and change to:
```python
ProcessPoolExecutor(max_workers=workers, initializer=_init_worker)
```

- [ ] **Step 2: Update `_render_one` to create random options**

Change signature from:
```python
def _render_one(smiles: str, size: tuple[int, int], opts) -> object | None:
```
to:
```python
def _render_one(
    smiles: str,
    size: tuple[int, int],
    rng: random.Random,
) -> object | None:
```

Replace body:
```python
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    from prepare_traindata.rdkit_chem import render_mol_random

    try:
        img = render_mol_random(mol, size, rng)
    except Exception:
        return None
    return img
```

- [ ] **Step 3: Remove unused `d_opts` import**

删除文件顶部的 `from prepare_traindata.rdkit_chem import d_opts`。

- [ ] **Step 4: Update call sites to use per-structure RNG**

Find:
```python
    for s, size in zip(cfg.smiles, cfg.sizes):
        img = _render_one(s, size, d_opts)
```

Replace with:
```python
    for i, (s, size) in enumerate(zip(cfg.smiles, cfg.sizes)):
        struct_rng = random.Random(cfg.seed + i + (hash(s) & 0xFFFFFFFF))
        img = _render_one(s, size, struct_rng)
```

**Why per-structure RNG:** 如果某个 SMILES 渲染失败（`mol is None`），使用 sample-level `rng` 会导致后续结构式的随机状态偏移，破坏可复现性。per-structure RNG 让每个结构式的样式仅由 `cfg.seed + i + hash(s)` 决定，与渲染成功/失败无关。

- [ ] **Step 5: Run ruff check**

```bash
ruff check prepare_traindata/generate_dense_layout.py --select I,E,W
```
Expected: All checks passed

- [ ] **Step 6: Commit**

```bash
git add prepare_traindata/generate_dense_layout.py
git commit -m "feat: use random RDKit rendering in dense_layout generator"
```

---

### Task 3: Update `generate_table_layout.py`

**Files:**
- Modify: `prepare_traindata/generate_table_layout.py`

- [ ] **Step 1: Update `_render_structure` signature and body**

Current:
```python
def _render_structure(smiles: str, target_size: tuple[int, int]) -> Any | None:
    from PIL import Image
    from rdkit.Chem import Draw
    from prepare_traindata.image import trim
    from prepare_traindata.rdkit_chem import d_opts

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        img = Draw.MolToImage(mol, size=target_size, options=d_opts, fitImage=True)
        img = trim(img)
    except Exception:
        return None

    if img is None:
        return None

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img
```

Replace with:
```python
def _render_structure(
    smiles: str,
    target_size: tuple[int, int],
    rng: random.Random,
) -> Any | None:
    from PIL import Image
    from prepare_traindata.rdkit_chem import render_mol_random

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        img = render_mol_random(mol, target_size, rng)
    except Exception:
        return None

    if img is None:
        return None

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img
```

- [ ] **Step 2: Remove unused `d_opts` import**

在 `_render_structure` 中，删除 `from prepare_traindata.rdkit_chem import d_opts` 和 `from rdkit.Chem import Draw`、`from prepare_traindata.image import trim`（这些已移至 `render_mol_random` 内部）。

- [ ] **Step 3: Update call sites with per-structure RNG**

Find:
```python
for _ in range(3):
    img = _render_structure(smiles_to_try, (target_w, target_h))
    if img is not None:
        break
    if _WORKER_SMILES_POOL:
        smiles_to_try = rng.choice(_WORKER_SMILES_POOL)
```

Replace with:
```python
struct_rng = random.Random(cfg.seed + idx + (hash(smiles_to_try) & 0xFFFFFFFF))
for attempt in range(3):
    img = _render_structure(smiles_to_try, (target_w, target_h), struct_rng)
    if img is not None:
        break
    if _WORKER_SMILES_POOL:
        smiles_to_try = rng.choice(_WORKER_SMILES_POOL)
        struct_rng = random.Random(cfg.seed + idx + (hash(smiles_to_try) & 0xFFFFFFFF))
```

**Note:** 每次重试切换 SMILES 时，重新派生 `struct_rng`；同一次重试链内使用同一个 `struct_rng` 是确定性的。

- [ ] **Step 4: Run ruff check**

```bash
ruff check prepare_traindata/generate_table_layout.py --select I,E,W
```

- [ ] **Step 5: Commit**

```bash
git add prepare_traindata/generate_table_layout.py
git commit -m "feat: use random RDKit rendering in table_layout generator"
```

---

### Task 4: Update `generate_text_mix_layout.py`

**Files:**
- Modify: `prepare_traindata/generate_text_mix_layout.py`

- [ ] **Step 1: Update `_render_structure` signature and body**

Same changes as Task 3. Change signature to add `rng: random.Random` parameter and replace body to call `render_mol_random`.

- [ ] **Step 2: Remove unused imports in `_render_structure`**

删除 `_render_structure` 中的 `from prepare_traindata.image import trim`、`from rdkit.Chem import Draw`、`from prepare_traindata.rdkit_chem import d_opts`。

- [ ] **Step 3: Update call site with per-structure RNG**

Find:
```python
    for s in cfg.smiles:
        img = _render_structure(s, (220, 160))
```

Change to:
```python
    for i, s in enumerate(cfg.smiles):
        struct_rng = random.Random(cfg.seed + i + (hash(s) & 0xFFFFFFFF))
        img = _render_structure(s, (220, 160), struct_rng)
```

- [ ] **Step 4: Run ruff check**

```bash
ruff check prepare_traindata/generate_text_mix_layout.py --select I,E,W
```

- [ ] **Step 5: Commit**

```bash
git add prepare_traindata/generate_text_mix_layout.py
git commit -m "feat: use random RDKit rendering in text_mix_layout generator"
```

---

### Task 5: Integration test

**Files:**
- No file changes (validation only)

- [ ] **Step 1: Generate 20 test samples from text_mix_layout**

```bash
rm -rf data/render_test
uv run python -m prepare_traindata.generate_text_mix_layout \
  --num-samples 20 \
  --output-dir data/render_test \
  --workers 0 \
  --no-watermark
```
Expected: 20 images generated successfully

- [ ] **Step 2: Visual inspection script**

```python
from pathlib import Path
from PIL import Image

img_dir = Path("data/render_test/images")
for p in sorted(img_dir.glob("*.png"))[:5]:
    img = Image.open(p)
    print(f"{p.name}: {img.size} mode={img.mode}")
```

Run:
```bash
uv run python -c "
from pathlib import Path
from PIL import Image
for p in sorted(Path('data/render_test/images').glob('*.png'))[:5]:
    print(f'{p.name}: {Image.open(p).size}')
"
```

- [ ] **Step 3: Manual visual check**

Open 5 images and verify:
- At least 2 different background colors visible
- Some structure lines are gray instead of pure black
- ~30% of structures have highlighted atoms
- ~30% of structures are in comic mode

- [ ] **Step 4: Generate 10 samples from dense_layout**

```bash
rm -rf data/render_test2
uv run python -m prepare_traindata.generate_dense_layout \
  --num-samples 10 \
  --output-dir data/render_test2 \
  --workers 0
```

- [ ] **Step 5: Generate 10 samples from table_layout**

```bash
rm -rf data/render_test3
uv run python -m prepare_traindata.generate_table_layout \
  --num-samples 10 \
  --output-dir data/render_test3 \
  --workers 0
```

- [ ] **Step 6: Clean up test dirs**

```bash
rm -rf data/render_test data/render_test2 data/render_test3
```

- [ ] **Step 7: Final commit**

```bash
git commit --allow-empty -m "test: verify random rendering across all generators"
```

---

## Notes

- `generate_synthetic_layout.py`（旧版随机散落布局）** intentionally 不修改**，该生成器已标记为 legacy，当前训练不依赖它。

## Rollback Plan

若出现问题，回滚到上一次提交即可：

```bash
git tag pre-render-diversity HEAD~5  # 可选：先打标签
# 回滚全部 5 个 commit（Task 1-4 + integration test）
git revert HEAD~5..HEAD
```

或单独回滚某个 generator：
```bash
git revert <commit-hash-of-that-task>
```
