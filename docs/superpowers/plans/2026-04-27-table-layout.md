# table_layout 数据集生成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成 2500 张合成表格图片，其中结构式图片随机嵌入单元格，用于微调 PP-DocLayoutV3 模型，同时修复水印背景下的切分问题。

**Architecture:** 新增 3 个纯 Python 模块（text_vocab, watermark_utils, generate_table_layout），复用现有 rdkit_chem/image 工具，输出 PaddleX-ready COCO 格式数据集到 `data/table_layout/`。

**Tech Stack:** Python 3.12, Pillow, RDKit, PaddleX COCO format

---

## Context

- 现有 `prepare_traindata/generate_dense_layout.py` 生成密集垂直排列的结构式（5000 张），用于训练 `image` 类切分。
- `prepare_traindata/rdkit_chem.py` 提供 `molecule_to_img()` 和 `d_opts`。
- `prepare_traindata/image.py` 提供 `trim()`。
- `prepare_traindata/remap_and_split.py` 处理 COCO 格式 remap 和 split（但本次输出直接为 PaddleX 格式，不需要 remap）。
- PP-DocLayoutV3 共 25 个分类。本次需要保留全部 25 类 categories，但图片中只出现 `table` (id=21) 和 `image` (id=14)。
- read_order: 先 table (order=0)，后单元格内图片 (order=1,2,3...)，从上到下、从左到右。
- 背景水印：为了解决之前 dense_layout 对浅色水印背景切分效果不好的问题，本次加入随机水印底纹。

---

### Task 1: text_vocab.py — 随机词组词库

**Files:**
- Create: `prepare_traindata/text_vocab.py`

- [ ] **Step 1: Write the module**

```python
# prepare_traindata/text_vocab.py
"""Random vocabulary for filling table cells."""

import random

# 中英文词组样本
CAS_SAMPLES = [...]  # ~20 CAS号样例
PURITY_SAMPLES = [...]  # ~10 纯度样例
MW_SAMPLES = [...]  # ~10 分子量样例
PRODUCT_NAMES = [...]  # ~30 产品名称（中英文混合）
SPECIFICATIONS = [...]  # ~10 规格样例
BATCH_NUMBERS = [...]  # ~10 批号样例
APPEARANCE = [...]  # ~10 外观样例
NOTES = [...]  # ~10 备注样例

ALL_SAMPLES = (
    CAS_SAMPLES + PURITY_SAMPLES + MW_SAMPLES +
    PRODUCT_NAMES + SPECIFICATIONS + BATCH_NUMBERS +
    APPEARANCE + NOTES
)

def get_random_text(rng: random.Random) -> str:
    """Return a random text string suitable for a table cell."""
    ...
```

要求：
- `get_random_text` 从 `ALL_SAMPLES` 随机选择，有一定概率（如 20%）将 2-3 个样本用空格连接，模拟长短不一的单元格内容。
- 所有样本为纯字符串，不涉及外部文件读取。
- 使用 `random.Random` 实例作为 rng，保持可复现性。

- [ ] **Step 2: Verify import and output**

Run: `python -c "from prepare_traindata.text_vocab import get_random_text, ALL_SAMPLES; import random; rng=random.Random(42); print([get_random_text(rng) for _ in range(5)])"`
Expected: 输出 5 个随机字符串，无异常。

- [ ] **Step 3: Commit**

```bash
git add prepare_traindata/text_vocab.py
git commit -m "feat: add random vocabulary for table cell filling"
```

---

### Task 2: watermark_utils.py — 随机水印底纹

**Files:**
- Create: `prepare_traindata/watermark_utils.py`

- [ ] **Step 1: Write the module**

实现 3 种水印类型 + 随机混合器：

1. `text_watermark(canvas_size, text, rng)` — 倾斜文字水印，低透明度，随机颜色（灰/浅蓝/浅绿）。
2. `texture_watermark(canvas_size, rng)` — 几何纹理（网格、点阵、斜线），低透明度。
3. `logo_watermark(canvas_size, rng)` — 简单图形（六边形、波浪线）作为 logo，低透明度。

以及 `apply_random_watermark(canvas, rng)` — 随机选择 0-3 种水印叠加到给定 PIL Image 上。

要求：
- 输入 canvas 为 PIL Image (RGB)，输出也是 RGB。
- 水印使用 RGBA 图层，通过 `Image.alpha_composite` 或 `paste(mask=...)` 叠加，再转回 RGB。
- 所有随机参数（颜色、角度、密度、透明度）通过传入的 `rng` 控制。
- 不涉及外部字体文件（使用默认 PIL 字体或纯图形绘制）。

- [ ] **Step 2: Verify import and output**

Run: `python -c "from PIL import Image; from prepare_traindata.watermark_utils import apply_random_watermark; import random; rng=random.Random(42); im=Image.new('RGB',(400,300),(255,255,255)); im=apply_random_watermark(im, rng); im.save('output/test_watermark.png')"`
Expected: 生成一张带水印的白色图片，无异常。

- [ ] **Step 3: Commit**

```bash
git add prepare_traindata/watermark_utils.py
git commit -m "feat: add random watermark generators for synthetic backgrounds"
```

---

### Task 3: generate_table_layout.py — 主生成器

**Files:**
- Create: `prepare_traindata/generate_table_layout.py`

- [ ] **Step 1: Write the module**

生成 2500 张表格图片，规格：
- 列数：1-5 列随机
- 行数：1-15 行随机
- 每个单元格尺寸：宽度 100-300px，高度 80-200px（随机）
- 边框宽度：2-4px 随机
- 结构式图片随机放入某些单元格（一个单元格一个结构式），其余单元格用 `text_vocab.get_random_text()` 填充
- 结构式渲染复用 `rdkit_chem.molecule_to_img()` 和 `image.trim()`
- 结构式图片缩放：如果大于单元格，等比缩放至适配单元格（保留 5-10px 边距）
- 背景先绘制白色，再应用 `watermark_utils.apply_random_watermark()`
- 表格绘制使用 Pillow `ImageDraw`，线条颜色深灰/黑色

COCO 标注要求：
- `category_id`: `table`=21, `image`=14
- `bbox`: [x, y, w, h]
- `segmentation`: 矩形多边形 [[x,y, x+w,y, x+w,y+h, x,y+h]]
- `area`: w * h
- `read_order`: table 为 0，每个 image 按从上到下、从左到右依次为 1,2,3...
- `id`: 全局唯一（image_id * 1000 + local_idx）
- `iscrowd`: 0

Categories 输出必须保留全部 25 类（参考 PP-DocLayoutV3 官方分类），即使图片中只有 21 和 14 出现。

多进程：
- 使用 `ProcessPoolExecutor`（类似 `generate_dense_layout.py`）
- 通过 `initializer` 加载 SMILES pool 到全局变量，避免每个 worker 重新读取

输出目录：`data/table_layout/images/` 和 `data/table_layout/annotations/instance_train.json`

- [ ] **Step 2: 快速测试生成 5 张样例**

修改脚本内 `NUM_SAMPLES=5`，运行：
```bash
uv run python prepare_traindata/generate_table_layout.py
```
检查 `data/table_layout/images/` 下是否生成 5 张图片，且 JSON 格式正确。

- [ ] **Step 3: Commit**

```bash
git add prepare_traindata/generate_table_layout.py
git commit -m "feat: add table-layout synthetic data generator"
```

---

### Task 4: 运行完整生成 (2500 张)

- [ ] **Step 1: 修改 NUM_SAMPLES 为 2500**

在 `generate_table_layout.py` 中将 `NUM_SAMPLES` 改回 2500。

- [ ] **Step 2: 执行生成**

```bash
uv run python prepare_traindata/generate_table_layout.py
```

- [ ] **Step 3: 验证输出**

1. 检查 `data/table_layout/images/` 是否有 2500 张 png
2. 检查 `data/table_layout/annotations/instance_train.json` 是否存在且格式正确
3. 抽样检查几张图片，确认：
   - 表格线条清晰
   - 结构式在单元格内，未超出
   - 水印可见但不过分干扰
   - JSON 中 categories 包含 25 类

- [ ] **Step 4: Commit 数据集元数据（不提交图片本身）**

更新 `.gitignore` 忽略 `data/table_layout/images/`（如果还没忽略的话）。

```bash
git add .gitignore data/table_layout/annotations/instance_train.json
git commit -m "feat: generate 2500 table-layout synthetic images with annotations"
```

---

### Task 5: 更新训练配置 (可选)

- [ ] **Step 1: 更新 configs/PP-DocLayoutV3.yaml**

将 `dataset_dir` 改为 `data/table_layout`（或合并后的数据集路径）。
如果用户选择将 table_layout 与 dense_layout 合并，则跳过此步，改用 `merge_datasets.py`。

- [ ] **Step 2: Commit**

```bash
git add configs/PP-DocLayoutV3.yaml
git commit -m "chore: update training config for table_layout dataset"
```

---

## PP-DocLayoutV3 25 分类清单（categories 必须全部保留）

| id | name |
|---|---|
| 0 | abstract |
| 1 | algorithm |
| 2 | aside_text |
| 3 | chart |
| 4 | content |
| 5 | display_formula |
| 6 | doc_title |
| 7 | figure_title |
| 8 | footer |
| 9 | footer_image |
| 10 | footnote |
| 11 | formula_number |
| 12 | header |
| 13 | header_image |
| 14 | image |
| 15 | inline_formula |
| 16 | number |
| 17 | paragraph_title |
| 18 | reference |
| 19 | reference_content |
| 20 | seal |
| 21 | table |
| 22 | text |
| 23 | table_caption |
| 24 | vision_footnote |
