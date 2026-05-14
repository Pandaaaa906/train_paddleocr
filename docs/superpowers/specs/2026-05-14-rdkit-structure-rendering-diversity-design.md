# RDKit 结构式渲染多样性设计

## 背景

当前所有训练数据生成器生成的化学结构式均为**黑色线条 + 透明/白色背景**，视觉风格单一。为提升模型对真实文档中多样化结构式图像的泛化能力，需引入随机渲染参数。

## 目标

每张结构式图像在渲染时**独立随机**以下维度，使训练集覆盖更多真实场景：

1. **背景颜色**：透明、白色、米白、浅蓝、浅灰随机选择
2. **分子主体颜色**：默认黑色 → 随机 60% 灰到黑色（RGB 各通道 0.0~0.4）
3. **原子高亮**：30% 概率随机高亮 1-3 个非氢原子
4. **ComicMode**：30% 概率开启卡通风格

## 方案：B — 每张结构式独立随机

### 为什么选 B

- 真实 PDF/截图中，同一页面上的结构式可能来自不同来源（ChemDraw、Marvin、截图拼接），风格差异合理
- 更强的多样性 = 更好的泛化
- 不影响布局逻辑，渲染层独立变化

### 随机配置生成

```python
from rdkit.Chem.Draw import rdMolDraw2D
import random

BACKGROUNDS = [
    (1.0, 1.0, 1.0, 0.0),   # 透明   15%
    (1.0, 1.0, 1.0, 1.0),   # 白色   40%
    (0.98, 0.96, 0.94, 1.0), # 米白   25%
    (0.94, 0.97, 1.0, 1.0), # 浅蓝   12%
    (0.95, 0.95, 0.95, 1.0), # 浅灰    8%
]

HIGHLIGHT_COLORS = [
    (1.0, 1.0, 0.0),   # 黄
    (0.5, 1.0, 0.5),   # 浅绿
    (1.0, 0.7, 0.2),   # 浅橙
]


def build_random_draw_options(rng: random.Random) -> rdMolDraw2D.MolDrawOptions:
    opts = rdMolDraw2D.MolDrawOptions()

    # 1. 背景
    bg = rng.choices(BACKGROUNDS, weights=[15, 40, 25, 12, 8])[0]
    opts.setBackgroundColour(bg)

    # 2. 分子主体颜色（60%灰 ~ 黑）
    gray = rng.uniform(0.0, 0.4)
    opts.setAtomPalette({k: (gray, gray, gray) for k in range(118)})

    # 3. ComicMode
    opts.comicMode = rng.random() < 0.30

    # 4. 其他固定选项保留
    opts.baseFontSize = 0.4
    opts.padding = 0.01
    opts.addStereoAnnotation = True
    opts.centreMoleculesBeforeDrawing = True
    opts.explicitMethyl = True
    opts.atomLabelDeuteriumTritium = True
    opts.simplifiedStereoGroupLabel = True

    return opts
```

### 高亮原子处理

高亮无法通过 `MolDrawOptions` 静态设置，需在 `Draw.MolToImage` 时传入 `highlightAtoms` 参数。

```python
def render_with_highlight(
    mol,
    target_size: tuple[int, int],
    opts: rdMolDraw2D.MolDrawOptions,
    rng: random.Random,
) -> Image.Image:
    highlight_atoms = None
    highlight_color = None

    if rng.random() < 0.30:
        # 选 1-3 个非氢原子
        heavy_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        if heavy_atoms:
            n = min(rng.randint(1, 3), len(heavy_atoms))
            highlight_atoms = rng.sample(heavy_atoms, n)
            highlight_color = rng.choice(HIGHLIGHT_COLORS)

    img = Draw.MolToImage(
        mol,
        size=target_size,
        options=opts,
        fitImage=True,
        highlightAtoms=highlight_atoms,
        highlightColor=highlight_color,
    )
    return trim(img)
```

### 接口变更

#### `rdkit_chem.py`

1. **新增** `build_random_draw_options(rng: random.Random) -> MolDrawOptions`
2. **新增** `render_mol_random(mol, size, rng) -> Image.Image | None` — 封装随机配置 + 高亮逻辑。可能返回 `None`（如 trim 失败或无效分子）
3. **保留** 现有 `molecule_to_img` / `d_opts` 供非随机场景使用

#### 各 Generator

`_render_structure()` 改为接收 `rng: random.Random`，内部调用 `render_mol_random`：

```python
def _render_structure(
    smiles: str,
    target_size: tuple[int, int],
    rng: random.Random,
) -> Image.Image | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        from prepare_traindata.rdkit_chem import render_mol_random
        img = render_mol_random(mol, target_size, rng)
    except Exception:
        return None
    ...
```

`rng` 由 `cfg.seed` 派生（如 `random.Random(cfg.seed + hash(smiles) % 1000)`），确保可复现。

### 修改范围

| 文件 | 动作 |
|------|------|
| `prepare_traindata/rdkit_chem.py` | 新增 `build_random_draw_options`、`render_mol_random` |
| `prepare_traindata/generate_dense_layout.py` | `_render_structure` 传入 `rng`，调用新接口 |
| `prepare_traindata/generate_table_layout.py` | 同上 |
| `prepare_traindata/generate_text_mix_layout.py` | 同上 |

### 回退兼容性

- 现有 `d_opts` 和 `molecule_to_img` **保持不变**
- 新接口为**新增**函数，不影响已有代码

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| `setAtomPalette` 对所有原子生效，可能导致高亮原子被覆盖 | 先设置主体色，高亮参数在 `MolToImage` 时传入，RDKit 会正确叠加 |
| 透明背景在白色画布上显示为白色，与白色背景难以区分 | 训练数据中两种都保留，模型需学习结构式本身而非背景差异 |
| ComicMode 下文字可读性变差 | 概率仅 30%，且训练目标为 bbox 检测而非 OCR，可读性非关键 |

## 验证计划

1. 生成 20 张测试图，目视检查：
   - 背景颜色是否按权重分布
   - 主体色是否为灰黑色而非纯黑
   - 高亮原子是否正确显示
   - ComicMode 是否生效
2. 检查 bbox 标注是否仍准确（结构式 trim 后尺寸变化需在预期内）

## 参考

- [RDKit Getting Started](https://www.rdkit.org/docs/GettingStartedInPython.html)
- [rdMolDraw2D API](https://www.rdkit.org/docs/source/rdkit.Chem.Draw.rdMolDraw2D.html)
