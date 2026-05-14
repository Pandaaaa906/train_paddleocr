# generate_text_mix_layout.py 设计规范

## 目标

新增 `prepare_traindata/generate_text_mix_layout.py`，生成 1000 张**图文混排**合成训练图，模拟真实化学文档（合成路线、杂质分析、询价单等），用于微调 PP-DocLayoutV3 同时检测 `text` (22) 和 `image` (14) 两类。

## 三种布局模式（每张图随机选择一种）

### 模式 A: Reaction Pathway（横向合成路线）

- 顶部：标题文本（1-2 行）
- 中部：横向 2-5 个结构式，用矢量箭头连接
  - 箭杆上方或下方可绘制反应条件文本（如 "·HCl", "NaOH, EtOH, reflux"）
- 底部：1-3 行备注/说明文本
- 所有文本行**独立标注**为 `text` 类，每行一个 bbox

### 模式 B: Vertical List（纵向列表）

- 顶部：标题文本
- 中部：1-3 列布局
  - 每列顶部为结构式
  - 结构式下方 2-4 行元数据文本（Formula / Mol.Wt. / CAS / Batch 等）
  - 每行元数据**独立标注**为 `text` 类
- 底部：可选备注文本

### 模式 C: Mixed Paragraph（混排段落）

- 上部：文本段落（2-4 行）
- 中部：单行 2-3 个结构式，或 2×2 网格
- 下部：文本段落（2-4 行）
- 所有文本行**独立标注**为 `text` 类

## 箭头绘制规范

独立函数 `_draw_arrow(draw, start, end, width, color)`：

- **箭杆**：`draw.line()` 直线连接 `start` → `end`
- **箭头**：`draw.polygon()` 三角形，顶点指向 `end`
- **尺寸自适应**：箭头大小与线宽成比例
- 支持水平箭头（合成路线）和可选的倾斜箭头

## 文本内容池

复用 `text_vocab.py`，新增专用文本池：

| 类别 | 示例 |
|------|------|
| 标题 | "盐酸米安色林前三步同分异构体杂质结构", "中间体合成路线", "客户询价单" |
| 反应条件 | "·HCl", "NaOH, EtOH, reflux", "rt, 2h", "Δ, 4h", "Pd/C, H₂" |
| 结构元数据 | "Formula: C₁₁H₁₇O₂N", "Mol.Wt.: 195.26", "CAS: 85141-93-1" |
| 底部备注 | "采购上述 3 个杂质，各买 100mg。", "请提供报价和交货期。" |
| 通用文本 | 复用 text_vocab 现有词库 |

## 标注规则

- `category_id`：`14=image`（结构式），`22=text`（文本行）
- `read_order`：按 top-to-bottom, left-to-right 排序
- `segmentation`：矩形多边形（从 bbox 合成，与现有 generator 一致）
- 关键约束：**每个文本行独立 bbox，不合并段落**——让模型学习逐行文本区域定位

## CLI 参数

```bash
uv run python -m prepare_traindata.generate_text_mix_layout \
  --num-samples 1000 \
  --output-dir data/text_mix_layout \
  --split 0.8 \
  --watermark \
  --workers 4
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-samples` | 1000 | 生成数量 |
| `--output-dir` | `data/text_mix_layout` | 输出目录 |
| `--split` | 0.8 | 训练/验证集拆分比例 |
| `--watermark` | True | 是否添加水印 |
| `--workers` | 0 | 并行进程数 |

## 代码规范

- 遵循 CLAUDE.md 第 10 节：
  - `frozen dataclass` 传递配置（Windows spawn 安全）
  - `random.Random(seed)` 局部 RNG
  - click CLI，复用 `cli.py` 共享选项
  - orjson 输出 JSON
  - 文件控制在 500 行以内
- 箭头绘制提取为独立函数 `_draw_arrow()`
- 结构式渲染复用现有 `_render_structure()` 模式

## 输出格式

PaddleX 扩展 COCO JSON：

```json
{
  "images": [...],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 14,
      "bbox": [x, y, w, h],
      "area": w * h,
      "segmentation": [[x, y, x+w, y, x+w, y+h, x, y+h]],
      "read_order": 0,
      "iscrowd": 0
    },
    {
      "id": 1,
      "image_id": 0,
      "category_id": 22,
      "bbox": [x, y, w, h],
      ...
    }
  ],
  "categories": CATEGORIES
}
```
