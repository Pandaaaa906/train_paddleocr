# train_paddleocr — PP-DocLayoutV3 化学结构式切分微调

## 1. 项目目标

解决 PP-DocLayoutV3（PaddleOCR-VL-1.5 的版面分析组件）在**一张图包含多个化学结构式时无法切分**的问题：

- **问题**：预训练模型将相邻化学结构式合并为一个大 `image` 区域
- **方案**：用合成化学结构式数据对 PP-DocLayoutV3 进行单类微调（`image` 类），**保留其余 24 类检测能力**
- **关键发现**：1 epoch + 低学习率 (`1e-5`) + backbone 学习率衰减 (`lr_mult_list`) 可在**不遗忘其他类别**的前提下提升 `image` 切分精度

> 验证结果：微调后 `image` 类 mAP@0.5:0.95 ≈ 0.964，且 `text`/`table`/`header` 等其他类别仍可正常检出。

---

## 2. 技术栈

| 组件 | 版本/说明 |
|------|----------|
| Python | 3.12 |
| 包管理 | `uv` |
| 深度学习框架 | `paddlepaddle-gpu` (CUDA 13) + `paddlex==3.5.0` |
| 化学渲染 | `rdkit` |
| 图像处理 | `Pillow` |
| 模型 | PP-DocLayoutV3 (DETR-based, 25-class layout detection) |
| 数据格式 | COCO JSON (PaddleX 扩展版，含 `segmentation` + `read_order`) |

---

## 3. 项目结构

```
.
├── smiles.txt                          # ~15K SMILES 源数据
├── pyproject.toml                      # uv 依赖管理
├── main.py                             # PaddleX 训练启动器 (train/eval/export)
├── model_predict.py                    # 推理测试脚本
├── check_env.py                        # Paddle 环境检查
│
├── configs/
│   └── PP-DocLayoutV3.yaml             # 训练配置（外层，供 main.py 读取）
│
├── pipelines/
│   └── pipeline_config_vllm.yaml       # PaddleOCR-VL-1.5 pipeline 部署配置
│
├── prepare_traindata/                  # 数据准备工具链
│   ├── __init__.py
│   ├── cli.py                          # click 共享 CLI 选项
│   ├── image.py                        # trim(im, margin=5) — 裁剪白边
│   ├── rdkit_chem.py                   # RDKit 渲染辅助（molecule_to_img）
│   ├── categories.py                   # PP-DocLayoutV3 25 类 CATEGORIES 定义
│   ├── text_vocab.py                   # 随机化学文本词库（中英混合）
│   ├── watermark_utils.py              # 水印生成器
│   ├── generate_synthetic_layout.py    # 【旧版】随机散落布局合成数据
│   ├── generate_dense_layout.py        # 【当前使用】密集垂直排列合成数据
│   ├── generate_table_layout.py        # 表格布局合成数据（结构式 + 文本混排）
│   ├── generate_text_mix_layout.py     # 图文混排合成数据（pathway / vertical / paragraph）
│   ├── remap_and_split.py              # COCO → PaddleX 格式转换（category_id 映射 + 加 segmentation/read_order + train/val 拆分）
│   └── merge_datasets.py               # 多数据集合并（synthetic + dense + text_mix → merged）
│
├── data/
│   ├── synthetic_chem/                 # 【旧版】随机布局合成数据 (~5000张)
│   ├── dense_layout/                     # 【当前训练数据】密集布局合成数据
│   │   ├── images/                     #   dense_000000.png ~ dense_004999.png
│   │   └── annotations/
│   │       ├── instance_train.json     #   90% 训练集（PaddleX COCO 格式，category_id=0）
│   │       └── instance_val.json       #   10% 验证集
│   └── merged_chem/                    # 合并数据集（如需要）
│
├── output/
│   ├── ppdoclayoutv3_ft/               # 训练输出
│   │   ├── train.log                   #   训练日志
│   │   ├── config.yaml                 #   PaddleX 展开后的完整配置
│   │   ├── train_result.json           #   训练结果摘要
│   │   ├── best_model/                 #   最佳模型 checkpoint
│   │   │   └── inference/              #   导出后的推理模型（.json + .pdiparams + .yml）
│   │   └── 0/                          #   epoch 0 checkpoint
│   └── test/                           # 推理测试结果
│
├── docs/
│   ├── superpowers/
│   │   ├── specs/                      # 设计规格文档
│   │   └── plans/                      # 实现计划文档
│   └── assets/                         # README 文档用图

└── doc/assets/                         # README 文档用图
```

---

## 4. 数据准备流程

### 4.1 生成合成训练数据

```bash
# 当前使用密集布局版本（2-10个结构式，1-2列垂直排列，更符合真实文档）
uv run python -m prepare_traindata.generate_dense_layout
```

**`generate_dense_layout.py` 行为：**
- 从 `smiles.txt` 读取并过滤有效 SMILES
- 每张图 2-10 个结构式，1 或 2 列垂直排列（带行间距 `ROW_GAP=5`、列间距 `COL_GAP=10`）
- 结构式尺寸：200×80 ~ 350×140
- 自动 trim 白边至 ~5px
- 输出 `instance_train.json`（原始 COCO，`category_id=0`）
- 多进程并行生成（`ProcessPoolExecutor`）

### 4.2 格式转换与拆分

```bash
# 将 COCO 转换为 PaddleX 要求的格式（加 segmentation + read_order）
uv run python -m prepare_traindata.remap_and_split
```

**`remap_and_split.py` 行为：**
- 将 `category_id` 从 14（原始）映射到 0（单类微调）
- 从 `bbox` 合成矩形 `segmentation`（PaddleX 实例分割要求）
- 按 top-to-bottom、left-to-right 排序生成 `read_order`
- 90/10 拆分为 `instance_train.json` / `instance_val.json`

### 4.3 表格布局数据（table + image 联合训练）

```bash
# 生成表格布局合成数据（结构式 + 文本混排）
uv run python -m prepare_traindata.generate_table_layout --workers 4
```

**`generate_table_layout.py` 行为：**
- 每张图包含一个随机行列数的表格（内层表格）
- 单元格内随机填充化学结构式（`image`, cat=14）或文本
- 表格整体标注为 `table`（cat=21）
- **5% 概率生成嵌套表格**（`nested=True`）：
  - 外层 1 列表格，1-2 个 header 单元格（随机化学文本）+ 1 个底部单元格
  - 外层边框 1px，内层表格边框 2-4px（形成视觉层次）
  - 外层表格和 header 文本**不生成 annotation**，仅内层表格和结构式正常标注
  - 内层 annotations 通过坐标偏移（`paste_x`, `paste_y`）映射到最终画布
- 支持 `--watermark/--no-watermark` 控制水印

### 4.4 图文混排数据（text + image 联合训练）

```bash
# 生成 1000 张图文混排合成图（3 种布局模式随机选择）
uv run python -m prepare_traindata.generate_text_mix_layout \
  --num-samples 1000 \
  --output-dir data/text_mix_layout \
  --min-structures 2 \
  --max-structures 5 \
  --min-cols 1 \
  --max-cols 3 \
  --watermark \
  --workers 4
```

**`generate_text_mix_layout.py` 行为：**
- 同时检测 `image` (14) 和 `text` (22) 两类
- 三种布局模式随机出现：
  - **pathway**：标题 + 横向结构式（2-5 个）+ 矢量箭头 + 反应条件标签 + 底部备注
  - **vertical**：标题 + 1-3 列（结构式 + 元数据文本）+ 底部备注
  - **paragraph**：顶部文本段落 + 结构式行/2×2 网格 + 底部文本段落
- 每个文本行**独立 bbox**，不合并段落
- 箭头标签仅在宽度小于箭头间隙时绘制，防止与结构式重叠
- 画布自动扩展防止内容溢出
- 精确的 text bbox：使用 `textbbox` 的 ink 偏移量，确保 bbox 完全包裹文字像素

### 4.4 合并多个数据集（可选）

```bash
uv run python -m prepare_traindata.merge_datasets \
  --datasets data/dense_layout data/table_layout data/text_mix_layout
```

**输入**：`data/dense_layout/` + `data/table_layout/` + `data/text_mix_layout/`  
**输出**：`data/merged_all/`（统一 ID、重新分配 read_order）

---

## 5. 训练配置与策略

### 5.1 当前有效配置（`configs/PP-DocLayoutV3.yaml`）

```yaml
Global:
  model: PP-DocLayoutV3
  mode: train
  dataset_dir: "data/dense_layout"
  device: gpu:0
  output: "output/ppdoclayoutv3_ft"

Train:
  num_classes: 25          # 保留25类，不改输出头
  epochs_iters: 1          # 外层配置（实际展开后 epoch: 2）
  batch_size: 4
  learning_rate: 0.00001   # 1e-5，保护预训练权重
  freeze_at: 3             # 不确定是否生效（PPHGNetV2 内部 freeze_at: 0）
  pretrain_weight_path: https://.../PP-DocLayoutV3_pretrained.pdparams
```

### 5.2 关键防遗忘策略（验证有效）

展开后的实际配置（`output/ppdoclayoutv3_ft/config.yaml`）中，**真正生效的防遗忘机制**：

```yaml
PPHGNetV2:
  freeze_at: 0
  freeze_norm: true
  freeze_stem_only: true
  lr_mult_list: [0.0, 0.05, 0.05, 0.05, 0.05]   # backbone 各 stage 学习率倍数
```

| 机制 | 作用 |
|------|------|
| `lr_mult_list[0] = 0.0` | backbone stage 0 不更新（最底层特征保护） |
| `lr_mult_list[1:4] = 0.05` | 其余 stage 以 5% 学习率微调 |
| `base_lr = 1e-5` | 整体学习率极低 |
| `epoch = 2` | 仅 2 个 epoch |
| `num_classes = 25` | 不改检测头结构，保留全部 25 类输出 |

> **用户实测结论**：该配置下训练后，其他 24 类检测能力仍然保留，`image` 类切分精度显著提升。

### 5.3 训练命令

```powershell
# Windows 环境变量（如缺失 paddle 动态库）
$env:PATH = ";J:\ProjectFiles\train_paddleocr\.venv\Lib\site-packages\paddle\include\paddle\phi\backends\dynload;$env:PATH"
$env:PATH = ";J:\ProjectFiles\train_paddleocr\.venv\Lib\site-packages\nvidia\cu13\bin\x86_64;$env:PATH"

# 训练
uv run main.py --mode train --device gpu:0

# 从 checkpoint 恢复
uv run main.py --mode train --device gpu:0 --resume output/ppdoclayoutv3_ft/epoch_10

# 导出推理模型
uv run main.py --mode export

# 数据集校验
uv run main.py --mode check_dataset
```

---

## 6. 推理与验证

### 6.1 单独推理测试

```bash
uv run python model_predict.py
```

**`model_predict.py` 行为：**
- 加载微调后的推理模型：`output/ppdoclayoutv3_ft/0/inference/`
- 对指定测试图进行版面分析
- 输出可视化结果到 `output/test/<filename>/`
- 保存 JSON 标注到 `output/test/<filename>/res.json`

### 6.2 评估指标（训练日志）

```
Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.964
Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.989
Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.979
Average Recall    (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.977
```

---

## 7. 部署到 PaddleOCR-VL-1.5

### 7.1 导出模型文件

训练完成后，推理模型位于：
```
output/ppdoclayoutv3_ft/best_model/inference/
├── inference.json
├── inference.pdiparams
└── inference.yml
```

### 7.2 修改 Pipeline 配置

将上述三个文件复制到部署服务器，修改 `pipeline_config_vllm.yaml`：

```yaml
SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayoutV3
    # model_dir: null                          # 使用官方默认模型
    model_dir: /home/paddleocr/models/PP-DocLayoutV3-ft/   # 使用微调模型
    batch_size: 8
    threshold: 0.3
    layout_nms: True
    layout_unclip_ratio: [1.0, 1.0]
    layout_merge_bboxes_mode:
      14: "union"    # image 类（根据部署场景可改为 large 或 small）
```

### 7.3 已知部署问题

- **RuntimeError: (PreconditionNotMet)**：检查 `paddlepaddle` 版本与 GPU/CUDA 版本匹配
- **缺少 repos 文件夹**：`mkdir .venv/Lib/site-packages/paddlex/repo_manager/repos`
- **缺少配置文件**：从 PaddleDetection 复制 `PP-DocLayoutV3.yaml` 到对应路径

---

## 8. 已知问题与 TODO

### 8.1 当前问题

| 问题 | 状态 | 说明 |
|------|------|------|
| 微调后某些场景效果变差 | 🔴 TODO | `doc/assets/fail_after_finetune.png` — 部分复杂布局下微调模型可能过度切分 |
| `freeze_at` 参数冲突 | 🟡 观察 | 外层 YAML 设置 `freeze_at: 3`，但展开后 `PPHGNetV2.freeze_at: 0`，以展开配置为准 |
| `epochs_iters: 1` vs `epoch: 2` | 🟡 观察 | 外层配置为 1，展开后为 2，实际跑了 2 epoch |
| 其他类别边缘检测精度 | 🟡 观察 | 低频类（`seal`, `vertical_text`, `algorithm`）可能轻微下降 |
| 中文/特殊字符渲染 | 🟢 已解决 | `generate_table_layout.py` 和 `generate_text_mix_layout.py` 已加载系统 CJK 字体 |
| 单元格文本自动换行 | 🟢 已解决 | `generate_table_layout.py` 已支持自动换行 + 字体大小回退 |
| 水印与单元格背景冲突 | 🟢 已解决 | 仅在 `--no-watermark` 时绘制白色单元格背景 |
| 图文混排训练数据 | 🟢 已完成 | `generate_text_mix_layout.py` 生成 1000 张，同时标注 `image`(14) + `text`(22) |
| 表格布局嵌套支持 | 🟢 已完成 | `generate_table_layout.py` 支持 5% 概率生成外层 1 列表格包裹内层表格 |
| RDKit 渲染多样性 | 🟢 已完成 | 支持随机背景色、灰度原子、comicMode、原子高亮等多样化渲染 |

### 8.2 后续优化方向

1. **数据增强**：加入真实化学文档（如论文截图、专利 PDF）进行混合训练
2. **类别平衡**：如其他类精度下降明显，考虑冻结 backbone + 只训练 head 的分类层
3. **后处理优化**：调整 `layout_merge_bboxes_mode` 中 `image` 类的合并策略（`union` → `large`/`small`）
4. **量化部署**：导出 INT8 模型以提升推理速度

---

## 9. PP-DocLayoutV3 25 类标签映射

| ID | 类别名 | 说明 | ID | 类别名 | 说明 |
|:--:|:-------|:-----|:--:|:-------|:-----|
| 0 | `abstract` | 摘要 | 13 | `header_image` | 页眉图片 |
| 1 | `algorithm` | 算法框 | 14 | `image` | 图片/化学结构式 |
| 2 | `aside_text` | 侧栏文本 | 15 | `inline_formula` | 行内公式 |
| 3 | `chart` | 统计图表 | 16 | `number` | 页码 |
| 4 | `content` | 正文内容 | 17 | `paragraph_title` | 段落标题 |
| 5 | `display_formula` | 行间公式 | 18 | `reference` | 参考文献引用 |
| 6 | `doc_title` | 文档主标题 | 19 | `reference_content` | 参考文献内容 |
| 7 | `figure_title` | 图标题 | 20 | `seal` | 印章 |
| 8 | `footer` | 页脚 | 21 | `table` | 表格 |
| 9 | `footer_image` | 页脚图片 | 22 | `text` | 文本块 |
| 10 | `footnote` | 脚注 | 23 | `vertical_text` | 竖排文本 |
| 11 | `formula_number` | 公式编号 | 24 | `vision_footnote` | 可视化脚注 |
| 12 | `header` | 页眉 | | | |

> 微调时数据集中的 `category_id` 被映射为 `0`，但模型输出头仍保留 25 类，因此推理时仍需按上表解析。

---

## 10. 代码规范

### 10.1 通用约定

| 项目 | 约定 |
|------|------|
| Python 版本 | >= 3.12 |
| 类型注解 | 全部函数签名必须带类型注解，使用 `\|None` 而非 `Optional` |
| 文件头 | 每个文件以 `from __future__ import annotations` 开头 |
| 导入顺序 | stdlib -> 第三方库 -> 本地模块（按 ruff/isort 规则） |
| 行宽 | 88 字符（black/ruff 默认） |
| 路径 | 统一使用 `pathlib.Path`，避免字符串路径 |
| JSON I/O | 性能敏感场景用 `orjson` 替代标准库 `json` |
| CLI | 使用 `click`，共享选项放在 `prepare_traindata/cli.py` |

### 10.2 命名规范

| 类型 | 风格 | 示例 |
|------|------|------|
| 模块常量 | `UPPER_CASE` | `CANVAS_MARGIN: int = 15` |
| 函数/变量 | `snake_case` | `build_configs()`, `load_valid_smiles()` |
| 类 | `PascalCase` | `SampleConfig`, `StructurePlacement` |
| 配置类 | `@dataclass(frozen=True)` | 所有 worker 间传递的配置 |
| 私有函数 | `_leading_underscore` | `_render_structure()`, `_load_font()` |

### 10.3 数据生成器开发规范

1. **Worker 安全（Windows `spawn` 关键）**
   - 所有进程间传递的配置必须使用 `dataclass(frozen=True)`
   - 禁止在 worker 函数中访问模块级全局变量
   - 通过 `ProcessPoolExecutor(initializer=..., initargs=...)` 传递 `SMILES` 池等共享状态
   - 示例：`generate_dense_layout.py:150-157`

2. **输出格式**
   - 直接使用 PaddleX 扩展 COCO 格式：
     ```json
     {
       "images": [...],
       "annotations": [...],
       "categories": CATEGORIES  // 25类完整列表
     }
     ```
   - `category_id` 必须使用 `CAT_ID_IMAGE` (14) 或 `CAT_ID_TABLE` (21)，禁止硬编码魔术数字
   - 每个 annotation 必须包含：`bbox`, `area`, `segmentation`（矩形多边形）, `read_order`, `iscrowd`

3. **随机性与可复现**
   - 使用 `random.Random(seed)` 创建局部 RNG，不触碰全局 `random`
   - `seed` 通过 CLI `--seed` 参数传入，默认 `42`

4. **共享模块**
   - `prepare_traindata/categories.py` — 25 类 CATEGORIES 和 CAT_ID_* 常量
   - `prepare_traindata/cli.py` — click 共享选项（如 `output_dir`, `num_samples`, `split`）
   - `prepare_traindata/watermark_utils.py` — 水印生成器
   - `prepare_traindata/text_vocab.py` — 随机化学文本词库
   - `prepare_traindata/rdkit_chem.py` — RDKit 渲染辅助，含 `render_mol_random` 多样化渲染

5. **RDKit 结构式渲染多样性**
   - 所有生成器使用 `render_mol_random()` 替代固定样式渲染
   - 随机维度：背景色（透明/白/米色/浅蓝/浅灰）、原子灰度（0.0-0.4）、comicMode（30%）、原子高亮（30%，1-3 个原子，黄/绿/橙）
   - 每个结构式使用独立 RNG：`random.Random(cfg.seed + idx + hash(smiles))`，保证渲染失败不影响后续结构式的确定性
   - 输出 RGBA，背景透明便于合成时与画布融合

6. **文件组织**
   - 每个 generator 独立文件：`generate_dense_layout.py`, `generate_table_layout.py`, `generate_text_mix_layout.py`, `generate_synthetic_layout.py`
   - 每文件 < 600 行；若膨胀则提取 helper 到新模块
   - 常量放在模块顶部 `CONSTANTS` 区块
   - `load_valid_smiles()` 等通用 helper 可复用，但每个文件保留独立 copy 以避免跨模块导入在 worker spawn 中失效

### 10.4 工具链

```bash
# 格式 + lint
ruff check prepare_traindata/ --select E,W,I,UP,B,A,C4,ICN,PIE
ruff check --fix .

# 运行 generator
uv run python -m prepare_traindata.generate_dense_layout --help
uv run python -m prepare_traindata.generate_table_layout --num-samples 10 --watermark
uv run python -m prepare_traindata.generate_text_mix_layout --num-samples 1000 --workers 4

# 合并数据集
uv run python -m prepare_traindata.merge_datasets \
  --datasets data/dense_layout --datasets data/table_layout --datasets data/text_mix_layout \
  --output-dir data/merged_all

# 数据集校验
uv run python main.py --mode check_dataset
```

---

## 11. 参考链接

- [PaddleX Layout Detection 文档](https://paddlepaddle.github.io/PaddleX/3.0-rc/en/module_usage/tutorials/ocr_modules/layout_detection.html)
- [PaddlePaddle GPU 安装指南](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html)
- [PP-DocLayoutV3 官方 Demo 数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/doclayoutv3_examples.tar)
- [PP-DocLayout 论文](https://arxiv.org/html/2503.17213v1)
