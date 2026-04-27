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
│   ├── image.py                        # trim(im, margin=5) — 裁剪白边
│   ├── rdkit_chem.py                   # RDKit 渲染辅助（molecule_to_img）
│   ├── generate_synthetic_layout.py    # 【旧版】随机散落布局合成数据
│   ├── generate_dense_layout.py        # 【当前使用】密集垂直排列合成数据
│   ├── remap_and_split.py              # COCO → PaddleX 格式转换（category_id 映射 + 加 segmentation/read_order + train/val 拆分）
│   └── merge_datasets.py               # 多数据集合并（synthetic + dense → merged）
│
├── data/
│   ├── synthetic_chem/                 # 【旧版】随机布局合成数据 (~5000张)
│   ├── dense_chem/                     # 【当前训练数据】密集布局合成数据
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

### 4.3 合并多个数据集（可选）

```bash
uv run python -m prepare_traindata.merge_datasets
```

**输入**：`data/synthetic_chem/` + `data/dense_chem/`  
**输出**：`data/merged_chem/`（统一 ID、重新分配 read_order）

---

## 5. 训练配置与策略

### 5.1 当前有效配置（`configs/PP-DocLayoutV3.yaml`）

```yaml
Global:
  model: PP-DocLayoutV3
  mode: train
  dataset_dir: "data/dense_chem"
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

## 10. 参考链接

- [PaddleX Layout Detection 文档](https://paddlepaddle.github.io/PaddleX/3.0-rc/en/module_usage/tutorials/ocr_modules/layout_detection.html)
- [PaddlePaddle GPU 安装指南](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html)
- [PP-DocLayoutV3 官方 Demo 数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/doclayoutv3_examples.tar)
- [PP-DocLayout 论文](https://arxiv.org/html/2503.17213v1)
