import paddlex as pdx

# 选择版面区域检测模块 (Layout Detection)
model = pdx.create_model(
    model_name="PP-DocLayoutV3",
    device="GPU:0",
)

# 开始微调
model.train(
    dataset_dir="./data/dense_chem/",
    epochs=50,  # 可根据实际情况调整
    batch_size=4,
    learning_rate=0.0001,
    save_dir="./output/ppdoclayoutv3_ft"
)