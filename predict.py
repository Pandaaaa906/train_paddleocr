from paddlex import create_model

# 1. 实例化模型，指定模型名称为 PP-DocLayoutV3
# 也可以配置具体骨干网络
model = create_model(
    model_name="PP-DocLayoutV3",
    model_dir="output/ppdoclayoutv3_ft/"
)

output = model.predict("data/dense_chem/images/dense_000008.png")
for res in output:
    res.print()            # Print result details
    res.save_to_json("output/res.json")  # Save as JSON


if __name__ == '__main__':
    pass
