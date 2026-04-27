from os import path
from pathlib import Path

from paddlex import create_model

fp = Path(r"C:\Users\Pandaaaa906\Pictures\客户询价-多个结构图片2.png")
# fp = Path(r"C:\Users\Pandaaaa906\Pictures\客户询价 - 带图片列表.png")
# fp = Path(r"C:\Users\Pandaaaa906\Pictures\20260416 客户询价 纯文字列表图片 识别出很多0.png")
# fp = Path(r"C:\Users\Pandaaaa906\Pictures\客户询价 - 列表截图 - 没法区分多个结构式.png")
# fp = Path(r"C:\Users\Pandaaaa906\Pictures\客户询价 - 多个结构图片.jpg")
out_base_dir = Path("./output/test")
fname, _ = path.splitext(fp.name)
out_dir = out_base_dir / fname
out_dir.mkdir(parents=True, exist_ok=True)


model_name = "PP-DocLayoutV3"
model = create_model(
    model_name=model_name,
    # model_dir="./output/ppdoclayoutv3_ft/0/inference/",
    device="CPU",
)
output = model.predict(str(fp), batch_size=1)

for res in output:
    res.save_to_img(save_path=str(out_dir))
    res.save_to_json(save_path=str(out_dir / "res.json"))


if __name__ == '__main__':
    pass