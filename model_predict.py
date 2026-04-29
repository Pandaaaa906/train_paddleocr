from os import path
from pathlib import Path
from typing import List

from paddlex import create_model


out_base_dir = Path("./output/test")

fps = [
    Path(r"C:\Users\Pandaaaa906\Pictures\20260416 客户询价 纯文字列表图片 识别出很多0.png"),
    Path(r"C:\Users\Pandaaaa906\Pictures\20260427 客户询价带图片列表截图.png"),  # 分子式分子量单元格部分识别成图片，实际用户docx样例没这个情况
    Path(r"C:\Users\Pandaaaa906\Pictures\20260427 客户询价带图片列表截图p2.png"),
    Path(r"C:\Users\Pandaaaa906\Pictures\客户询价 - 列表截图 - 没法区分多个结构式.png"),
    Path(r"C:\Users\Pandaaaa906\Pictures\客户询价 - 多个结构图片.jpg"),
    Path(r"C:\Users\Pandaaaa906\Pictures\客户询价 - 带图片列表.png"),
    Path(r"C:\Users\Pandaaaa906\Pictures\客户询价-多个结构图片2.png"),
    Path(r"C:\Users\Pandaaaa906\Pictures\客户询价 - 带图列表截图 20260428.png"),
]


def model_predict(
        fps: List[Path | str],
        model_name: str = "PP-DocLayoutV3",
        model_dir: str = None,
        device: str = None,
):
    model = create_model(
        model_name=model_name,
        model_dir=model_dir,
        device=device,
    )
    output = model.predict([str(fp) for fp in fps], batch_size=1)

    for fp, res in zip(fps, output):
        if isinstance(fp, str):
            fp = Path(fp)
        fname, _ = path.splitext(fp.name)
        out_dir = out_base_dir / fname
        out_dir.mkdir(parents=True, exist_ok=True)

        res.save_to_img(save_path=str(out_dir))
        res.save_to_json(save_path=str(out_dir / "res.json"))


if __name__ == '__main__':
    model_predict(
        fps,
        model_dir="./output/ppdoclayoutv3_ft/best_model/inference/",
        device="CPU",
    )
    pass