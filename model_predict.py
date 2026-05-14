from os import path
from pathlib import Path

from paddlex import create_model

out_base_dir = Path("./output/test")

fps = [
    # Path(r"C:/Users/Pandaaaa906/Pictures/供应商图谱样例 HPLC C4X-14356TMG-015003-2-COA-周-250902-5.png"),
    # Path(r"C:/Users/Pandaaaa906/Pictures/扬信 HNMR 水印 盖章.png"),
    # Path(r"C:/Users/Pandaaaa906/Pictures/20260506 供应商图谱 大赛路 氢谱.png"),

    Path("samples/images/20260514 客户询价 原word 内嵌表格.png"),
    Path("samples/images/20260506 客户询价 纯文字列表 截图.jpg"),
    Path("samples/images/20260507 客户询价 文献截图.png"),
    Path("samples/images/Fake 询价列表截图.png"),
    Path("samples/images/客户询价 - 带图片列表.png"),
    Path("samples/images/客户询价 - 多个结构图片.jpg"),
    Path("samples/images/客户询价 - 列表截图 - 没法区分多个结构式.png"),
    Path("samples/images/客户询价 - 列表截图.png"),
    Path("samples/images/客户询价.png"),
    Path("samples/images/客户询价-多个结构图片2.png"),
    Path("samples/images/客户询价例子 - 单行列表图片.png"),
    Path("samples/images/客户询价例子 - 单行列表图片2.png"),
    Path("samples/images/询价样例 文字 列表截图.png"),
    Path("samples/images/20260415 客户询价 带结构图列表图片.jpg"),
    Path("samples/images/20260416 客户询价 纯文字列表图片 识别出很多0.png"),
    Path("samples/images/20260421 客户询价 多个结构截图.png"),
    Path("samples/images/20260423 客户询价 纯文本列表 截图.png"),
    Path("samples/images/20260427 客户询价 带图片列表截图.png"),
    Path("samples/images/20260427 客户询价 带图片列表截图p2.png"),
    Path("samples/images/20260428 客户询价 带图列表截图.png"),
    Path("samples/images/20260430 客户需求 带结构图 word截图 非常规列表.jpg"),
    Path("samples/images/20260430 客户询价 带结构图 两列分栏 列表截图.png"),
    Path("samples/images/20260513 客户询价 多化合物 类合成路线.png"),
]


def model_predict(
    fps: list[Path | str],
    model_name: str = "PP-DocLayoutV3",
    model_dir: str | None = None,
    device: str | None = None,
):
    model = create_model(
        model_name=model_name,
        model_dir=model_dir,
        device=device,
    )
    output = model.predict([str(fp) for fp in fps], batch_size=1)

    for fp, res in zip(fps, output, strict=True):
        if isinstance(fp, str):
            fp = Path(fp)
        fname, _ = path.splitext(fp.name)
        out_dir = out_base_dir / fname
        out_dir.mkdir(parents=True, exist_ok=True)

        res.save_to_img(save_path=str(out_dir))
        res.save_to_json(save_path=str(out_dir / "res.json"))


if __name__ == "__main__":
    model_predict(
        fps,
        model_dir="./output/ppdoclayoutv3_ft/best_model/inference/",
        # model_dir="./output/ppdoclayoutv3_ft_20260507/best_model/inference/",
        device="CPU",
    )
