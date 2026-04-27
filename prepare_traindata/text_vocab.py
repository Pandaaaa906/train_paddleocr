"""Random vocabulary module for filling table cells with chemistry-document content."""

import random

CAS_SAMPLES: list[str] = [
    "123-45-6",
    "7782-44-7",
    "64-17-5",
    "67-64-1",
    "7732-18-5",
    "7440-44-0",
    "1333-74-0",
    "630-08-0",
    "10024-97-2",
    "7446-09-5",
    "1310-73-2",
    "7647-14-5",
    "7757-82-6",
    "10102-44-0",
    "24634-61-5",
    "5329-14-6",
    "110-82-7",
    "108-88-3",
    "71-43-2",
    "50-00-0",
]

PURITY_SAMPLES: list[str] = [
    "≥98%",
    "99.5%",
    "ACS grade",
    "≥99.9%",
    "HPLC grade",
    "AR grade",
    "CP grade",
    "≥95%",
    "Ultra pure",
    "Pharmaceutical grade",
]

MW_SAMPLES: list[str] = [
    "MW: 180.16",
    "Mol.Wt. 342.30",
    "MW 58.08",
    "M.W. 60.10",
    "Mol.Wt: 150.22",
    "MW: 78.11",
    "M.W. 46.07",
    "MW 74.12",
    "Mol.Wt. 106.17",
    "MW: 132.16",
]

PRODUCT_NAMES: list[str] = [
    "Acetone",
    "乙醇",
    "Sodium Chloride",
    "苯甲酸",
    "Methanol",
    "硫酸",
    "Ethyl Acetate",
    "氢氧化钠",
    "Toluene",
    "硝酸",
    "Dimethyl Sulfoxide",
    "氯化钾",
    "Hexane",
    "磷酸",
    "Isopropanol",
    "碳酸钠",
    "Dichloromethane",
    "乙酸",
    "Tetrahydrofuran",
    "硫酸铜",
    "Chloroform",
    "丙酮",
    "Acetonitrile",
    "盐酸",
    "Diethyl Ether",
    "硝酸银",
    "Formic Acid",
    "氢氧化钾",
    "Petroleum Ether",
    "柠檬酸",
]

SPECIFICATIONS: list[str] = [
    "500g/bottle",
    "25kg/drum",
    "100mL",
    "1L/bottle",
    "5g/vial",
    "250mL",
    "10kg/box",
    "50mL",
    "2.5L",
    "20kg/barrel",
]

BATCH_NUMBERS: list[str] = [
    "Lot: 20240401A",
    "Batch# 230915",
    "Lot No. 20231012B",
    "Batch: 240301C",
    "Lot# 231125",
    "Batch 20240228D",
    "Lot: 20231201E",
    "Batch# 240515",
    "Lot No. 231008F",
    "Batch: 20240320G",
]

APPEARANCE: list[str] = [
    "White powder",
    "Colorless liquid",
    "淡黄色固体",
    "White crystalline",
    "无色透明液体",
    "Yellowish granules",
    "Clear liquid",
    "白色片状结晶",
    "Pale yellow oil",
    "无色至淡黄色液体",
]

NOTES: list[str] = [
    "Store at 4°C",
    "Hygroscopic",
    "避光保存",
    "Keep dry",
    "易燃液体",
    "Corrosive",
    "冷藏保存",
    "Oxidizer",
    "有毒，注意防护",
    "Air sensitive",
]

ALL_SAMPLES: list[str] = (
    CAS_SAMPLES
    + PURITY_SAMPLES
    + MW_SAMPLES
    + PRODUCT_NAMES
    + SPECIFICATIONS
    + BATCH_NUMBERS
    + APPEARANCE
    + NOTES
)


def get_random_text(rng: random.Random) -> str:
    """Return a random string suitable for filling a table cell.

    With 80% probability returns a single random sample.
    With 20% probability returns 2-3 samples joined by spaces.
    """
    if rng.random() < 0.2:
        count = rng.randint(2, 3)
        parts = [rng.choice(ALL_SAMPLES) for _ in range(count)]
        return " ".join(parts)
    return rng.choice(ALL_SAMPLES)
