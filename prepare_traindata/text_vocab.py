"""Random vocabulary module for filling table cells with chemistry-document content."""

from __future__ import annotations

import random

CAS_SAMPLES: tuple[str, ...] = (
    "123-45-6",
    "CAS: 789-01-2",
    "CAS No. 111-11-1",
    "56-78-9",
    "CAS: 999-88-7",
    "100-00-5",
    "CAS 654-32-1",
    "321-54-8",
    "CAS: 777-66-5",
    "888-99-0",
    "CAS No. 222-33-4",
    "444-55-6",
    "CAS: 135-79-1",
    "246-80-3",
    "CAS 111-22-3",
    "333-44-5",
    "CAS: 666-77-8",
    "999-00-1",
    "CAS No. 112-23-4",
    "556-67-8",
)

PURITY_SAMPLES: tuple[str, ...] = (
    ">99%",
    "98.5%",
    "HPLC grade",
    ">99.5%",
    "98.0%",
    "ACS grade",
    ">99.9%",
    "97.5%",
    "99.0%",
    "Reagent grade",
)

MW_SAMPLES: tuple[str, ...] = (
    "MW: 234.5",
    "M=456.7 g/mol",
    "MW 128.3",
    "M=180.2 g/mol",
    "MW: 92.1",
    "M=60.1 g/mol",
    "MW 150.0",
    "M=300.5 g/mol",
    "MW: 78.1",
    "M=500.0 g/mol",
)

PRODUCT_NAMES: tuple[str, ...] = (
    "Benzene",
    "Acetone",
    "甲醇",
    "Ethyl acetate",
    "异丙醇",
    "Toluene",
    "乙醇",
    "Methanol",
    "乙酸乙酯",
    "Chloroform",
    "甲醛",
    "Hexane",
    "丙酮",
    "Dichloromethane",
    "苯",
    "Acetonitrile",
    "甲苯",
    "Dimethyl sulfoxide",
    "正己烷",
    "Tetrahydrofuran",
    "乙腈",
    "Diethyl ether",
    "二甲基亚砜",
    "Ethanol",
    "四氢呋喃",
    "Isopropanol",
    "乙醚",
    "Pyridine",
    "二氯甲烷",
    "Phenol",
)

SPECIFICATIONS: tuple[str, ...] = (
    "100mg",
    "1g",
    "500mL",
    "25g/bottle",
    "5g",
    "250mL",
    "10g/vial",
    "1kg",
    "50mL",
    "100g/bottle",
)

BATCH_NUMBERS: tuple[str, ...] = (
    "Lot: A12345",
    "Batch: 20250427",
    "Lot No. B67890",
    "Batch: 20260315",
    "Lot: C11111",
    "Batch: 20271201",
    "Lot No. D22222",
    "Batch: 20280120",
    "Lot: E33333",
    "Batch: 20290505",
)

APPEARANCE: tuple[str, ...] = (
    "White solid",
    "Colorless liquid",
    "淡黄色粉末",
    "Yellow crystals",
    "无色透明液体",
    "Off-white solid",
    "淡黄色液体",
    "Clear liquid",
    "白色结晶",
    "Brown powder",
)

NOTES: tuple[str, ...] = (
    "Store at -20°C",
    "Hygroscopic",
    "易燃",
    "Light sensitive",
    "Keep dry",
    "腐蚀性",
    "Air sensitive",
    "Store in dark",
    "易挥发",
    "Handle in fume hood",
)

ALL_SAMPLES: tuple[str, ...] = (
    *CAS_SAMPLES,
    *PURITY_SAMPLES,
    *MW_SAMPLES,
    *PRODUCT_NAMES,
    *SPECIFICATIONS,
    *BATCH_NUMBERS,
    *APPEARANCE,
    *NOTES,
)


def get_random_text(rng: random.Random) -> str:
    """Return a random chemistry-catalog-style string.

    With 80 % probability returns a single sample; with 20 % probability
    returns 2–3 samples joined by a single space.
    """
    if rng.random() < 0.8:
        return rng.choice(ALL_SAMPLES)

    count = rng.randint(2, 3)
    return " ".join(rng.choices(ALL_SAMPLES, k=count))
