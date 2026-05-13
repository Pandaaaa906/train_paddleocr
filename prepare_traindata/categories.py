"""PP-DocLayoutV3 category definitions.

All 25 layout classes used by the PP-DocLayoutV3 model.  Imported by data
generators and merge tools so the category list is defined in one place.
"""

from __future__ import annotations

from typing import Any

CATEGORIES: list[dict[str, Any]] = [
    {"id": 0, "name": "abstract"},
    {"id": 1, "name": "algorithm"},
    {"id": 2, "name": "aside_text"},
    {"id": 3, "name": "chart"},
    {"id": 4, "name": "content"},
    {"id": 5, "name": "display_formula"},
    {"id": 6, "name": "doc_title"},
    {"id": 7, "name": "figure_title"},
    {"id": 8, "name": "footer"},
    {"id": 9, "name": "footer_image"},
    {"id": 10, "name": "footnote"},
    {"id": 11, "name": "formula_number"},
    {"id": 12, "name": "header"},
    {"id": 13, "name": "header_image"},
    {"id": 14, "name": "image"},
    {"id": 15, "name": "inline_formula"},
    {"id": 16, "name": "number"},
    {"id": 17, "name": "paragraph_title"},
    {"id": 18, "name": "reference"},
    {"id": 19, "name": "reference_content"},
    {"id": 20, "name": "seal"},
    {"id": 21, "name": "table"},
    {"id": 22, "name": "text"},
    {"id": 23, "name": "vertical_text"},
    {"id": 24, "name": "vision_footnote"},
]

# Convenience lookups
CAT_ID_TABLE: int = 21
CAT_ID_IMAGE: int = 14
CAT_ID_TEXT: int = 22
