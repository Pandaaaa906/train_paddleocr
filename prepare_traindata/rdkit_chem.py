from __future__ import annotations

import random
from io import BytesIO

from PIL import Image
from rdkit.Chem import Draw, Mol, rdAbbreviations, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from .image import trim

rdDepictor.SetPreferCoordGen(True)
# 应对读取大环内酯的Smiles，不会画成一个圈  TODO 有Bug有时导致（0xC0000005）

d_opts = rdMolDraw2D.MolDrawOptions()
d_opts.useBWAtomPalette()  # 不会高亮原子和键
d_opts.baseFontSize = 0.4
d_opts.padding = 0.01
d_opts.addStereoAnnotation = True
# 显示R/S, E/Z在手性中心，顺反双键旁  # TODO 大环内酯不生成S/R E/Z
d_opts.centreMoleculesBeforeDrawing = True
d_opts.explicitMethyl = True  # 碳链末端显示CH3, CH2
d_opts.atomLabelDeuteriumTritium = True  # True, 氘显示为D，否则为2H
d_opts.simplifiedStereoGroupLabel = True
# 不会有and1显示在手性中心旁边，改为显示"AND enantiomer"
d_opts.setBackgroundColour((1, 1, 1, 0))  # Transparent white background

BACKGROUNDS: list[tuple[float, float, float, float]] = [
    (1.0, 1.0, 1.0, 0.0),    # transparent 15%
    (1.0, 1.0, 1.0, 1.0),    # white      40%
    (0.98, 0.96, 0.94, 1.0),  # beige      25%
    (0.94, 0.97, 1.0, 1.0),   # light blue 12%
    (0.95, 0.95, 0.95, 1.0),  # light gray  8%
]

BACKGROUND_WEIGHTS: list[int] = [15, 40, 25, 12, 8]

HIGHLIGHT_COLORS: list[tuple[float, float, float]] = [
    (1.0, 1.0, 0.0),   # yellow
    (0.5, 1.0, 0.5),   # light green
    (1.0, 0.7, 0.2),   # light orange
]


def build_random_draw_options(rng: random.Random) -> rdMolDraw2D.MolDrawOptions:
    """Return a MolDrawOptions with randomized styling."""
    opts = rdMolDraw2D.MolDrawOptions()

    # 1. Background
    bg = rng.choices(BACKGROUNDS, weights=BACKGROUND_WEIGHTS)[0]
    opts.setBackgroundColour(bg)

    # 2. Atom colour (60% gray ~ black)
    gray = rng.uniform(0.0, 0.4)
    opts.setAtomPalette({k: (gray, gray, gray) for k in range(119)})

    # 3. ComicMode
    opts.comicMode = rng.random() < 0.30

    # 4. Fixed options
    opts.baseFontSize = 0.4
    opts.padding = 0.01
    opts.addStereoAnnotation = True
    opts.centreMoleculesBeforeDrawing = True
    opts.explicitMethyl = True
    opts.atomLabelDeuteriumTritium = True
    opts.simplifiedStereoGroupLabel = True

    return opts


def render_mol_random(
    mol: Mol,
    size: tuple[int, int],
    rng: random.Random,
) -> Image.Image | None:
    """Render a molecule with randomized styling. Returns None on failure."""
    opts = build_random_draw_options(rng)

    highlight_atoms = None
    highlight_color = None

    if rng.random() < 0.30:
        heavy_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        if heavy_atoms:
            n = min(rng.randint(1, 3), len(heavy_atoms))
            highlight_atoms = rng.sample(heavy_atoms, n)
            highlight_color = rng.choice(HIGHLIGHT_COLORS)

    try:
        img = Draw.MolToImage(
            mol,
            size=size,
            options=opts,
            fitImage=True,
            highlightAtoms=highlight_atoms,
            highlightColor=highlight_color,
        )
        img = trim(img)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Failed to render molecule")
        return None

    if img is None:
        return None

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


def molecule_to_img(
    m: Mol,
    size: tuple[int, int] = (800, 400),
    options: rdMolDraw2D.MolDrawOptions | None = None,
    condense_abbrev: bool = True,
) -> Image.Image:
    if options is None:
        options = d_opts
    if condense_abbrev:
        m = rdAbbreviations.CondenseAbbreviationSubstanceGroups(m)
    im = Draw.MolToImage(m, size=size, options=options, fitImage=True,)
    im = trim(im)
    return im


def molecule_to_png_bytes(m: Mol,  size=(800, 400), options=None) -> bytes:
    im = molecule_to_img(m, size=size, options=options)
    b = BytesIO()
    im.save(b, format='png')
    return b.getvalue()
