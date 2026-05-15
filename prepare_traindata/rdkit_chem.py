from __future__ import annotations

import random
from io import BytesIO

from PIL import Image, ImageEnhance
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
    (255, 255, 255, 0),    # transparent 15%
    (255, 255, 255, 255),    # white      40%
    (249, 244, 239, 255),  # beige      25%
    (239, 247, 255, 255),   # light blue 12%
    (242, 242, 242, 255),  # light gray  8%
]

BACKGROUND_WEIGHTS: list[int] = [15, 40, 25, 12, 8]

HIGHLIGHT_COLORS: list[tuple[float, float, float]] = [
    (1.0, 1.0, 0.0),   # yellow
    (0.5, 1.0, 0.5),   # light green
    (1.0, 0.7, 0.2),   # light orange
]


def build_random_draw_options(opts: rdMolDraw2D.MolDrawOptions, rng: random.Random) -> rdMolDraw2D.MolDrawOptions:
    """Return a MolDrawOptions with randomized styling."""
    opts.clearBackground = False

    # 2. Atom colour (60% gray ~ black)
    gray = rng.uniform(0.0, 0.4)
    opts.setAtomPalette({k: (gray, gray, gray) for k in range(119)})

    # 3. ComicMode
    opts.comicMode = rng.random() < 0.30

    # 4. Fixed options
    opts.baseFontSize = rng.uniform(0.4, 0.8)
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
    d2d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = d2d.drawOptions()
    build_random_draw_options(opts, rng)

    highlight_atoms = None
    highlight_color = None

    if rng.random() < 0.30:
        heavy_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        if heavy_atoms:
            n = min(rng.randint(1, 3), len(heavy_atoms))
            highlight_atoms = rng.sample(heavy_atoms, n)
            highlight_color = { atom: rng.choice(HIGHLIGHT_COLORS) for atom in highlight_atoms }
            pass

    d2d.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_color,
    )
    d2d.FinishDrawing()
    png_data = d2d.GetDrawingText()
    img = Image.open(BytesIO(png_data)).convert("RGBA")
    img = trim(img)

    if img is None:
        return None

    # 随机选择背景颜色
    bg_color = rng.choices(BACKGROUNDS, weights=BACKGROUND_WEIGHTS)[0]
    bg = Image.new("RGBA", img.size, bg_color)
    bg.paste(img, mask=img.split()[3])
    img = bg

    # 增加随机灰度、模仿过曝
    if rng.random() < 0.1:
        # 1. Reduce color saturation to make it "pale"
        pale_img = ImageEnhance.Color(img).enhance(rng.uniform(0.6, 0.9))
        # 2. Reduce contrast to turn black into grey
        grey_img = ImageEnhance.Contrast(pale_img).enhance(rng.uniform(0.6, 0.9))
        # 3. Increase brightness slightly if needed
        img = ImageEnhance.Brightness(grey_img).enhance(rng.uniform(1.1, 1.2))

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
