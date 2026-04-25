from io import BytesIO

from PIL import Image
from rdkit.Chem import Mol, Draw, rdDepictor, rdAbbreviations
from rdkit.Chem.Draw import rdMolDraw2D

from .image import trim

rdDepictor.SetPreferCoordGen(True)  # 应对读取大环内酯的Smiles，不会画成一个圈  TODO 有Bug有时导致（0xC0000005）

d_opts = rdMolDraw2D.MolDrawOptions()
d_opts.useBWAtomPalette()  # 不会高亮原子和键
d_opts.baseFontSize = 0.4
d_opts.padding = 0.01
d_opts.addStereoAnnotation = True  # 显示R/S, E/Z在手性中心，顺反双键旁  # TODO 大环内酯不生成S/R E/Z
d_opts.centreMoleculesBeforeDrawing = True
d_opts.explicitMethyl = True  # 碳链末端显示CH3, CH2
d_opts.atomLabelDeuteriumTritium = True  # True, 氘显示为D，否则为2H
d_opts.simplifiedStereoGroupLabel = True  # 不会有and1显示在手性中心旁边，改为显示”AND enantiomer"
d_opts.setBackgroundColour((1, 1, 1, 0))  # Transparent white background


def molecule_to_img(m: Mol, size=(800, 400), options=None, condense_abbrev: bool = True) -> Image.Image:
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


