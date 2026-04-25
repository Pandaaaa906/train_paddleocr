from PIL import Image, ImageChops


def trim(im, margin=5):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if not bbox:
        return
    im = im.crop(bbox)
    w, h = im.size
    ret = Image.new(im.mode, (w + 2 * margin, h + 2 * margin), bg.getpixel((0, 0)))
    ret.paste(im, (margin, margin))
    return ret
