"""Random watermark generators for synthetic training backgrounds.

Provides text, texture, and logo watermarks that can be composited onto
existing images to improve model robustness on watermarked documents.
"""

from __future__ import annotations

import math
import random
from typing import Callable

from PIL import Image, ImageDraw


def text_watermark(canvas_size: tuple[int, int], text: str, rng: random.Random) -> Image.Image:
    """Return an RGBA image with a tiled, tilted text watermark."""
    width, height = canvas_size
    base_color = rng.choice([
        (200, 200, 200),  # light gray
        (173, 216, 230),  # light blue
        (144, 238, 144),  # light green
    ])
    alpha = rng.randint(20, 60)
    color = (*base_color, alpha)

    angle = rng.randint(15, 45)

    # Create a small tile with the text on a transparent background
    tile_size = 120
    tile = Image.new("RGBA", (tile_size, tile_size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(tile)

    # Use default font; position roughly centered in tile
    bbox = draw.textbbox((0, 0), text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (tile_size - text_w) // 2
    y = (tile_size - text_h) // 2
    draw.text((x, y), text, fill=color)

    # Rotate the tile
    rotated = tile.rotate(angle, expand=True, resample=Image.BICUBIC)
    rw, rh = rotated.size

    # Tile across the canvas
    canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 0))
    for y_offset in range(0, height + rh, rh):
        for x_offset in range(0, width + rw, rw):
            canvas.paste(rotated, (x_offset, y_offset))

    return canvas


def _draw_grid(canvas_size: tuple[int, int], alpha: int) -> Image.Image:
    img = Image.new("RGBA", canvas_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    color = (180, 180, 180, alpha)
    spacing = 20
    w, h = canvas_size
    for x in range(0, w, spacing):
        draw.line([(x, 0), (x, h)], fill=color, width=1)
    for y in range(0, h, spacing):
        draw.line([(0, y), (w, y)], fill=color, width=1)
    return img


def _draw_dots(canvas_size: tuple[int, int], alpha: int) -> Image.Image:
    img = Image.new("RGBA", canvas_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    color = (180, 180, 180, alpha)
    spacing = 16
    radius = 1
    w, h = canvas_size
    for x in range(0, w, spacing):
        for y in range(0, h, spacing):
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color)
    return img


def _draw_diagonal_lines(canvas_size: tuple[int, int], alpha: int) -> Image.Image:
    img = Image.new("RGBA", canvas_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    color = (180, 180, 180, alpha)
    spacing = 24
    w, h = canvas_size
    # Draw diagonal lines from top-left to bottom-right
    # Start points along left and top edges
    for start in range(-h, w + h, spacing):
        x1, y1 = start, 0
        x2, y2 = start + h, h
        draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
    return img


def texture_watermark(canvas_size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Return an RGBA image with a geometric texture watermark."""
    alpha = rng.randint(15, 40)
    texture_fn: Callable[[tuple[int, int], int], Image.Image] = rng.choice([
        _draw_grid,
        _draw_dots,
        _draw_diagonal_lines,
    ])
    return texture_fn(canvas_size, alpha)


def _draw_hexagon_outline(canvas_size: tuple[int, int], alpha: int, rng: random.Random) -> Image.Image:
    img = Image.new("RGBA", canvas_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    color = (160, 160, 160, alpha)

    w, h = canvas_size
    hex_radius = 30
    spacing_x = int(hex_radius * 3)
    spacing_y = int(hex_radius * 2.6)

    def hex_points(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.radians(angle_deg)
            points.append((cx + r * math.cos(angle_rad), cy + r * math.sin(angle_rad)))
        return points

    for row in range(-1, h // spacing_y + 2):
        for col in range(-1, w // spacing_x + 2):
            cx = col * spacing_x + (row % 2) * (spacing_x / 2)
            cy = row * spacing_y
            points = hex_points(cx, cy, hex_radius)
            draw.polygon(points, outline=color)

    return img


def _draw_zigzag_wave(canvas_size: tuple[int, int], alpha: int, rng: random.Random) -> Image.Image:
    img = Image.new("RGBA", canvas_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    color = (160, 160, 160, alpha)

    w, h = canvas_size
    amplitude = 10
    period = 40
    spacing = 30

    for y_offset in range(-amplitude, h + amplitude, spacing):
        points: list[tuple[float, float]] = []
        x = 0
        while x <= w:
            y = y_offset + amplitude * math.sin(2 * math.pi * x / period)
            points.append((x, y))
            x += 2
        if len(points) > 1:
            draw.line(points, fill=color, width=1)

    return img


def logo_watermark(canvas_size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Return an RGBA image with a simple logo-like shape watermark."""
    alpha = rng.randint(20, 50)
    logo_fn: Callable[[tuple[int, int], int, random.Random], Image.Image] = rng.choice([
        _draw_hexagon_outline,
        _draw_zigzag_wave,
    ])
    return logo_fn(canvas_size, alpha, rng)


def apply_random_watermark(canvas: Image.Image, rng: random.Random) -> Image.Image:
    """Apply 0-3 random watermark layers to an RGB image and return the result in RGB."""
    if canvas.mode != "RGB":
        canvas = canvas.convert("RGB")

    num_layers = rng.randint(0, 3)
    if num_layers == 0:
        return canvas

    texts = ["CONFIDENTIAL", "SAMPLE", "DRAFT", "TEST", "INTERNAL"]

    # Convert to RGBA for compositing
    result = canvas.convert("RGBA")

    for _ in range(num_layers):
        watermark_type: Callable[..., Image.Image] = rng.choice([
            text_watermark,
            texture_watermark,
            logo_watermark,
        ])

        if watermark_type is text_watermark:
            text = rng.choice(texts)
            layer = text_watermark(canvas.size, text, rng)
        else:
            layer = watermark_type(canvas.size, rng)

        result = Image.alpha_composite(result, layer)

    # Convert back to RGB
    return result.convert("RGB")
