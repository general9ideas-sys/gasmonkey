"""Rebuild all logo assets from the new higher-quality source."""
from pathlib import Path
from PIL import Image, ImageFilter, ImageChops
import numpy as np
import cv2

ROOT = Path(r"C:\Users\Lenovo\Desktop\gasmonkey\images")
SRC = ROOT / "logo-original.png"

img = Image.open(SRC).convert("RGBA")
w, h = img.size
px = img.load()
print("source", w, h)


def strength_at(r, g, b):
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    white = max(0.0, min(1.0, (luma - 45) / 90.0))
    red_dom = r - max(g, b)
    red = 0.0
    if r > 80 and red_dom > 28:
        red = max(0.0, min(1.0, (red_dom - 12) / 60.0)) * max(
            0.0, min(1.0, (r - 50) / 80.0)
        )
    return white, red


def extract(y0, y1, mode="white", pad=14, scale=4):
    band = Image.new("RGBA", (w, y1 - y0), (0, 0, 0, 0))
    bp = band.load()
    for y in range(y0, y1):
        for x in range(w):
            r, g, b, a = px[x, y]
            white, red = strength_at(r, g, b)
            if mode == "white":
                s = white if white >= red else 0.0
                color = (255, 255, 255)
            else:
                s = red if red > white * 0.75 else 0.0
                color = (237, 47, 36)
            if s <= 0.03:
                continue
            bp[x, y - y0] = (*color, int(round(255 * min(1.0, s * 1.2))))
    bbox = band.getbbox()
    if not bbox:
        raise SystemExit(f"empty {mode} {y0}-{y1}")
    band = band.crop(bbox)
    out = Image.new("RGBA", (band.width + pad * 2, band.height + pad * 2), (0, 0, 0, 0))
    out.paste(band, (pad, pad), band)

    arr = np.array(out)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]
    size = (rgb.shape[1] * scale, rgb.shape[0] * scale)
    rgb_up = cv2.resize(rgb, size, interpolation=cv2.INTER_LANCZOS4)
    a_up = cv2.resize(alpha, size, interpolation=cv2.INTER_CUBIC)

    blur = cv2.GaussianBlur(rgb_up, (0, 0), 0.9)
    rgb_up = cv2.addWeighted(rgb_up, 1.45, blur, -0.45, 0)

    a = np.clip((a_up.astype(np.float32) - 18) * (255.0 / 200.0), 0, 255).astype(np.uint8)
    strong = a > 70
    if mode == "white":
        rgb_up[strong] = (255, 255, 255)
        a[strong] = 255
    else:
        red_dom = rgb_up[:, :, 0].astype(np.int16) - np.maximum(
            rgb_up[:, :, 1], rgb_up[:, :, 2]
        ).astype(np.int16)
        keep = strong & (red_dom > 20)
        rgb_up[keep] = (237, 47, 36)
        a[keep] = 255
        # clear non-red leftovers
        drop = strong & ~keep
        a[drop] = 0

    result = Image.fromarray(np.dstack([rgb_up, a]), "RGBA")
    result = result.filter(ImageFilter.UnsharpMask(radius=1.3, percent=140, threshold=2))
    return result


# Bands tuned for 1024x1024 HQ logo (solid black background)
car = extract(290, 540, mode="white", scale=3)
name = extract(540, 635, mode="white", scale=3)
tag = extract(630, 740, mode="red", scale=3)

car.save(ROOT / "intro-car.png", optimize=True)
name.save(ROOT / "intro-name.png", optimize=True)
tag.save(ROOT / "intro-tag.png", optimize=True)
print("car", car.size, "name", name.size, "tag", tag.size)

# Edge for light pass
alpha = car.getchannel("A")
edge_alpha = ImageChops.subtract(alpha, alpha.filter(ImageFilter.MinFilter(9)))
edge = Image.new("RGBA", car.size, (255, 255, 255, 0))
edge.putalpha(edge_alpha)
edge.save(ROOT / "intro-car-edge.png", optimize=True)

# Transparent full logo for header/footer
gap = 28
cw = max(car.width, name.width, tag.width) + 40
ch = car.height + name.height + tag.height + gap * 2 + 40
logo = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
y = 20
for piece in (car, name, tag):
    logo.paste(piece, ((cw - piece.width) // 2, y), piece)
    y += piece.height + gap
logo_h = logo.resize((1100, int(logo.height * 1100 / logo.width)), Image.Resampling.LANCZOS)
logo_h = logo_h.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))
logo_h.save(ROOT / "logo.png", optimize=True)
print("logo", logo_h.size)

# Also keep PNGs as primary for intro (more faithful than old messy SVGs)
print("done")
