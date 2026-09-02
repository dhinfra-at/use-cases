#!/usr/bin/env python3
"""
make_synth.py — generate synthetic Latin stone-inscription training data.

Renders carved-look inscription images + matching PAGE-XML ground truth from a
corpus of real Latin inscription texts. Everything it produces is freely
licensable: the texts are EDH transcriptions (CC BY-SA 4.0) and the images are
procedurally rendered, so no inscription photographs are involved.

    python make_synth.py --n 20 --outdir data/synth --seed 42             # middot (·) between words
    python make_synth.py --n 20 --outdir data/synth --seed 42 --no-dots   # scriptio continua

Only needs Pillow + numpy. All bundled OFL fonts in ./fonts are used (one per
image, seeded). Stone backgrounds come from real texture patches in ./stone_images
if present, otherwise procedural stone; either way it works offline.

Pipeline per image:  text -> layout -> Capitalis glyphs -> weathered carved
relief (erosion, crossbar loss, directional lighting, cracks, pits) on stone
-> camera (affine warp, vignette, exposure/B&W fade, blur, JPEG) -> image + PAGE-XML.
The line geometry is placed by the generator and transformed through the same
affine map as the pixels, so the PAGE baselines and polygons stay exact.
"""
from __future__ import annotations
import argparse, math, os, random, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent

# --- built-in fallback corpus (used only if the corpus file is missing) ---
CORPUS = [
    ["DIS MANIBVS", "BABRIAE EVTERPE", "FECIT BABRIA"],
    ["IMP CAESAR DIVI F", "AVGVSTVS PONTIFEX", "MAXIMVS COS XIII"],
    ["L CORNELIVS L F", "SCIPIO BARBATVS", "CONSOL CENSOR"],
    ["D M", "AVRELIVS VICTOR", "VIXIT ANNIS XXXV", "MENSIBVS VII"],
    ["GENIO LOCI", "SACRVM"],
    ["TI CLAVDIVS CAESAR", "AVG GERMANICVS", "PONT MAX TRIB POT"],
    ["VIBIA SABINA", "AVGVSTA", "HADRIANI AVG"],
    ["MARCVS AGRIPPA", "L F COS TERTIVM", "FECIT"],
    ["DEO INVICTO", "MITHRAE", "PRO SALVTE"],
    ["Q LOLLIVS Q F", "URBICVS LEG AVG", "PR PR"],
]

FONT_URL = "https://github.com/google/fonts/raw/main/ofl/cinzel/Cinzel%5Bwght%5D.ttf"


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
def get_font_paths() -> list[str | None]:
    """All bundled OFL fonts in ./fonts (offline-safe). If none are bundled,
    download Cinzel once and cache it; if that fails, [None] -> Pillow default."""
    fonts = sorted(str(p) for p in (HERE / "fonts").glob("*.ttf"))
    if fonts:
        return fonts
    cache = Path.home() / ".cache" / "make_synth_cinzel.ttf"
    if not cache.exists():
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(FONT_URL, cache)
        except Exception as e:
            print(f"  (font download failed: {e} -> using Pillow default font)")
            return [None]
    return [str(cache)]


# ---------------------------------------------------------------------------
# Stone: real texture patches (./stone_images) or procedural
# ---------------------------------------------------------------------------
def fractal_noise(h, w, rng: np.random.Generator, octaves=5, persistence=0.55):
    """Fractal value noise in [0,1] from summed upsampled random grids."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh = max(2, h // (2 ** (octaves - o)))
        gw = max(2, w // (2 ** (octaves - o)))
        grid = rng.random((gh, gw)).astype(np.float32)
        img = Image.fromarray((grid * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        out += amp * (np.asarray(img, np.float32) / 255.0)
        total += amp
        amp *= persistence
    return out / total


STONE_PALETTES = {
    "limestone": (201, 193, 172),
    "marble":    (216, 213, 207),
    "sandstone": (188, 158, 118),
}


def make_stone(w, h, kind, nrng: np.random.Generator) -> np.ndarray:
    """Procedural stone texture as float RGB (0..255)."""
    base = np.array(STONE_PALETTES[kind], np.float32)
    n = fractal_noise(h, w, nrng, octaves=6)
    grain = fractal_noise(h, w, nrng, octaves=3)
    tex = np.ones((h, w, 3), np.float32) * base[None, None, :]
    tex += (n[..., None] - 0.5) * 38.0
    tex += (grain[..., None] - 0.5) * 14.0
    if kind == "marble":
        xx = np.linspace(0, 1, w)[None, :].repeat(h, 0)
        turb = fractal_noise(h, w, nrng, octaves=5)
        veins = np.abs(np.sin((xx * nrng.uniform(2, 5) + turb * nrng.uniform(2.5, 5.0)) * math.pi))
        tex -= ((1.0 - veins) ** 6)[..., None] * nrng.uniform(25, 60)
    elif kind == "sandstone":
        yy = np.linspace(0, 1, h)[:, None].repeat(w, 1)
        band = np.sin(yy * nrng.uniform(20, 50) * math.pi + fractal_noise(h, w, nrng, 3) * 4)
        tex += band[..., None] * 6.0
    gx, gy = nrng.uniform(-1, 1), nrng.uniform(-1, 1)
    xx = np.linspace(-0.5, 0.5, w)[None, :]
    yy = np.linspace(-0.5, 0.5, h)[:, None]
    tex += ((gx * xx + gy * yy) * nrng.uniform(10, 45))[..., None]
    return np.clip(tex, 0, 255)


_STONE_IMGS: list[Path] | None = None


def load_stone_image(W, H, rng: random.Random) -> np.ndarray | None:
    """A real stone patch from ./stone_images, flipped/rotated and mirror-tiled to
    WxH. Returns None if the folder is empty (-> procedural stone)."""
    global _STONE_IMGS
    if _STONE_IMGS is None:
        d = HERE / "stone_images"
        _STONE_IMGS = sorted(p for p in d.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if d.exists() else []
    if not _STONE_IMGS:
        return None
    img = Image.open(rng.choice(_STONE_IMGS)).convert("RGB")
    a = np.asarray(img.convert("L"), np.float32)
    if a.mean() < 115:                              # lift dark stone so text stays legible
        img = Image.fromarray(np.clip(np.asarray(img, np.float32) * min(1.8, 130.0 / max(a.mean(), 1)), 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    rot = rng.choice([0, 90, 180, 270])
    if rot:
        img = img.rotate(rot, expand=True)
    scale = max(rng.uniform(0.9, 1.4), 0.65 * W / img.width, 0.65 * H / img.height)
    tw, th = max(64, int(img.width * scale)), max(64, int(img.height * scale))
    tile = img.resize((tw, th), Image.LANCZOS)
    tile_fl = tile.transpose(Image.FLIP_LEFT_RIGHT)
    canvas = Image.new("RGB", (W, H))
    y = row = 0
    while y < H:
        x = col = 0
        while x < W:
            t = tile if (row + col) % 2 == 0 else tile_fl
            if row % 2 == 1:
                t = t.transpose(Image.FLIP_TOP_BOTTOM)
            canvas.paste(t, (x, y))
            x += tw; col += 1
        y += th; row += 1
    return np.asarray(canvas, np.float32)


# ---------------------------------------------------------------------------
# Layout: render text to a mask + exact per-line geometry
# ---------------------------------------------------------------------------
def layout(lines, W, H, font_path, rng: random.Random):
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    top, bot = int(H * 0.10), int(H * 0.10)
    line_box = (H - top - bot) // len(lines)
    geoms, y = [], top
    for ln in lines:
        px = max(10, int(line_box * 0.72))
        while px > 10:
            f = ImageFont.truetype(font_path, px) if font_path else ImageFont.load_default()
            l, t, r, b = d.textbbox((0, 0), ln, font=f)
            if (r - l) <= 0.90 * W or font_path is None:
                break
            px = int(px * 0.92)
        f = ImageFont.truetype(font_path, px) if font_path else ImageFont.load_default()
        l, t, r, b = d.textbbox((0, 0), ln, font=f)
        tw, th = r - l, b - t
        x = (W - tw) // 2 - l
        yy = y + (line_box - th) // 2 - t
        d.text((x, yy), ln, fill=255, font=f)
        asc = f.getmetrics()[0] if font_path else th
        x0, x1 = (W - tw) // 2, (W - tw) // 2 + tw
        yt, yb = y + (line_box - th) // 2, y + (line_box - th) // 2 + th
        base_y = yy + asc
        geoms.append({
            "text": ln,
            "baseline": [(x0, base_y), (x1, base_y)],
            "poly": [(x0 - 5, yt - 5), (x1 + 5, yt - 5), (x1 + 5, yb + 5), (x0 - 5, yb + 5)],
        })
        y += line_box
    return np.asarray(mask, np.float32) / 255.0, geoms, float(line_box)


# ---------------------------------------------------------------------------
# Weathered carve
# ---------------------------------------------------------------------------
def _erode_line(mb, r, axis):
    """Binary erosion by a line structuring element (length 2r+1) along axis."""
    out = mb.copy()
    for s in range(1, r + 1):
        out &= np.roll(mb, s, axis) & np.roll(mb, -s, axis)
    return out


def _dilate_line(mb, r, axis):
    out = mb.copy()
    for s in range(1, r + 1):
        out |= np.roll(mb, s, axis) | np.roll(mb, -s, axis)
    return out


def structural_erosion(mask, font_px, nrng: np.random.Generator):
    """Preferentially erode thin HORIZONTAL strokes (crossbars/arms of T E F L)
    while sparing vertical stems, so a worn letter drifts toward an I-stub with
    the GT label kept. Pure-numpy morphology (no scipy). Returns a keep-map."""
    mb = mask > 0.4
    if not mb.any():
        return np.ones_like(mask)
    rv = max(2, int(font_px * 0.20))                # tall SE half-length: stems
    rh = max(2, int(font_px * 0.11))                # wide SE half-length: bars
    stems = _dilate_line(_erode_line(mb, rv, 0), rv, 0)
    bars = _dilate_line(_erode_line(mb, rh, 1), rh, 1)
    thin = bars & ~_dilate_line(stems, max(1, int(font_px * 0.06)), 0)
    if not thin.any():
        return np.ones_like(mask)
    heavy = nrng.random() < 0.28
    strength = nrng.uniform(0.85, 1.0) if heavy else nrng.uniform(0.35, 0.75)
    thr = nrng.uniform(0.30, 0.45) if heavy else nrng.uniform(0.45, 0.62)
    h, w = mask.shape
    noise = fractal_noise(h, w, nrng, octaves=5)
    thin_soft = np.asarray(Image.fromarray((thin * 255).astype(np.uint8))
                           .filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255.0
    eat = strength * (noise > thr).astype(np.float32) * thin_soft
    return np.clip(1.0 - eat, 0.0, 1.0)


def carve(stone, mask, nrng: np.random.Generator, font_px=90.0):
    """Composite the text mask onto stone as a weathered carved v-cut relief."""
    h, w = mask.shape
    # erosion eats the carving before lighting
    erosion = fractal_noise(h, w, nrng, octaves=5)
    keep = 1.0 - nrng.uniform(0.0, 0.45) * (erosion > nrng.uniform(0.55, 0.75)).astype(np.float32)
    m = mask * keep * structural_erosion(mask, font_px, nrng)

    depth_img = Image.fromarray((m * 255).astype(np.uint8))
    blur_r = nrng.uniform(0.8, 3.2) * max(0.35, min(1.0, font_px / 90.0))
    height = np.asarray(depth_img.filter(ImageFilter.GaussianBlur(blur_r)), np.float32) / 255.0
    gy, gx = np.gradient(height)

    az = nrng.uniform(0, 2 * math.pi)
    strength = nrng.uniform(280, 540)
    shade = (gx * math.cos(az) + gy * math.sin(az)) * strength
    if nrng.random() < 0.15:                        # raised relief (bronze/caelatae)
        shade = -shade
        groove_dark = -m * nrng.uniform(20, 60)
    else:
        groove_dark = m * nrng.uniform(38, 95)

    local_lum = np.asarray(Image.fromarray(stone.mean(axis=2).astype(np.uint8))
                           .filter(ImageFilter.GaussianBlur(25)), np.float32) / 255.0
    lum_mod = 0.55 + 0.45 * local_lum
    out = stone + (shade * lum_mod)[..., None] - (groove_dark * lum_mod)[..., None]

    # cracks: random-walk polylines with a bright offset edge
    crack_layer = Image.new("L", (w, h), 0)
    cd = ImageDraw.Draw(crack_layer)
    for _ in range(int(nrng.integers(0, 4))):
        x, y = nrng.uniform(0, w), nrng.uniform(0, h)
        ang = nrng.uniform(0, 2 * math.pi)
        pts = [(x, y)]
        for _ in range(int(nrng.integers(10, 40))):
            ang += nrng.uniform(-0.6, 0.6)
            x += math.cos(ang) * nrng.uniform(5, 18)
            y += math.sin(ang) * nrng.uniform(5, 18)
            pts.append((x, y))
        cd.line(pts, fill=255, width=int(nrng.integers(1, 3)))
    crack = np.asarray(crack_layer, np.float32) / 255.0
    out -= crack[..., None] * nrng.uniform(25, 70)
    out += np.roll(crack, 2, axis=1)[..., None] * nrng.uniform(5, 20)

    pit = fractal_noise(h, w, nrng, octaves=6)
    out -= ((pit > 0.78).astype(np.float32) * nrng.uniform(8, 30))[..., None]
    return np.clip(out, 0, 255)


# ---------------------------------------------------------------------------
# Camera: affine warp (exact coords) + photometric (fade, B&W, noise, blur)
# ---------------------------------------------------------------------------
def camera(arr, geoms, rng: random.Random, nrng: np.random.Generator):
    """Global affine + photometric camera. Transforms the PAGE geometry through
    the same forward map, so baselines/polygons stay pixel-exact."""
    h, w = arr.shape[:2]
    img = Image.fromarray(arr.astype(np.uint8))
    rot = math.radians(rng.uniform(-3.0, 3.0))
    shx = rng.uniform(-0.04, 0.04)
    cx, cy = w / 2, h / 2
    cos, sin = math.cos(rot), math.sin(rot)

    def fwd(x, y):
        x, y = x - cx, y - cy
        x = x + shx * y
        x, y = x * cos - y * sin, x * sin + y * cos
        return x + cx, y + cy

    def inv(xp, yp):
        x, y = xp - cx, yp - cy
        x, y = x * cos + y * sin, -x * sin + y * cos
        x = x - shx * y
        return x + cx, y + cy

    p00, p10, p01 = inv(0, 0), inv(1, 0), inv(0, 1)
    coeffs = (p10[0] - p00[0], p01[0] - p00[0], p00[0],
              p10[1] - p00[1], p01[1] - p00[1], p00[1])
    img = img.transform((w, h), Image.AFFINE, coeffs, resample=Image.BILINEAR,
                        fillcolor=(120, 115, 105))

    new_geoms = [{
        "text": g["text"],
        "baseline": [fwd(x, y) for x, y in g["baseline"]],
        "poly": [fwd(x, y) for x, y in g["poly"]],
    } for g in geoms]

    arr = np.asarray(img, np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    arr *= (1.0 - nrng.uniform(0.05, 0.30) * d ** 2)[..., None]   # vignette
    arr *= nrng.uniform(0.70, 1.25)                               # exposure
    arr[..., 0] *= nrng.uniform(0.95, 1.06)
    arr[..., 2] *= nrng.uniform(0.94, 1.05)

    if rng.random() < 0.40:                                       # B/W archive photo
        gray = arr.mean(axis=2, keepdims=True)
        gray = np.clip(gray / 255.0, 0, 1) ** nrng.uniform(0.65, 1.5) * 255.0
        if rng.random() < 0.65:
            arr = np.repeat(gray, 3, axis=2)
        else:
            arr = gray * np.array([1.12, 0.97, 0.78], np.float32)[None, None, :]
        arr += nrng.normal(0, nrng.uniform(2.0, 7.0), arr.shape)

    arr += nrng.normal(0, nrng.uniform(1.0, 5.0), arr.shape)      # sensor noise
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.8)))   # blur
    jpeg_q = rng.randint(45, 92)
    return img, new_geoms, jpeg_q


# ---------------------------------------------------------------------------
# PAGE-XML
# ---------------------------------------------------------------------------
def page_xml(name, W, H, geoms) -> str:
    def pts(seq): return " ".join(f"{int(round(x))},{int(round(y))}" for x, y in seq)
    lines = []
    for i, g in enumerate(geoms):
        lines.append(
            f'      <TextLine id="l{i}">\n'
            f'        <Coords points="{pts(g["poly"])}"/>\n'
            f'        <Baseline points="{pts(g["baseline"])}"/>\n'
            f'        <TextEquiv><Unicode>{g["text"]}</Unicode></TextEquiv>\n'
            f'      </TextLine>')
    region = f'0,0 {W},0 {W},{H} 0,{H}'
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">\n'
        f'  <Page imageFilename="../images/{name}.jpg" imageWidth="{W}" imageHeight="{H}">\n'
        f'    <TextRegion id="r0">\n      <Coords points="{region}"/>\n'
        + "\n".join(lines) +
        '\n    </TextRegion>\n  </Page>\n</PcGts>\n')


def render_one(corpus, W, H, fonts, dots, rng, nrng):
    lines = list(rng.choice(corpus))
    font_path = fonts[rng.randrange(len(fonts))]        # one font per image (seeded)
    lines_gt = ["·".join(ln.split()) for ln in lines] if dots else lines
    mask, geoms, font_px = layout(lines_gt, W, H, font_path, rng)
    stone = load_stone_image(W, H, rng)
    if stone is None:
        stone = make_stone(W, H, rng.choice(list(STONE_PALETTES)), nrng)
    carved = carve(stone, mask, nrng, font_px)
    img, geoms, _ = camera(carved, geoms, rng, nrng)
    return img, geoms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="number of images")
    ap.add_argument("--outdir", default="data/synth", help="output dir (creates images/ + page/)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dots", action=argparse.BooleanOptionalAction, default=True,
                    help="middot (·) between words (default: on; --no-dots for scriptio continua)")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--corpus", default="inscription_texts.txt",
                    help="text corpus (one inscription/line, its lines '|'-joined); "
                         "falls back to the small built-in list if absent")
    a = ap.parse_args()

    out = Path(a.outdir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "page").mkdir(parents=True, exist_ok=True)
    fonts = get_font_paths()

    cpath = Path(a.corpus)
    if cpath.exists():
        corpus = [[p.strip() for p in ln.split("|") if p.strip()]
                  for ln in cpath.read_text(encoding="utf-8").splitlines()
                  if ln.strip() and not ln.startswith("#")]
        corpus = [c for c in corpus if c]
        print(f"corpus: {len(corpus)} inscriptions from {cpath}")
    else:
        corpus = CORPUS
        print(f"corpus: built-in {len(corpus)} inscriptions ({cpath} not found)")
    shown = ", ".join(Path(f).stem for f in fonts if f) or "Pillow default"
    n_stone = len(list((HERE / "stone_images").glob("*"))) if (HERE / "stone_images").exists() else 0
    stone_src = f"{n_stone} real stone patches" if n_stone else "procedural stone"
    print(f"fonts ({len(fonts)}): {shown}")
    print(f"stone: {stone_src}  ->  {out}  (n={a.n}, dots={a.dots})")

    for i in range(a.n):
        rng = random.Random(a.seed + i)
        nrng = np.random.default_rng(a.seed + i)
        name = f"synth_{i:05d}"
        img, geoms = render_one(corpus, a.width, a.height, fonts, a.dots, rng, nrng)
        img.save(out / "images" / f"{name}.jpg", quality=int(nrng.integers(70, 92)))
        (out / "page" / f"{name}.xml").write_text(
            page_xml(name, a.width, a.height, geoms), encoding="utf-8")
    print(f"wrote {a.n} images + PAGE-XML to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
