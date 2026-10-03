"""Documentário hexagonal — Pedro Álvares Cabral.
Gera frames 1080x1920 com mosaico de hexágonos que revelam cada imagem, Ken Burns e legendas.
"""
import math, os, glob, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

W, H, FPS = 1080, 1920, 30
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "frames"); os.makedirs(OUT, exist_ok=True)
DUR = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "csv=p=0", os.path.join(HERE, "narracao.mp3")]).decode()) + 1.5

# (imagem, legenda, início em segundos) — inícios ajustados às pausas da narração
SCENES = [
    ("img/portrait.jpg",  "Pedro Álvares Cabral saiu de Portugal com um objetivo: chegar às Índias.", 0.0),
    ("img/armada.jpg",    "Mas o caminho tomou outro rumo.", 5.4),
    ("img/cantino.jpg",   "Nem sempre o caminho que planejamos será o caminho que vamos percorrer.", 11.0),
    ("img/carta.jpg",     "Às vezes, as coisas simplesmente não acontecem como imaginava.", 17.0),
    ("img/escola.jpg",    "O erro é acreditar que mudar a rota significa fracassar.", 26.1),
    ("img/desembarque.jpg", "Cabral encontrou um novo território e continuou sua jornada.", 30.1),
    ("img/missa.jpg",     "Reconheça oportunidades — sem abandonar seus objetivos.", 34.1),
    ("img/estatua.jpg",   "Planeje. Tenha uma direção. Mas não seja preso ao plano.", 43.5),
    ("img/portrait.jpg",  "O que parece um desvio pode revelar uma oportunidade.", 49.7),
]

GOLD = (201, 162, 92)
font_path = next((p for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"] if os.path.exists(p)), None)
FONT = ImageFont.truetype(font_path, 50) if font_path else ImageFont.load_default()
SMALL = ImageFont.truetype(font_path, 30) if font_path else ImageFont.load_default()

# grade de hexágonos (pontas para cima)
R = 120
hw = math.sqrt(3) * R
centers = []
row = 0
y = -R
while y < H + R:
    x0 = -hw / 2 if row % 2 else 0
    x = x0
    while x < W + hw:
        centers.append((x, y)); x += hw
    y += 1.5 * R; row += 1

def hexpoly(cx, cy, r):
    return [(cx + r * math.cos(math.radians(60 * k - 30)), cy + r * math.sin(math.radians(60 * k - 30))) for k in range(6)]

def prep(path):
    im = ImageOps.grayscale(Image.open(os.path.join(HERE, path)).convert("RGB"))
    im = ImageOps.autocontrast(im, cutoff=1)
    # tom sépia documental
    im = ImageOps.colorize(im, (14, 10, 6), (236, 214, 170), mid=(120, 92, 58))
    s = max(W * 1.15 / im.width, H * 1.15 / im.height)
    return im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)

cache = {}
def frame_img(path, t, d, seed):
    if path not in cache: cache[path] = prep(glob.glob(os.path.join(HERE, os.path.splitext(path)[0] + ".*"))[0])
    im = cache[path]
    p = t / d
    z = 1.0 + 0.08 * p
    cw, ch = W * 1.0 / z * (im.width / (W * 1.15)), H * 1.0 / z * (im.height / (H * 1.15))
    dx = (im.width - cw) * (0.5 + 0.35 * math.sin(seed) * (p - 0.5))
    dy = (im.height - ch) * (0.5 + 0.35 * math.cos(seed) * (p - 0.5))
    return im.crop((int(dx), int(dy), int(dx + cw), int(dy + ch))).resize((W, H), Image.BILINEAR)

def ease(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)

def wrap(text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        tst = (cur + " " + w_).strip()
        if font.getlength(tst) <= maxw: cur = tst
        else: lines.append(cur); cur = w_
    lines.append(cur); return lines

rng = np.random.default_rng(7)
orders = [rng.permutation(len(centers)) for _ in SCENES]
vignette = Image.new("L", (W, H), 0)
ImageDraw.Draw(vignette).ellipse((-300, -200, W + 300, H + 200), fill=255)
vignette = vignette.filter(ImageFilter.GaussianBlur(220))

TRANS = 1.4  # duração da revelação hexagonal
total = int(DUR * FPS)
for f in range(total):
    t = f / FPS
    i = max(k for k, s in enumerate(SCENES) if s[2] <= t)
    start = SCENES[i][2]
    end = SCENES[i + 1][2] if i + 1 < len(SCENES) else DUR
    cur = frame_img(SCENES[i][0], t - start, end - start, i * 1.7)
    if i > 0 and t - start < TRANS:
        ps, pe = SCENES[i - 1][2], start
        prev = frame_img(SCENES[i - 1][0], t - ps, end - ps, (i - 1) * 1.7)
        mask = Image.new("L", (W, H), 0); md = ImageDraw.Draw(mask)
        q = (t - start) / TRANS
        n = len(centers)
        for rank, idx in enumerate(orders[i]):
            local = ease((q - rank / n * 0.6) / 0.4)
            if local > 0:
                cx, cy = centers[idx]; md.polygon(hexpoly(cx, cy, R * local + 1), fill=255)
        base = Image.composite(cur, prev, mask)
    else:
        base = cur.copy()
    base = Image.composite(base, Image.new("RGB", (W, H), (8, 6, 4)), vignette)

    # moldura hexagonal: grade fina sempre visível + hexágono central destacado
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
    for cx, cy in centers:
        od.polygon(hexpoly(cx, cy, R), outline=(10, 8, 6, 150), width=5)
    pulse = 0.5 + 0.5 * math.sin(t * 1.3)
    od.polygon(hexpoly(W / 2, H * 0.42, 430), outline=GOLD + (int(150 + 80 * pulse),), width=4)
    od.polygon(hexpoly(W / 2, H * 0.42, 448), outline=GOLD + (70,), width=2)

    # legenda
    a = ease((t - start) / 0.5) * ease((end - t) / 0.4)
    lines = wrap(SCENES[i][1], FONT, W - 160)
    ly = H - 420
    od.rectangle((0, ly - 60, W, H), fill=(0, 0, 0, int(150 * a)))
    for k, ln in enumerate(lines):
        lw = FONT.getlength(ln)
        od.text(((W - lw) / 2, ly + k * 66), ln, font=FONT, fill=(245, 235, 215, int(255 * a)))
    od.text((70, 90), "PEDRO ÁLVARES CABRAL  ·  1467–1520", font=SMALL, fill=GOLD + (200,))
    frame = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
    # granulação de filme
    if f % 2 == 0:
        grain = (rng.standard_normal((H // 4, W // 4)) * 10).astype(np.int16)
        grain = np.kron(grain, np.ones((4, 4), dtype=np.int16))[:, :, None]
    frame = Image.fromarray(np.clip(np.asarray(frame, dtype=np.int16) + grain, 0, 255).astype(np.uint8))
    frame.save(os.path.join(OUT, f"{f:05d}.jpg"), quality=88)
    if f % 150 == 0: print(f, "/", total, flush=True)
