"""Documentário binário — Pedro Álvares Cabral.
Cada imagem é desenhada por dígitos 0 e 1 (brilho do dígito = luminância da obra),
com transições de "decodificação" e legendas. Saída: frames 1080x1920.
"""
import math, os, glob, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

W, H, FPS = 1080, 1920, 30
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "frames_bin"); os.makedirs(OUT, exist_ok=True)
DUR = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "csv=p=0", os.path.join(HERE, "narracao.mp3")]).decode()) + 1.5

SCENES = [
    ("portrait",    "Pedro Álvares Cabral saiu de Portugal com um objetivo: chegar às Índias.", 0.0),
    ("armada",      "Mas o caminho tomou outro rumo.", 5.4),
    ("cantino",     "Nem sempre o caminho que planejamos será o caminho que vamos percorrer.", 11.0),
    ("carta",       "Às vezes, as coisas simplesmente não acontecem como imaginava.", 17.0),
    ("escola",      "O erro é acreditar que mudar a rota significa fracassar.", 26.1),
    ("desembarque", "Cabral encontrou um novo território e continuou sua jornada.", 30.1),
    ("missa",       "Reconheça oportunidades — sem abandonar seus objetivos.", 34.1),
    ("estatua",     "Planeje. Tenha uma direção. Mas não seja preso ao plano.", 43.5),
    ("portrait",    "O que parece um desvio pode revelar uma oportunidade.", 49.7),
]

SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT = ImageFont.truetype(SERIF, 50)
SMALL = ImageFont.truetype(MONO, 28)
GOLD = np.array([226, 186, 110], np.float32)

# grade de células: cada célula recebe um "0" ou "1"
CW, CH = 12, 18
COLS, ROWS = W // CW, H // CH
gfont = ImageFont.truetype(MONO, 17)
def glyph(ch):
    g = Image.new("L", (CW, CH), 0)
    ImageDraw.Draw(g).text((CW / 2, CH / 2), ch, font=gfont, fill=255, anchor="mm")
    return np.asarray(g, np.float32) / 255
G = np.stack([glyph("0"), glyph("1")])  # (2, CH, CW)

def prep(name):
    path = glob.glob(os.path.join(HERE, "img", name + ".*"))[0]
    im = ImageOps.autocontrast(ImageOps.grayscale(Image.open(path).convert("RGB")), cutoff=1)
    s = max(W * 1.12 / im.width, H * 1.12 / im.height)
    return im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)

cache = {}
def lum_grid(name, p, seed):
    """Luminância (ROWS, COLS) com Ken Burns lento."""
    if name not in cache: cache[name] = prep(name)
    im = cache[name]
    z = 1.0 + 0.10 * p
    cw, ch = im.width / 1.12 / z, im.height / 1.12 / z
    dx = (im.width - cw) * (0.5 + 0.4 * math.sin(seed) * (p - 0.5))
    dy = (im.height - ch) * (0.5 + 0.4 * math.cos(seed) * (p - 0.5))
    c = im.crop((int(dx), int(dy), int(dx + cw), int(dy + ch))).resize((COLS, ROWS), Image.BOX)
    return np.asarray(c, np.float32) / 255

def ease(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

def wrap(text, font, maxw):
    lines, cur = [], ""
    for w_ in text.split():
        tst = (cur + " " + w_).strip()
        if font.getlength(tst) <= maxw: cur = tst
        else: lines.append(cur); cur = w_
    lines.append(cur); return lines

rng = np.random.default_rng(11)
bits = rng.integers(0, 2, (ROWS, COLS))
# ordem de decodificação por cena: varredura diagonal + ruído
yy, xx = np.mgrid[0:ROWS, 0:COLS]
delays = [np.clip(0.55 * ((xx / COLS + yy / ROWS) / 2 if k % 2 else 1 - (xx / COLS + yy / ROWS) / 2)
                  + 0.45 * rng.random((ROWS, COLS)), 0, 1) for k in range(len(SCENES))]
# vinheta
vig = np.clip(1.15 - 0.9 * (((xx - COLS / 2) / (COLS / 1.6)) ** 2 + ((yy - ROWS / 2.3) / (ROWS / 1.5)) ** 2), 0.15, 1)

TRANS = 1.6
total = int(DUR * FPS)
import sys
ONLY = set(int(x) for x in sys.argv[1:])
for f in (sorted(ONLY) if ONLY else range(total)):
    t = f / FPS
    i = max(k for k, s in enumerate(SCENES) if s[2] <= t)
    start = SCENES[i][2]
    end = SCENES[i + 1][2] if i + 1 < len(SCENES) else DUR
    L = lum_grid(SCENES[i][0], (t - start) / (end - start), i * 1.7)
    glow = np.zeros_like(L)
    if i > 0 and t - start < TRANS:
        prevL = lum_grid(SCENES[i - 1][0], 1.0, (i - 1) * 1.7)
        q = (t - start) / TRANS
        m = ease((q - delays[i] * 0.7) / 0.3)
        glow = np.exp(-((m - 0.5) ** 2) / 0.02) * 0.9      # brilho na frente de decodificação
        L = prevL * (1 - m) + L * m
    if t < 1.2:  # abertura: imagem surge do escuro
        L = L * ease((t / 1.2 - delays[0] * 0.6) / 0.4)

    # dígitos trocam constantemente; mais rápido onde está decodificando
    flip = rng.random((ROWS, COLS)) < (0.03 + 0.5 * glow)
    bits = np.where(flip, 1 - bits, bits)

    inten = np.clip(1.7 * L ** 0.85 * vig + glow, 0, 1.6)            # (ROWS, COLS)
    cells = G[bits] * inten[:, :, None, None]                   # (ROWS, COLS, CH, CW)
    img = cells.transpose(0, 2, 1, 3).reshape(ROWS * CH, COLS * CW)
    rgb = np.zeros((H, W, 3), np.float32) + np.array([6, 5, 4], np.float32)
    col = GOLD + (np.array([255, 245, 220]) - GOLD) * np.clip(img - 0.8, 0, 1)[..., None] * 2
    rgb[:ROWS * CH, :COLS * CW] += img[..., None] * col
    frame = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGBA")

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
    a = float(ease((t - start) / 0.5) * ease((end - t) / 0.4))
    lines = wrap(SCENES[i][1], FONT, W - 160)
    ly = H - 420
    od.rectangle((0, ly - 60, W, H), fill=(0, 0, 0, int(185 * a)))
    for k, ln in enumerate(lines):
        od.text(((W - FONT.getlength(ln)) / 2, ly + k * 66), ln, font=FONT, fill=(248, 238, 218, int(255 * a)))
    od.rectangle((0, 60, W, 140), fill=(0, 0, 0, 170))
    od.text((60, 82), "PEDRO ÁLVARES CABRAL · 1467–1520", font=SMALL, fill=(240, 205, 130, 235))
    od.text((W - 60, 82), f"{int(t // 60):02d}:{t % 60:05.2f}", font=SMALL, fill=(240, 205, 130, 160), anchor="ra")
    Image.alpha_composite(frame, ov).convert("RGB").save(os.path.join(OUT, f"{f:05d}.jpg"), quality=90)
    if f % 300 == 0: print(f, "/", total, flush=True)
