#!/usr/bin/env python3
"""ScanMend dataset synthesis: render clean synthetic documents, then apply
programmatic degradations with exact labels. Sharded + idempotent (skips
existing shards) so it can resume across runs.

Honesty notes:
- Documents are synthetic renders (random layouts/fonts/words), not real scans.
- Labels are the applied degradation parameters: exact for the synthetic
  pipeline, NOT real-world quality ground truth.
- Eval split uses held-out fonts AND held-out layout seeds.
"""
import os, random, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

random.seed(42); np.random.seed(42)

OUT = os.path.join(os.path.dirname(__file__), '..', 'data')
IMG = 160
N_TRAIN = 4200
N_EVAL = 600
SHARD = 300

WORDS = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
         "incididunt ut labore et dolore magna aliqua enim ad minim veniam quis nostrud "
         "exercitation ullamco laboris nisi aliquip ex ea commodo consequat duis aute irure").split()

def font_files():
    out = []
    for r in ['/usr/share/fonts', '/usr/local/share/fonts']:
        for dp, _, fns in os.walk(r):
            for fn in fns:
                if fn.lower().endswith(('.ttf', '.otf')):
                    out.append(os.path.join(dp, fn))
    return sorted(out)

_ALL = font_files()
assert _ALL, 'no fonts found'
random.Random(7).shuffle(_ALL)
ALL_FONTS = _ALL[:96]
EVAL_FONTS = [f for i, f in enumerate(ALL_FONTS) if i % 6 == 0]
TRAIN_FONTS = [f for i, f in enumerate(ALL_FONTS) if i % 6 != 0]

_FONT_CACHE = {}
def get_font(path, size):
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]

def render_doc(rng, fonts, layout_seed):
    S = IMG * 4
    img = Image.new('L', (S, S), 255)
    d = ImageDraw.Draw(img)
    rng2 = random.Random(layout_seed)
    layout = rng2.randint(0, 3)
    fp = rng.choice(fonts)
    fs_body = rng2.randint(20, 30)
    fs_head = fs_body + rng2.randint(8, 18)
    body = get_font(fp, fs_body)
    head = get_font(fp, fs_head)
    margin = rng2.randint(40, 70)

    def para(d, x, y, w, font, nlines):
        for _ in range(nlines):
            words = ' '.join(rng2.choice(WORDS) for _ in range(rng2.randint(6, 12)))
            d.text((x, y), words, fill=rng2.randint(0, 40), font=font)
            y += int(font.size * 1.45)
        return y

    if layout == 0:
        y = para(d, margin, margin, S - 2 * margin, head, 1)
        para(d, margin, y + 10, S - 2 * margin, body, rng2.randint(14, 20))
    elif layout == 1:
        cw = (S - 3 * margin) // 2
        para(d, margin, margin, cw, body, rng2.randint(16, 22))
        para(d, 2 * margin + cw, margin, cw, body, rng2.randint(16, 22))
    elif layout == 2:
        d.text((margin, margin), ' '.join(rng2.choice(WORDS) for _ in range(3)).title(),
               fill=0, font=head)
        d.line((margin, margin + fs_head + 12, S - margin, margin + fs_head + 12), fill=0, width=3)
        para(d, margin, margin + fs_head + 30, S - 2 * margin, body, rng2.randint(14, 19))
    else:
        y = margin
        rows = rng2.randint(6, 10); cols = rng2.randint(3, 5)
        cw = (S - 2 * margin) // cols; rh = (S - 2 * margin) // (rows + 1)
        for r in range(rows + 1):
            d.line((margin, y, S - margin, y), fill=0, width=2)
            for c in range(cols + 1):
                d.line((margin + c * cw, y, margin + c * cw, y + rh), fill=0, width=2)
            if r < rows:
                for c in range(cols):
                    d.text((margin + c * cw + 6, y + 6), rng2.choice(WORDS), fill=0, font=body)
            y += rh
    return img

def degrade(img, rng, params):
    out = img
    if params['skew'] != 0:
        out = out.rotate(params['skew'], resample=Image.BICUBIC, fillcolor=255)
    if params['blur'] > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=params['blur'] * 3.0))
    if params['contrast'] < 1:
        arr = np.asarray(out).astype(np.float32)
        arr = 128 + (arr - 128) * params['contrast'] + (1 - params['contrast']) * 60
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'L')
    if params['noise'] > 0:
        arr = np.asarray(out).astype(np.float32)
        arr += np.random.normal(0, params['noise'] * 22.0, arr.shape)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'L')
    return out

def sample_params(rng):
    if rng.random() < 0.2:
        return {'blur': 0.0, 'contrast': 1.0, 'noise': 0.0, 'skew': 0.0}
    p = {'blur': 0.0, 'contrast': 1.0, 'noise': 0.0, 'skew': 0.0}
    for k in rng.sample(['blur', 'contrast', 'noise', 'skew'], rng.randint(1, 4)):
        if k == 'blur': p['blur'] = rng.uniform(0.15, 1.0)
        elif k == 'contrast': p['contrast'] = rng.uniform(0.35, 0.85)
        elif k == 'noise': p['noise'] = rng.uniform(0.15, 1.0)
        else: p['skew'] = rng.uniform(-8, 8) or 0.5
    return p

def gen_split(name, n, fonts, seed0):
    rng = random.Random(hash(name) & 0xffff)
    done = {f for f in os.listdir(OUT) if f.startswith(f'X_{name}_s')}
    for s in range(0, n, SHARD):
        tag = f'{name}_s{s // SHARD:03d}'
        if f'X_{tag}.npy' in done:
            continue
        m = min(SHARD, n - s)
        X = np.zeros((m, IMG, IMG, 1), np.uint8)
        Y = np.zeros((m, 4), np.float32)
        metas = []
        for i in range(m):
            clean = render_doc(rng, fonts, seed0 + s + i)
            p = sample_params(rng)
            deg = degrade(clean, rng, p)
            X[i, :, :, 0] = np.asarray(deg.resize((IMG, IMG), Image.LANCZOS))
            Y[i] = [p['blur'], 1.0 - p['contrast'], p['noise'], p['skew'] / 8.0]
            metas.append({'i': s + i, **p})
        np.save(os.path.join(OUT, f'X_{tag}.npy'), X)
        np.save(os.path.join(OUT, f'Y_{tag}.npy'), Y)
        with open(os.path.join(OUT, f'meta_{tag}.json'), 'w') as f:
            json.dump(metas, f)
        print(tag, 'done', flush=True)

os.makedirs(OUT, exist_ok=True)
print(f'{len(ALL_FONTS)} fonts: {len(TRAIN_FONTS)} train / {len(EVAL_FONTS)} eval (held out)')
gen_split('train', N_TRAIN, TRAIN_FONTS, 100000)
gen_split('eval', N_EVAL, EVAL_FONTS, 500000)
