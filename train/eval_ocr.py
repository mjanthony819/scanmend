"""OCR-recovery evaluation for ScanMend.

Renders fresh full-res documents with KNOWN ground-truth text (held-out fonts),
degrades them with the same distribution as training, then OCRs three variants:
  a) degraded        - no correction
  b) model           - model-driven correction (deskew + sharpen + contrast boost,
                       matching app behavior; denoise never auto-applied)
  c) baseline        - fixed deterministic correction (autocontrast + unsharp), no model
Metric: word error rate (WER) vs ground truth, per variant, plus harm rate
(model worse than degraded). Incremental: appends to eval_ocr_rows.jsonl and
resumes; final summary written to eval_ocr.json.

Usage: python3 eval_ocr.py --n 40 --seconds 100
"""
import os, sys, json, glob, random, argparse, subprocess, tempfile, time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# do NOT import synthesize (it generates at import); copy its constants
WORDS = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
         "incididunt ut labore et dolore magna aliqua enim ad minim veniam quis nostrud "
         "exercitation ullamco laboris nisi aliquip ex ea commodo consequat").split()
IMG = 160
S = IMG * 4

def all_fonts():
    out = []
    for root in ('/usr/share/fonts', '/usr/local/share/fonts'):
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(('.ttf', '.otf')):
                    out.append(os.path.join(dirpath, f))
    return sorted(out)

# replicate synthesize.py's shuffle, then take fonts OUTSIDE its 96 used in training,
# filtered to plain document text families (no keyboard/symbol/math/display variants)
_GOOD = ('DejaVuSerif', 'DejaVuSans-', 'Liberation', 'Nimbus', 'FreeSerif', 'FreeSans',
         'NotoSerif-', 'NotoSans-', 'Carlito', 'Caladea', 'Gelasio', 'Tinos', 'Arimo')
_BAD = ('_K', 'Keyboard', 'Math', 'Icon', 'Bold', 'Italic')  # keep weight/style plain-ish
_ALL = all_fonts()
random.Random(7).shuffle(_ALL)
UNSEEN = _ALL[96:]
EVAL_FONTS = [f for f in UNSEEN if any(g in os.path.basename(f) for g in _GOOD)
              and '_K' not in os.path.basename(f) and 'Keyboard' not in f and 'Math' not in f]
assert len(EVAL_FONTS) >= 8, f'only {len(EVAL_FONTS)} usable held-out fonts'

def degrade(img, rng, p):  # same as synthesize.degrade
    out = img
    if p['skew'] != 0:
        out = out.rotate(p['skew'], resample=Image.BICUBIC, fillcolor=255)
    if p['blur'] > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=p['blur'] * 3.0))
    if p['contrast'] < 1:
        arr = np.asarray(out).astype(np.float32)
        arr = 128 + (arr - 128) * p['contrast'] + (1 - p['contrast']) * 60
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'L')
    if p['noise'] > 0:
        arr = np.asarray(out).astype(np.float32)
        arr += np.random.normal(0, p['noise'] * 22.0, arr.shape)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'L')
    return out

def sample_params(rng):
    if rng.random() < 0.2:
        return {'blur': 0.0, 'contrast': 1.0, 'noise': 0.0, 'skew': 0.0}
    p = {'blur': 0.0, 'contrast': 1.0, 'noise': 0.0, 'skew': 0.0}
    for k in rng.sample(['blur', 'contrast', 'noise', 'skew'], rng.randint(1, 4)):
        if k == 'blur': p['blur'] = rng.uniform(0.15, 1.0)
        elif k == 'contrast': p['contrast'] = rng.uniform(0.45, 0.9)
        elif k == 'noise': p['noise'] = rng.uniform(0.15, 1.0)
        else: p['skew'] = rng.uniform(-8, 8)
    return p

def render_gt_doc(rng, font_path):
    """Render a paragraph document, return (image, ground_truth_text)."""
    img = Image.new('L', (S, S), 255)
    d = ImageDraw.Draw(img)
    title_f = ImageFont = __import__('PIL.ImageFont', fromlist=['truetype']).truetype(font_path, 34)
    body = __import__('PIL.ImageFont', fromlist=['truetype']).truetype(font_path, 22)
    margin = 60
    y = margin
    title = ' '.join(rng.sample(WORDS, 4))
    d.text((margin, y), title, fill=0, font=title_f)
    y += 60
    words = [rng.choice(WORDS) for _ in range(rng.randint(60, 90))]
    x = margin
    gt = [title]
    line = []
    for w in words:
        ww = d.textlength(w + ' ', font=body)
        if x + ww > S - margin:
            d.text((margin, y), ' '.join(line), fill=0, font=body)
            gt.append(' '.join(line)); line = []; y += 34; x = margin
        line.append(w); x += ww
    if line:
        d.text((margin, y), ' '.join(line), fill=0, font=body)
        gt.append(' '.join(line))
    return img, '\n'.join(gt)

# ---- model correction (mirrors app JS: deskew, unsharp, contrast boost) ----
_model = None
def model_params(deg_img):
    global _model
    if _model is None:
        import tensorflow as tf
        import importlib.util
        spec = importlib.util.spec_from_file_location('tm', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_model.py'))
        tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
        _model = tm.build(); _model.load_weights(tm.BEST)
    import tensorflow as tf
    th = deg_img.resize((IMG, IMG), Image.LANCZOS)
    x = np.asarray(th).astype(np.float32)[None, :, :, None] / 255.0
    out = _model.predict(x, verbose=0)
    pos, skew = out['pos'][0], out['skew'][0]
    return float(pos[0]), float(pos[1]), float(pos[2]), float(skew[0]) * 8.0

def box_blur_np(g, r):
    from scipy import ndimage  # may not exist; fallback below
    return ndimage.uniform_filter(g, size=2 * r + 1)

def box_blur(g, r):
    try:
        return box_blur_np(g, r)
    except Exception:
        k = 2 * r + 1
        c = np.cumsum(np.pad(g, ((k, k), (k, k))), axis=0)
        gg = (c[k:, :] - c[:-k, :]) ; c2 = np.cumsum(np.pad(gg, ((0, 0), (k, k))), axis=1)
        out = (c2[:, k:] - c2[:, :-k]) / (k * k)
        return out[: g.shape[0], : g.shape[1]]

def model_correct(deg_img, blur, con_loss, skew_deg):
    img = deg_img.rotate(-skew_deg, resample=Image.BICUBIC, fillcolor=255) if abs(skew_deg) > 0.05 else deg_img
    g = np.asarray(img).astype(np.float32)
    if blur > 0.01:
        bl = box_blur(g, 2); amt = blur * 1.4
        g = np.clip(g + amt * (g - bl), 0, 255)
    if con_loss > 0.01:
        cst = max(0.15, 1 - con_loss)
        g = np.clip((g - 128 - (1 - cst) * 60) / cst + 128, 0, 255)
    return Image.fromarray(g.astype(np.uint8), 'L')

def baseline_correct(deg_img):
    img = ImageOps.autocontrast(deg_img, cutoff=1)
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))

def ocr(img, tmpdir, tag):
    p = os.path.join(tmpdir, tag + '.png')
    img.save(p)
    r = subprocess.run(['tesseract', p, 'stdout', '--psm', '6'], capture_output=True, text=True, timeout=60)
    return r.stdout

def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r: return 0.0 if not h else 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), np.int32)
    d[:, 0] = np.arange(len(r) + 1); d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (r[i-1] != h[j-1]))
    return float(d[len(r), len(h)]) / len(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--seconds', type=int, default=100)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows_path = os.path.join(root, 'model', 'eval_ocr_rows.jsonl')
    done = set()
    if os.path.exists(rows_path):
        for line in open(rows_path):
            try: done.add(json.loads(line)['i'])
            except Exception: pass
    rng = random.Random(777)
    np_rng_state = np.random.get_state()
    np.random.seed(777)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(a.n):
            if i in done or time.time() - t0 > a.seconds: continue
            frng = random.Random(900000 + i)
            clean, gt = render_gt_doc(frng, EVAL_FONTS[i % len(EVAL_FONTS)])
            p = sample_params(rng)
            deg = degrade(clean, rng, p)
            blur, con, noi, skew = model_params(deg)
            variants = {'degraded': deg,
                        'model': model_correct(deg, blur, con, skew),
                        'baseline': baseline_correct(deg)}
            row = {'i': i, 'params': p, 'est': {'blur': blur, 'con': con, 'noi': noi, 'skew': skew}}
            for name, im in variants.items():
                row[name] = wer(gt, ocr(im, tmp, f'{i}_{name}'))
            with open(rows_path, 'a') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
    # aggregate
    rows = [json.loads(l) for l in open(rows_path)]
    agg = {}
    for name in ('degraded', 'model', 'baseline'):
        ws = [r[name] for r in rows]
        agg[name] = {'mean_wer': float(np.mean(ws)), 'median_wer': float(np.median(ws))}
    agg['n'] = len(rows)
    agg['model_worse_than_degraded'] = float(np.mean([r['model'] > r['degraded'] for r in rows]))
    agg['model_better_than_baseline'] = float(np.mean([r['model'] < r['baseline'] for r in rows]))
    with open(os.path.join(root, 'model', 'eval_ocr.json'), 'w') as f:
        json.dump(agg, f, indent=2)
    print('AGG ' + json.dumps(agg))

if __name__ == '__main__':
    main()
