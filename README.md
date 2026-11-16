# ScanMend

Fix rough document photos and scans entirely in your browser: a small multi-head CNN
diagnoses blur, contrast loss, noise and skew from a thumbnail; ScanMend then applies
targeted corrections (model-driven unsharp-mask sharpening and contrast boost, plus manual deskew and optional manual denoise), shows a
before/after comparison, OCRs the result, and exports a **searchable PDF** (corrected
image + invisible text layer).

Everything runs client-side. Your document never leaves the browser.

## Run it

- Hosted: GitHub Pages deployment (see repo About / deployments).
- Local: serve `app/` statically and open it, e.g. `cd app && python3 -m http.server 8000`
  then http://localhost:8000. First load needs internet access for the CDN libraries
  (TensorFlow.js, Tesseract.js, jsPDF) and the OCR language data.

## What the model actually does

- Input: 160x160 grayscale thumbnail of the document (direct squash resize, no letterbox - matches synthesize.py).
- Architecture: 4 conv blocks (24/48/96/128 filters) -> GAP -> dense 96 -> two heads:
  `pos` = [blur, contrast_loss, noise] severities in [0,1]; `skew` = tanh, x8 = degrees.
- Training data: 4,800 synthetic documents (96 fonts, held-out fonts/layouts for eval)
  degraded with known parameters (`train/synthesize.py`), 4,200 train / 600 eval.
- Trained with `train/train_model.py` (chunked CPU training), exported with
  `train/export_tfjs_graph.py` + `train/patch_graph_tanh.py` to `app/model/` (tfjs 4.22 is run on the CPU backend - WebGL returns input-independent output for this graph; CPU matches the Python model exactly).

## Honest evaluation (600 held-out synthetic images, held-out fonts)

| Signal | MAE | Presence detection (thr 0.15) |
|---|---|---|
| blur | 0.140 | 78.8% |
| contrast_loss | 0.141 | 80.3% |
| noise | 0.260 | 54.7% (near chance) |
| skew | 0.28 degrees | - |

The noise head is weak: in the current synthesis noise is applied at 4x resolution and
the downscale to the 160px thumbnail averages most of it away, so low-severity noise is
genuinely hard to see. The app therefore never auto-applies denoise:
the slider starts at 0 and is manual-only, with the estimate shown for reference. A fix (apply noise after downscaling)
is documented in `train/synthesize.py` plans but not yet trained.

Evaluated on synthetic data only; real phone photos may differ. Corrections are tuned
on synthetic degradations; on real photos the sharpen/contrast controls are visibility
aids, not verified restorations. OCR quality depends on the source image;
low-confidence words are reported in the UI. Skew estimation failed validation on real photos (wrong sign), so deskew is manual-only; the estimate is shown for reference.

## OCR recovery evaluation (40 fresh documents, unseen document fonts, known ground-truth text)

Word error rate (WER, lower is better) of Tesseract OCR on three variants:

| Variant | Mean WER | Median WER |
|---|---|---|
| degraded as-is | 0.505 | 0.189 |
| model-driven correction | **0.412** | **0.118** |
| fixed baseline (autocontrast + unsharp) | 0.549 | 0.224 |

Model-driven correction improved OCR recovery on average against both doing nothing
and a simple deterministic baseline. It made things *worse* than the uncorrected
image in 20% of cases (mostly mild degradations where sharpening overshoots) - the
sliders exist so you can dial back, and the original is always preserved. The fixed
baseline was worse than no correction on mean WER, which is why the correction is
model-driven rather than a fixed recipe.

## Known limits

- Export is a *searchable* PDF (image + invisible text layer), not a tagged
  *accessible* PDF - tagging and reading order were not implemented or tested.
- CDN dependency: the app is not guaranteed to work offline; cold load needs network.
- Corrections are parameterized filters, not learned restoration - ScanMend will not
  hallucinate missing text. OCR errors are shown, not hidden.
- When fewer than 50% of OCR words are usable, the app flags the result as too low
  quality for reliable text search before export: the PDF keeps the full image, the
  text layer is labeled as possibly inaccurate, and retaining omitted words is an
  explicit opt-in marked "search accuracy not guaranteed". Words whose boxes fall
  outside the page are always omitted.

## AI assistance disclosure

Built with AI coding assistance (design iteration, boilerplate, debugging). Model
architecture, training data design, evaluation and all final decisions verified by the
author. Kane CLI was used to QA the deployed app.

## Licenses

Code: MIT (see LICENSE). Libraries via CDN: TensorFlow.js (Apache-2.0), Tesseract.js
(Apache-2.0), jsPDF (MIT). Model weights: trained by the author on synthetic data,
MIT.
