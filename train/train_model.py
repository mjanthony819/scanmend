#!/usr/bin/env python3
"""ScanMend model: multi-head CNN over 160x160 grayscale document thumbnails.
Heads: blur / contrast-loss / noise (sigmoid), skew/8 (tanh).
Chunked CPU training with warm restarts; --finalize runs held-out eval + tfjs export.
"""
import os, glob, json, time, argparse
import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(ROOT, 'data')
CKPT = os.path.join(ROOT, 'model', 'ckpt.weights.h5')
BEST = os.path.join(ROOT, 'model', 'best.weights.h5')

def load_split(name):
    Xs, Ys = [], []
    for f in sorted(glob.glob(os.path.join(DATA, f'X_{name}_s*.npy'))):
        Xs.append(np.load(f))
        Ys.append(np.load(f.replace('X_', 'Y_')))
    return np.concatenate(Xs).astype(np.float32) / 255.0, np.concatenate(Ys)

def build():
    inp = tf.keras.Input((160, 160, 1))
    x = inp
    for f in (24, 48, 96, 128):
        x = tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(96, activation='relu')(x)
    pos = tf.keras.layers.Dense(3, activation='sigmoid', name='pos')(x)
    skew = tf.keras.layers.Dense(1, activation='tanh', name='skew')(x)
    return tf.keras.Model(inp, {'pos': pos, 'skew': skew})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=int, default=90)
    ap.add_argument('--finalize', action='store_true')
    ap.add_argument('--lr', type=float, default=1e-3)
    a = ap.parse_args()

    Xtr, Ytr = load_split('train')
    model = build()
    if os.path.exists(CKPT):
        model.load_weights(CKPT)
    model.compile(optimizer=tf.keras.optimizers.Adam(a.lr),
                  loss={'pos': 'mse', 'skew': 'mse'})
    t0 = time.time(); steps = 0; bs = 64; n = len(Xtr)
    rng = np.random.default_rng(0)
    h = {'loss': float('nan')}
    while time.time() - t0 < a.seconds:
        idx = rng.integers(0, n, bs)
        h = model.train_on_batch(Xtr[idx], {'pos': Ytr[idx, :3], 'skew': Ytr[idx, 3:4]}, return_dict=True)
        steps += 1
    model.save_weights(CKPT)
    pred = model.predict(Xtr[-512:], verbose=0)
    mae = float(np.mean(np.abs(pred['pos'] - Ytr[-512:, :3])) + np.mean(np.abs(pred['skew'] - Ytr[-512:, 3:4])))
    best = float(np.loadtxt(os.path.join(ROOT, 'model', 'best.txt'))) if os.path.exists(os.path.join(ROOT, 'model', 'best.txt')) else 1e9
    if mae <= best:
        model.save_weights(BEST)
        np.savetxt(os.path.join(ROOT, 'model', 'best.txt'), [mae])
    print(json.dumps({'steps': steps, 'loss': float(h['loss']), 'quick_mae': mae}))

    if a.finalize:
        model.load_weights(BEST)
        Xev, Yev = load_split('eval')
        pred = model.predict(Xev, verbose=0)
        out = {}
        for i, nm in enumerate(['blur', 'contrast_loss', 'noise']):
            p = pred['pos'][:, i]; t = Yev[:, i]
            out[nm] = {'mae': float(np.mean(np.abs(p - t))),
                       'bin_acc@0.15': float(np.mean((p > 0.15) == (t > 0.15)))}
        out['skew_deg'] = {'mae_deg': float(np.mean(np.abs(pred['skew'][:, 0] - Yev[:, 3])) * 8.0)}
        print('EVAL ' + json.dumps(out))
        os.makedirs(os.path.join(ROOT, 'model'), exist_ok=True)
        with open(os.path.join(ROOT, 'model', 'eval.json'), 'w') as f:
            json.dump(out, f, indent=2)
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, os.path.join(ROOT, 'model', 'tfjs'))
        print('tfjs exported')

if __name__ == '__main__':
    main()
