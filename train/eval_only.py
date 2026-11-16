"""Held-out eval only: load BEST weights, predict eval shards, write model/eval.json."""
import os, json
import numpy as np
import tensorflow as tf
import importlib.util
spec = importlib.util.spec_from_file_location('tm', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_model.py'))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)  # safe: main() is __main__-guarded

model = tm.build()
model.load_weights(tm.BEST)
Xev, Yev = tm.load_split('eval')
pred = model.predict(Xev, batch_size=64, verbose=0)
out = {}
for i, nm in enumerate(['blur', 'contrast_loss', 'noise']):
    p = pred['pos'][:, i]; t = Yev[:, i]
    out[nm] = {'mae': float(np.mean(np.abs(p - t))),
               'bin_acc@0.15': float(np.mean((p > 0.15) == (t > 0.15)))}
out['skew_deg'] = {'mae_deg': float(np.mean(np.abs(pred['skew'][:, 0] - Yev[:, 3])) * 8.0)}
with open(os.path.join(tm.ROOT, 'model', 'eval.json'), 'w') as f:
    json.dump(out, f, indent=2)
print('EVAL ' + json.dumps(out))
