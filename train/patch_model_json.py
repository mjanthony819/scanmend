"""Patch the Keras-3-format tfjs model.json for tfjs 4.x compatibility.

tensorflowjs 4.22's save_keras_model emits Keras 3 topology where InputLayer
uses `batch_shape`; the tfjs layers loader expects `batchInputShape` (Keras 2
naming). Without this patch the browser load fails with
'An InputLayer should be passed either a batchInputShape or an inputShape'.
Run after train/export_tfjs.py.
"""
import json, os
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'model', 'model.json')
m = json.load(open(p))
cfg = m['modelTopology'].get('model_config', {}).get('config') or m['modelTopology'].get('config')
fixed = 0
for layer in cfg['layers']:
    if layer['class_name'] == 'InputLayer' and 'batch_shape' in layer['config']:
        layer['config']['batchInputShape'] = layer['config'].pop('batch_shape')
        fixed += 1
json.dump(m, open(p, 'w'))
print(f'patched {fixed} InputLayer(s) in {p}')
