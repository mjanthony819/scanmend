"""LEGACY: exports a tfjs LAYERS model (model/tfjs/). Keras 3 layers JSON is not loadable by tfjs 4.22 in the browser (batch_shape/inbound_nodes format). Kept for reference; use export_tfjs_graph.py + patch_graph_tanh.py instead."""
import os
import importlib.util
spec = importlib.util.spec_from_file_location('tm', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_model.py'))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)
import tensorflowjs as tfjs
model = tm.build()
model.load_weights(tm.BEST)
tfjs.converters.save_keras_model(model, os.path.join(tm.ROOT, 'model', 'tfjs'))
print('tfjs exported')
