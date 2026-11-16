"""Export BEST weights as a tfjs GRAPH model (SavedModel -> tfjs_graph_model).

The Keras 3 layers-model JSON that save_keras_model emits is not readable by
tfjs 4.x in the browser (InputLayer batch_shape / inbound_nodes nodeData
format). Exporting via SavedModel sidesteps Keras topology entirely: the app
loads it with tf.loadGraphModel and finds outputs by shape (3 = severities,
1 = skew).
"""
import os
import importlib.util
spec = importlib.util.spec_from_file_location('tm', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_model.py'))
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)
import tensorflow as tf
import tensorflowjs as tfjs

# Skew head is exported LINEAR and tanh is applied as a standalone op in the
# wrapper below: tfjs (4.22) has no fused-tanh kernel for _FusedMatMul on any
# backend, but standalone Tanh works. Same math, same weights.
model = tm.build()
model.get_layer('skew').activation = tf.keras.activations.linear
model.load_weights(tm.BEST)

class Diagnose(tf.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    @tf.function(input_signature=[tf.TensorSpec([None, 160, 160, 1], tf.float32, name='document')])
    def __call__(self, x):
        out = self.m(x, training=False)
        return {'pos': out['pos'], 'skew': tf.tanh(out['skew'])}

root = tm.ROOT
sm_dir = os.path.join(root, 'model', 'savedmodel')
tf.saved_model.save(Diagnose(model), sm_dir)
tfjs.converters.convert_tf_saved_model(sm_dir, os.path.join(root, 'model', 'tfjs_graph'))
print('graph model exported')
print('IMPORTANT: now run patch_graph_tanh.py - the converter re-fuses tanh into _FusedMatMul, which tfjs 4.22 cannot run')
