"""Split the fused _FusedMatMul(BiasAdd+Tanh) skew head into
_FusedMatMul(BiasAdd) + standalone Tanh. tfjs 4.22 has no fused-tanh
kernel on any backend (webgl or cpu), but standalone Tanh is supported.
Weights are unchanged, so the shard is untouched.
"""
import json, os

p = os.path.join(os.path.dirname(__file__), '..', 'model', 'tfjs_graph', 'model.json')
j = json.load(open(p))
nodes = j['modelTopology']['node']
for i, n in enumerate(nodes):
    if n['op'] == '_FusedMatMul' and n['name'] == 'StatefulPartitionedCall/Tanh':
        fused = n['attr']['fused_ops']['list']['s']
        assert fused == ['Qmlhc0FkZA==', 'VGFuaA=='], fused  # BiasAdd, Tanh
        n['attr']['fused_ops']['list']['s'] = ['Qmlhc0FkZA==']
        n['name'] = 'StatefulPartitionedCall/functional_1/skew_1/MatMul'
        nodes.insert(i + 1, {
            'name': 'StatefulPartitionedCall/Tanh',
            'op': 'Tanh',
            'input': ['StatefulPartitionedCall/functional_1/skew_1/MatMul'],
            'attr': {'T': {'type': 'DT_FLOAT'}},
        })
        break
else:
    raise SystemExit('fused tanh node not found')
json.dump(j, open(p, 'w'))
print('patched; nodes:', len(nodes))
