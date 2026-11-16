"""Unfuse _FusedConv2D/_FusedMatMul nodes in the tfjs graph model.json into
plain Conv2D/MatMul + BiasAdd + activation chains. tfjs 4.22's fused executors
silently ignore the x input for this graph (constant output for any input);
plain ops are the heavily tested path. Weights unchanged; shard untouched.
Run AFTER export_tfjs_graph.py and patch_graph_tanh.py.
"""
import json, os

p = os.path.join(os.path.dirname(__file__), '..', 'model', 'tfjs_graph', 'model.json')
j = json.load(open(p))
nodes = j['modelTopology']['node']

BIASADD_S = 'Qmlhc0FkZA=='
def act_name(b64):
    return {'Qmlhc0FkZA==': 'BiasAdd', 'UmVsdQ==': 'Relu', 'U2lnbW9pZA==': 'Sigmoid', 'VGFuaA==': 'Tanh'}[b64]

out = []
for n in nodes:
    if n['op'] == '_FusedConv2D':
        fused = [act_name(s) for s in n['attr']['fused_ops']['list']['s']]
        assert fused[0] == 'BiasAdd', fused
        x, filt, bias = n['input']
        base = n['name']
        conv_attrs = {k: n['attr'][k] for k in ('T', 'strides', 'padding', 'dilations', 'data_format', 'explicit_paddings', 'use_cudnn_on_gpu') if k in n['attr']}
        out.append({'name': base + '/conv', 'op': 'Conv2D', 'input': [x, filt], 'attr': conv_attrs})
        out.append({'name': base + '/bias', 'op': 'BiasAdd', 'input': [base + '/conv', bias], 'attr': {'T': {'type': 'DT_FLOAT'}}})
        if len(fused) > 1:
            out.append({'name': base, 'op': fused[1], 'input': [base + '/bias'], 'attr': {'T': {'type': 'DT_FLOAT'}}})
        else:
            out.append({'name': base, 'op': 'Identity', 'input': [base + '/bias'], 'attr': {'T': {'type': 'DT_FLOAT'}}})
    elif n['op'] == '_FusedMatMul':
        fused = [act_name(s) for s in n['attr']['fused_ops']['list']['s']]
        assert fused[0] == 'BiasAdd', fused
        x, w, bias = n['input']
        base = n['name']
        mm_attrs = {k: n['attr'][k] for k in ('T', 'transpose_a', 'transpose_b') if k in n['attr']}
        out.append({'name': base + '/matmul', 'op': 'MatMul', 'input': [x, w], 'attr': mm_attrs})
        out.append({'name': base + '/bias', 'op': 'BiasAdd', 'input': [base + '/matmul', bias], 'attr': {'T': {'type': 'DT_FLOAT'}}})
        if len(fused) > 1:
            out.append({'name': base, 'op': fused[1], 'input': [base + '/bias'], 'attr': {'T': {'type': 'DT_FLOAT'}}})
        else:
            out.append({'name': base, 'op': 'Identity', 'input': [base + '/bias'], 'attr': {'T': {'type': 'DT_FLOAT'}}})
    else:
        out.append(n)

j['modelTopology']['node'] = out
json.dump(j, open(p, 'w'))
ops = {}
for n in out: ops[n['op']] = ops.get(n['op'], 0) + 1
print('unfused; nodes:', len(out), ops)
