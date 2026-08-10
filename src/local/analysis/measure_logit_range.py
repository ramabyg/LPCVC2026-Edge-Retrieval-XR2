"""
Measure the pre-mask attention-logit range in the FP32 CLIP text encoder.

Exposes, for each of the 12 transformer layers, the two tensors that matter for
quantization:
    pre-mask  = MatMul output feeding Add(mask)   -> the real logit distribution
    post-mask = Add output feeding Softmax        -> what a quantizer would see

Then runs all 211 calibration prompts and reports min/max/percentiles.
Read-only: writes an instrumented copy to scratchpad, never touches exported_onnx/.
"""
import os, sys, numpy as np, onnx, onnxruntime as ort
from onnx import helper, numpy_helper
sys.path.insert(0, os.getcwd())
from src.local.inference_onnx import load_text_tokens

import tempfile
# Instrumented copies are ~255 MB — keep them out of the repo tree.
SCRATCH = tempfile.mkdtemp(prefix="lpcvc_logit_range_")
SRC = "exported_onnx/text_encoder.onnx"
INSTRUMENTED = os.path.join(SCRATCH, "text_encoder_instrumented.onnx")

m = onnx.load(SRC)
g = m.graph

# 1. locate the -inf mask initializers
mask_names = set()
for init in g.initializer:
    a = numpy_helper.to_array(init)
    if a.dtype.kind == "f" and a.size and not np.isfinite(a).all():
        mask_names.add(init.name)
print(f"mask initializers: {len(mask_names)}")

# 2. find Add(mask) nodes -> (pre_mask_tensor, post_mask_tensor)
pairs = []
for n in g.node:
    hit = mask_names & set(n.input)
    if not hit:
        continue
    assert n.op_type == "Add", f"unexpected mask consumer {n.op_type}"
    pre = [i for i in n.input if i not in mask_names]
    assert len(pre) == 1
    pairs.append((pre[0], n.output[0]))
pairs.sort(key=lambda p: int(p[0].split("_")[-1]))
print(f"mask Add nodes: {len(pairs)}")

# 3. expose them as graph outputs
existing = {o.name for o in g.output}
for pre, post in pairs:
    for t in (pre, post):
        if t not in existing:
            g.output.append(helper.make_empty_tensor_value_info(t))
            existing.add(t)

onnx.save(m, INSTRUMENTED, save_as_external_data=True,
          all_tensors_to_one_file=True, location=os.path.basename(INSTRUMENTED) + ".data")

# 4. run with graph optimization DISABLED so nodes aren't fused away
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(INSTRUMENTED, sess_options=so)
inp = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]
idx = {n: i for i, n in enumerate(out_names)}

toks = load_text_tokens()
print(f"running {len(toks)} prompts...\n")

pre_samples = [[] for _ in pairs]
stats = [dict(lo=np.inf, hi=-np.inf) for _ in pairs]
post_lo, post_hi = np.inf, -np.inf

for k, t in enumerate(toks):
    outs = sess.run(out_names, {inp: t})
    for li, (pre, post) in enumerate(pairs):
        a = outs[idx[pre]].astype(np.float32).ravel()
        stats[li]["lo"] = min(stats[li]["lo"], float(a.min()))
        stats[li]["hi"] = max(stats[li]["hi"], float(a.max()))
        if k % 8 == 0:                      # subsample for percentiles
            pre_samples[li].append(a[::7])
        b = outs[idx[post]].astype(np.float32).ravel()
        b = b[np.isfinite(b)]
        if b.size:
            post_lo = min(post_lo, float(b.min())); post_hi = max(post_hi, float(b.max()))

print(f"{'layer':>5} {'L_min':>10} {'L_max':>10} {'spread D':>10} {'p0.1':>9} {'p99.9':>9}")
print("-" * 60)
glo, ghi = np.inf, -np.inf
for li, s in enumerate(stats):
    v = np.concatenate(pre_samples[li])
    p_lo, p_hi = np.percentile(v, 0.1), np.percentile(v, 99.9)
    print(f"{li:>5} {s['lo']:>10.3f} {s['hi']:>10.3f} {s['hi']-s['lo']:>10.3f} {p_lo:>9.3f} {p_hi:>9.3f}")
    glo, ghi = min(glo, s["lo"]), max(ghi, s["hi"])

D = ghi - glo
print("-" * 60)
print(f"GLOBAL pre-mask logits:  L_min={glo:.3f}  L_max={ghi:.3f}  spread D={D:.3f}")
print(f"post-mask (finite entries only): [{post_lo:.3f}, {post_hi:.3f}]")

print("\n--- additive mask: span S = D + |M|, step = S/255 ---")
print(f"{'M':>10} {'exp(M)*77':>14} {'span S':>10} {'step':>9}  verdict")
for M in (-10, -15, -20, -25, -30, -40, -50, -100, -1e4):
    leak = 77 * np.exp(M)
    S = D + abs(M)
    step = S / 255
    ok = "OK" if (leak < 1e-6 and step < 0.25) else ("mask leaks" if leak >= 1e-6 else "too coarse")
    print(f"{M:>10.0f} {leak:>14.2e} {S:>10.1f} {step:>9.4f}  {ok}")
