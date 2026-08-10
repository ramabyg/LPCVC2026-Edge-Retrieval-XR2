"""
Directly measure FP32 Recall@10 as a function of the causal-mask constant M.

Replaces the 12 baked -inf mask initializers with a finite M and evaluates.
This settles 'is masking still correct at M?' by measurement instead of by a
worst-case bound. Quantization resolution is a separate, arithmetic question.

Image embeddings are computed once and reused (image encoder is untouched).
"""
import os, sys, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper
sys.path.insert(0, os.getcwd())
from src.common.eval import evaluate_track1
from src.common.config import ONNX_DIR, IMG_LIST, TXT_LIST
from src.local.inference_onnx import load_images, load_text_tokens

import tempfile
# Each patched text encoder is ~255 MB — keep them out of the repo tree.
SCRATCH = tempfile.mkdtemp(prefix="lpcvc_mask_sweep_")
images, toks = load_images(), load_text_tokens()

# image side: unchanged, compute once
s = ort.InferenceSession(os.path.join(ONNX_DIR, "image_encoder.onnx"))
n = s.get_inputs()[0].name
img_out = [s.run(None, {n: a})[0] for a in images]
print(f"image embeddings done ({len(img_out)})\n")

base = onnx.load(os.path.join(ONNX_DIR, "text_encoder.onnx"))
mask_names = [i.name for i in base.graph.initializer
              if numpy_helper.to_array(i).dtype.kind == "f"
              and numpy_helper.to_array(i).size
              and not np.isfinite(numpy_helper.to_array(i)).all()]
print(f"patching {len(mask_names)} mask initializers per run\n")

print(f"{'M':>10} {'Recall@10':>11} {'vs FP32':>9}   {'max|attn err|':>13}")
print("-" * 52)
BASE_RECALL = 0.8728

for M in (-10, -15, -20, -25, -30, -40, -50, -60, -100):
    m = onnx.load(os.path.join(ONNX_DIR, "text_encoder.onnx"))
    for init in m.graph.initializer:
        if init.name in mask_names:
            a = numpy_helper.to_array(init).copy()
            a[~np.isfinite(a)] = M
            init.CopyFrom(numpy_helper.from_array(a, init.name))
    p = os.path.join(SCRATCH, f"txt_M{abs(M)}.onnx")
    onnx.save(m, p, save_as_external_data=True, all_tensors_to_one_file=True,
              location=os.path.basename(p) + ".data")
    ss = ort.InferenceSession(p)
    nn = ss.get_inputs()[0].name
    txt_out = [ss.run(None, {nn: t})[0] for t in toks]
    r = evaluate_track1(img_out, txt_out, TXT_LIST, IMG_LIST)

    # worst-case spurious attention mass on the hardest row (row 0: 1 unmasked, 76 masked)
    D = 26.746
    leak = 76 * np.exp(D + M)
    print(f"{M:>10} {r:>11.4f} {r-BASE_RECALL:>+9.4f}   {leak:>13.2e}")
    os.remove(p); os.remove(p + ".data")
