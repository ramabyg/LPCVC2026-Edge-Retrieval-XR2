"""Does CLE alone (no quantization) change FP32 accuracy? Decisive test for the ViT+CLE hypothesis."""
import os, sys, tempfile, numpy as np, onnx, onnxruntime as ort
sys.path.insert(0, os.getcwd())
from src.common.eval import evaluate_track1
from src.common.config import ONNX_DIR, IMG_LIST, TXT_LIST
from src.local.inference_onnx import load_images, load_text_tokens
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from aimet_onnx.cross_layer_equalization import equalize_model

# Equalized copies are ~600 MB combined — keep them out of the repo tree.
SCRATCH = tempfile.mkdtemp(prefix="lpcvc_cle_check_")

images, toks = load_images(), load_text_tokens()
outs = {}
for kind, fname, inp in [("image", "image_encoder.onnx", images), ("text", "text_encoder.onnx", toks)]:
    m = onnx.load(os.path.join(ONNX_DIR, fname))
    print(f"[{kind}] BN fold...", flush=True)
    pairs = fold_all_batch_norms_to_weight(m)
    print(f"[{kind}] folded pairs: {pairs}", flush=True)
    print(f"[{kind}] CLE...", flush=True)
    equalize_model(m)
    p = os.path.join(SCRATCH, f"{kind}_cle.onnx")
    onnx.save(m, p, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(p)+".data")
    s = ort.InferenceSession(p)
    n = s.get_inputs()[0].name
    outs[kind] = [s.run(None, {n: a})[0] for a in inp]
    print(f"[{kind}] done", flush=True)

r = evaluate_track1(outs["image"], outs["text"], TXT_LIST, IMG_LIST)
print(f"\nFP32 + BNfold + CLE (NO quantization) Recall@10 = {r:.4f}   (FP32 baseline 0.8728)")
