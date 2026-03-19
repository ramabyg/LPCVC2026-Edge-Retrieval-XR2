"""
Local ONNX inference for CLIP — compares FP32 vs INT8 Recall@10.

Runs both the FP32 and INT8 ONNX models via ONNXRuntime and prints
side-by-side Recall@10 scores.

Prerequisites:
  1. Run export_onnx.py       → exported_onnx/image_encoder.onnx + text_encoder.onnx
  2. Run quantize_local.py    → exported_onnx/image_encoder_int8.onnx + text_encoder_int8.onnx

Usage:
  python inference_onnx_local.py
"""

import sys
import os
import numpy as np
import pandas as pd
from PIL import Image

import onnxruntime as ort

sys.path.insert(0, "clip_model")
import clip as clip_lib

from inference import evaluate_track1

# --- Configuration ---
ONNX_DIR = "exported_onnx"
IMAGE_FP32  = os.path.join(ONNX_DIR, "image_encoder.onnx")
TEXT_FP32   = os.path.join(ONNX_DIR, "text_encoder.onnx")
IMAGE_INT8  = os.path.join(ONNX_DIR, "image_encoder_int8.onnx")
TEXT_INT8   = os.path.join(ONNX_DIR, "text_encoder_int8.onnx")

DATA_DIR  = r"C:\rama\projects\data\lpcvc_track1_sample_data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
IMG_LIST  = os.path.join(DATA_DIR, "img_list.csv")
TXT_LIST  = os.path.join(DATA_DIR, "txt_list.csv")
# ---------------------


def load_images():
    """Load all images with competition-style /255 preprocessing."""
    df = pd.read_csv(IMG_LIST)
    filenames = df.iloc[:, 0].tolist()
    images = []
    for f in filenames:
        img = Image.open(os.path.join(IMAGE_DIR, f)).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, :]  # (1, 3, 224, 224)
        images.append(arr)
    return images


def load_text_tokens():
    """Tokenize all text prompts."""
    df = pd.read_csv(TXT_LIST)
    prompts = df.iloc[:, 1].dropna().tolist()
    import torch
    tokens = clip_lib.tokenize(prompts)  # (M, 77) int64
    return [tokens[i:i+1].numpy().astype(np.int64) for i in range(len(tokens))]


def run_inference(img_model_path, txt_model_path, images, text_tokens, label):
    """Run ONNX inference and return (img_embeddings, txt_embeddings)."""
    print(f"\n[{label}] Loading ONNX sessions...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    img_sess = ort.InferenceSession(img_model_path, sess_options=sess_options)
    txt_sess = ort.InferenceSession(txt_model_path, sess_options=sess_options)

    img_input_name = img_sess.get_inputs()[0].name
    txt_input_name = txt_sess.get_inputs()[0].name

    print(f"[{label}] Running image encoder on {len(images)} images...")
    img_output = []
    for arr in images:
        out = img_sess.run(None, {img_input_name: arr})
        img_output.append(out[0])  # (1, 512)

    print(f"[{label}] Running text encoder on {len(text_tokens)} prompts...")
    txt_output = []
    for arr in text_tokens:
        out = txt_sess.run(None, {txt_input_name: arr})
        txt_output.append(out[0])  # (1, 512)

    return img_output, txt_output


if __name__ == "__main__":
    print("=== ONNX Local Inference: FP32 vs INT8 ===\n")

    # Check which models are available
    fp32_available = os.path.exists(IMAGE_FP32) and os.path.exists(TEXT_FP32)
    int8_available = os.path.exists(IMAGE_INT8) and os.path.exists(TEXT_INT8)

    if not fp32_available:
        print("Warning: FP32 ONNX models not found. Run export_onnx.py first.")
    if not int8_available:
        print("Warning: INT8 ONNX models not found. Run quantize_local.py first.")
    if not fp32_available and not int8_available:
        sys.exit(1)

    print("Loading data...")
    images = load_images()
    text_tokens = load_text_tokens()
    print(f"  {len(images)} images, {len(text_tokens)} text prompts loaded.")

    results = {}

    if fp32_available:
        img_out, txt_out = run_inference(IMAGE_FP32, TEXT_FP32, images, text_tokens, "FP32")
        recall = evaluate_track1(img_out, txt_out, TXT_LIST, IMG_LIST)
        results["FP32"] = recall

    if int8_available:
        img_out, txt_out = run_inference(IMAGE_INT8, TEXT_INT8, images, text_tokens, "INT8")
        recall = evaluate_track1(img_out, txt_out, TXT_LIST, IMG_LIST)
        results["INT8"] = recall

    # --- Summary ---
    print("\n" + "=" * 40)
    print("Recall@10 Results")
    print("=" * 40)
    for variant, recall in results.items():
        print(f"  {variant:<6} {recall:.4f}")

    if "FP32" in results and "INT8" in results:
        delta = results["INT8"] - results["FP32"]
        sign = "+" if delta >= 0 else ""
        print(f"  Delta  {sign}{delta:.4f}  (INT8 vs FP32)")
    print()
