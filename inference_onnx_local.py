"""
Local ONNX inference for CLIP — compares FP32 vs INT8 Recall@10.

Runs both the FP32 and INT8 ONNX models via ONNXRuntime and prints
side-by-side Recall@10 scores.

Prerequisites:
  1. Run export_onnx.py       → exported_onnx/image_encoder.onnx + text_encoder.onnx
  2. Run quantize_local.py    → exported_onnx/image_encoder_int8.onnx + text_encoder_int8.onnx

Usage:
  python inference_onnx_local.py                        # run all available combinations
  python inference_onnx_local.py --mode fp32            # FP32 only
  python inference_onnx_local.py --mode int8            # INT8 only
  python inference_onnx_local.py --mode fp32_img_int8_txt  # cross: FP32 image + INT8 text
  python inference_onnx_local.py --mode int8_img_fp32_txt  # cross: INT8 image + FP32 text
  python inference_onnx_local.py --inspect-embeddings   # print embedding stats for first sample
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import onnxruntime as ort

sys.path.insert(0, "clip_model")
import clip as clip_lib

from inference import evaluate_track1

# --- Configuration (paths set after argparse below) ---
ONNX_DIR = "exported_onnx"

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


def print_embedding_stats(embed, label):
    """Print diagnostic stats for an embedding array (shape, dtype, range, norm)."""
    arr = embed.flatten().astype(np.float32)
    norm = float(np.linalg.norm(arr))
    print(f"  {label:<30} dtype={embed.dtype}  shape={embed.shape}  "
          f"min={arr.min():.3f}  max={arr.max():.3f}  norm={norm:.3f}")


def run_inference(img_model_path, txt_model_path, images, text_tokens, label,
                  inspect=False):
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
    for i, arr in enumerate(images):
        out = img_sess.run(None, {img_input_name: arr})
        img_output.append(out[0])  # (1, 512)
        if inspect and i == 0:
            print(f"\n  --- Embedding inspection (first sample) ---")
            print_embedding_stats(out[0], f"[{label}] image embed[0]")

    print(f"[{label}] Running text encoder on {len(text_tokens)} prompts...")
    txt_output = []
    for i, arr in enumerate(text_tokens):
        out = txt_sess.run(None, {txt_input_name: arr})
        txt_output.append(out[0])  # (1, 512)
        if inspect and i == 0:
            print_embedding_stats(out[0], f"[{label}] text embed[0]")

    if inspect and img_output and txt_output:
        # Cosine similarity between first image and first text embedding
        img_vec = img_output[0].flatten().astype(np.float32)
        txt_vec = txt_output[0].flatten().astype(np.float32)
        img_norm = np.linalg.norm(img_vec)
        txt_norm = np.linalg.norm(txt_vec)
        if img_norm > 0 and txt_norm > 0:
            cos_sim = float(np.dot(img_vec, txt_vec) / (img_norm * txt_norm))
            print(f"  cosine_sim(img[0], txt[0]) = {cos_sim:.4f}")
        print()

    return img_output, txt_output


def make_session(model_path):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=sess_options)


def run_cross_inference(img_model_path, txt_model_path, images, text_tokens, label,
                        inspect=False):
    """Same as run_inference but with explicit path args (used for cross combos)."""
    return run_inference(img_model_path, txt_model_path, images, text_tokens, label,
                         inspect=inspect)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Local Inference: FP32 vs INT8")
    parser.add_argument(
        "--model", default="ViT-B/16", choices=["ViT-B/16", "ViT-L/14"],
        help="CLIP model variant to evaluate (default: ViT-B/16)",
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["all", "fp32", "int8", "fp32_img_int8_txt", "int8_img_fp32_txt"],
        help=(
            "Which encoder combination(s) to run:\n"
            "  all               — all available combinations (default)\n"
            "  fp32              — FP32 image + FP32 text\n"
            "  int8              — INT8 image + INT8 text\n"
            "  fp32_img_int8_txt — FP32 image + INT8 text (cross check)\n"
            "  int8_img_fp32_txt — INT8 image + FP32 text (cross check)"
        ),
    )
    parser.add_argument(
        "--inspect-embeddings", action="store_true",
        help="Print dtype/shape/min/max/norm for the first embedding of each encoder",
    )
    args = parser.parse_args()

    inspect = args.inspect_embeddings

    # Derive paths from --model
    if args.model == "ViT-B/16":
        slug = ""
    else:
        slug = "_" + args.model.lower().replace("/", "").replace("-", "")  # "_vitl14"

    IMAGE_FP32 = os.path.join(ONNX_DIR, f"image_encoder{slug}.onnx")
    TEXT_FP32  = os.path.join(ONNX_DIR, f"text_encoder{slug}.onnx")
    IMAGE_INT8 = os.path.join(ONNX_DIR, f"image_encoder{slug}_int8.onnx")
    TEXT_INT8  = os.path.join(ONNX_DIR, f"text_encoder{slug}_int8.onnx")

    print(f"=== ONNX Local Inference: FP32 vs INT8 ({args.model}) ===\n")

    fp32_available = os.path.exists(IMAGE_FP32) and os.path.exists(TEXT_FP32)
    int8_available = os.path.exists(IMAGE_INT8) and os.path.exists(TEXT_INT8)

    if not fp32_available:
        print("Warning: FP32 ONNX models not found. Run export_onnx.py first.")
    if not int8_available:
        print("Warning: INT8 ONNX models not found. Run quantize_local.py first.")
    if not fp32_available and not int8_available:
        sys.exit(1)

    # Validate requested mode has the required models
    needs_fp32 = args.mode in ("all", "fp32", "fp32_img_int8_txt", "int8_img_fp32_txt")
    needs_int8 = args.mode in ("all", "int8", "fp32_img_int8_txt", "int8_img_fp32_txt")
    if needs_fp32 and not fp32_available:
        print(f"Error: mode '{args.mode}' requires FP32 models but they were not found.")
        sys.exit(1)
    if needs_int8 and not int8_available:
        print(f"Error: mode '{args.mode}' requires INT8 models but they were not found.")
        sys.exit(1)

    print("Loading data...")
    images = load_images()
    text_tokens = load_text_tokens()
    print(f"  {len(images)} images, {len(text_tokens)} text prompts loaded.")

    # Define which combinations to run
    combos = []  # list of (img_path, txt_path, label)

    if args.mode == "all":
        if fp32_available:
            combos.append((IMAGE_FP32, TEXT_FP32, "FP32"))
        if int8_available:
            combos.append((IMAGE_INT8, TEXT_INT8, "INT8"))
        if fp32_available and int8_available:
            combos.append((IMAGE_FP32, TEXT_INT8, "FP32_img+INT8_txt"))
            combos.append((IMAGE_INT8, TEXT_FP32, "INT8_img+FP32_txt"))
    elif args.mode == "fp32":
        combos.append((IMAGE_FP32, TEXT_FP32, "FP32"))
    elif args.mode == "int8":
        combos.append((IMAGE_INT8, TEXT_INT8, "INT8"))
    elif args.mode == "fp32_img_int8_txt":
        combos.append((IMAGE_FP32, TEXT_INT8, "FP32_img+INT8_txt"))
    elif args.mode == "int8_img_fp32_txt":
        combos.append((IMAGE_INT8, TEXT_FP32, "INT8_img+FP32_txt"))

    results = {}
    for img_path, txt_path, label in combos:
        img_out, txt_out = run_inference(img_path, txt_path, images, text_tokens,
                                         label, inspect=inspect)
        recall = evaluate_track1(img_out, txt_out, TXT_LIST, IMG_LIST)
        results[label] = recall

    # --- Summary ---
    print("\n" + "=" * 50)
    print("Recall@10 Results")
    print("=" * 50)
    for variant, recall in results.items():
        print(f"  {variant:<26} {recall:.4f}")

    if "FP32" in results and "INT8" in results:
        delta = results["INT8"] - results["FP32"]
        sign = "+" if delta >= 0 else ""
        print(f"  {'Delta (INT8 vs FP32)':<26} {sign}{delta:.4f}")
    print()
