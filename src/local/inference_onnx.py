"""
Local ONNX inference for CLIP — compares FP32 vs INT8 Recall@10.

Runs ONNX models via ONNXRuntime and prints Recall@10 scores.

Prerequisites:
  1. Run export_onnx.py       → exported_onnx/image_encoder*.onnx + text_encoder*.onnx
  2. Run quantize_local.py    → exported_onnx/image_encoder*_int8_*.onnx

Usage:
  python inference_onnx_local.py --sweep               # auto-discover all combos, print table
  python inference_onnx_local.py --model ViT-B/16      # all available combos for one model
  python inference_onnx_local.py --mode fp32           # FP32 only
  python inference_onnx_local.py --mode int8           # INT8 only (legacy _int8.onnx or first discovered)
  python inference_onnx_local.py --mode fp32_img_int8_txt
  python inference_onnx_local.py --mode int8_img_fp32_txt
  python inference_onnx_local.py --inspect-embeddings  # print embedding stats for first sample
"""

import sys
import os
import re
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import onnxruntime as ort

sys.path.insert(0, "clip_model")
import clip as clip_lib

from src.common.eval import evaluate_track1
from src.common.config import (
    ONNX_DIR, IMAGE_DIR, IMG_LIST, TXT_LIST, SLUG_TO_MODEL, ensure_output_dirs,
)

ensure_output_dirs()
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
        img_output.append(out[0])
        if inspect and i == 0:
            print(f"\n  --- Embedding inspection (first sample) ---")
            print_embedding_stats(out[0], f"[{label}] image embed[0]")

    print(f"[{label}] Running text encoder on {len(text_tokens)} prompts...")
    txt_output = []
    for i, arr in enumerate(text_tokens):
        out = txt_sess.run(None, {txt_input_name: arr})
        txt_output.append(out[0])
        if inspect and i == 0:
            print_embedding_stats(out[0], f"[{label}] text embed[0]")

    if inspect and img_output and txt_output:
        img_vec = img_output[0].flatten().astype(np.float32)
        txt_vec = txt_output[0].flatten().astype(np.float32)
        img_norm = np.linalg.norm(img_vec)
        txt_norm = np.linalg.norm(txt_vec)
        if img_norm > 0 and txt_norm > 0:
            cos_sim = float(np.dot(img_vec, txt_vec) / (img_norm * txt_norm))
            print(f"  cosine_sim(img[0], txt[0]) = {cos_sim:.4f}")
        print()

    return img_output, txt_output


def discover_int8_combos(model_filter=None):
    """
    Scan exported_onnx/ for image_encoder*_int8_*.onnx files.
    Returns list of (slug, fmt_name, act_name, img_path, txt_path).
    Only includes entries where both image and text encoder files exist.
    """
    # Pattern: image_encoder{slug}_int8_{format}_{activation}.onnx
    pattern = re.compile(r"^image_encoder((?:_vitl14)?(?:_vitb16)?)_int8_([a-z]+)_([a-z0-9]+)\.onnx$")
    combos = []
    if not os.path.isdir(ONNX_DIR):
        return combos

    for fname in sorted(os.listdir(ONNX_DIR)):
        m = pattern.match(fname)
        if not m:
            continue
        slug, fmt_name, act_name = m.group(1), m.group(2), m.group(3)

        # Apply model filter if specified
        if model_filter is not None:
            expected_slug = "" if model_filter == "ViT-B/16" else "_vitl14"
            if slug != expected_slug:
                continue

        txt_fname = f"text_encoder{slug}_int8_{fmt_name}_{act_name}.onnx"
        img_path = os.path.join(ONNX_DIR, fname)
        txt_path = os.path.join(ONNX_DIR, txt_fname)

        if not os.path.exists(txt_path):
            print(f"Warning: found {fname} but missing {txt_fname} — skipping.")
            continue

        combos.append((slug, fmt_name, act_name, img_path, txt_path))

    return combos


def print_summary_table(results):
    """
    Print a formatted summary table.
    results: list of dicts with keys: model, format, activation, recall, fp32_recall
    """
    print("\n" + "=" * 72)
    print("Recall@10 Benchmark Summary")
    print("=" * 72)
    print(f"  {'Model':<10} {'Format':<12} {'Activation':<12} {'Recall@10':>10} {'vs FP32':>9}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*9}")

    for r in results:
        delta_str = ""
        if r.get("fp32_recall") is not None and r["format"] != "FP32":
            delta = r["recall"] - r["fp32_recall"]
            delta_str = f"{delta:+.4f}"
        print(f"  {r['model']:<10} {r['format']:<12} {r['activation']:<12} "
              f"{r['recall']:>10.4f} {delta_str:>9}")

    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Local Inference: FP32 vs INT8 benchmark")
    parser.add_argument(
        "--model", default=None, choices=["ViT-B/16", "ViT-L/14"],
        help="CLIP model variant. If omitted with --sweep, runs all discovered models.",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Auto-discover all INT8 combo files in exported_onnx/ and benchmark them all.",
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["all", "fp32", "int8", "fp32_img_int8_txt", "int8_img_fp32_txt"],
        help=(
            "Which encoder combination(s) to run (used without --sweep):\n"
            "  all               — all available combinations (default)\n"
            "  fp32              — FP32 image + FP32 text\n"
            "  int8              — INT8 image + INT8 text\n"
            "  fp32_img_int8_txt — FP32 image + INT8 text\n"
            "  int8_img_fp32_txt — INT8 image + FP32 text"
        ),
    )
    parser.add_argument(
        "--inspect-embeddings", action="store_true",
        help="Print dtype/shape/min/max/norm for the first embedding of each encoder",
    )
    args = parser.parse_args()

    inspect = args.inspect_embeddings

    print("Loading data...")
    images = load_images()
    text_tokens = load_text_tokens()
    print(f"  {len(images)} images, {len(text_tokens)} text prompts loaded.\n")

    # -----------------------------------------------------------------------
    # SWEEP MODE: auto-discover all available INT8 combos
    # -----------------------------------------------------------------------
    if args.sweep or (args.model is None and args.mode == "all"):
        discovered = discover_int8_combos(model_filter=args.model)

        if not discovered:
            print("No INT8 combo files found in exported_onnx/.")
            print("Run: python quantize_local.py")
            sys.exit(1)

        print(f"Discovered {len(discovered)} INT8 combo(s).\n")

        # FP32 baselines — run once per slug, cache
        fp32_recalls = {}
        slugs_seen = sorted({slug for slug, *_ in discovered})
        for slug in slugs_seen:
            model_name = SLUG_TO_MODEL.get(slug, slug or "ViT-B/16")
            img_fp32 = os.path.join(ONNX_DIR, f"image_encoder{slug}.onnx")
            txt_fp32 = os.path.join(ONNX_DIR, f"text_encoder{slug}.onnx")
            if os.path.exists(img_fp32) and os.path.exists(txt_fp32):
                label = f"{model_name} FP32"
                img_out, txt_out = run_inference(img_fp32, txt_fp32, images, text_tokens,
                                                  label, inspect=inspect)
                fp32_recalls[slug] = evaluate_track1(img_out, txt_out, TXT_LIST, IMG_LIST)
                print(f"  FP32 Recall@10: {fp32_recalls[slug]:.4f}")
            else:
                fp32_recalls[slug] = None
                print(f"Warning: FP32 models for slug='{slug}' not found — skipping baseline.")

        # Run all discovered INT8 combos
        all_results = []

        # Add FP32 rows first
        for slug in slugs_seen:
            model_name = SLUG_TO_MODEL.get(slug, slug or "ViT-B/16")
            if fp32_recalls.get(slug) is not None:
                all_results.append({
                    "model": model_name, "format": "FP32", "activation": "—",
                    "recall": fp32_recalls[slug], "fp32_recall": None,
                })

        for slug, fmt_name, act_name, img_path, txt_path in discovered:
            model_name = SLUG_TO_MODEL.get(slug, slug or "ViT-B/16")
            label = f"{model_name} {fmt_name} {act_name}"
            img_out, txt_out = run_inference(img_path, txt_path, images, text_tokens,
                                              label, inspect=inspect)
            recall = evaluate_track1(img_out, txt_out, TXT_LIST, IMG_LIST)
            print(f"  Recall@10: {recall:.4f}")
            all_results.append({
                "model": model_name,
                "format": fmt_name,
                "activation": act_name,
                "recall": recall,
                "fp32_recall": fp32_recalls.get(slug),
            })

        print_summary_table(all_results)
        sys.exit(0)

    # -----------------------------------------------------------------------
    # LEGACY / TARGETED MODE: --model + --mode
    # -----------------------------------------------------------------------
    model_name = args.model or "ViT-B/16"
    slug = "" if model_name == "ViT-B/16" else "_vitl14"

    IMAGE_FP32 = os.path.join(ONNX_DIR, f"image_encoder{slug}.onnx")
    TEXT_FP32  = os.path.join(ONNX_DIR, f"text_encoder{slug}.onnx")

    # For legacy INT8 path: prefer new-style naming, fall back to old _int8.onnx
    # Try to find at least one INT8 combo (first available)
    discovered_for_model = discover_int8_combos(model_filter=model_name)
    if discovered_for_model:
        _, first_fmt, first_act, IMAGE_INT8, TEXT_INT8 = discovered_for_model[0]
    else:
        IMAGE_INT8 = os.path.join(ONNX_DIR, f"image_encoder{slug}_int8.onnx")
        TEXT_INT8  = os.path.join(ONNX_DIR, f"text_encoder{slug}_int8.onnx")

    print(f"=== ONNX Local Inference: {model_name} | mode={args.mode} ===\n")

    fp32_available = os.path.exists(IMAGE_FP32) and os.path.exists(TEXT_FP32)
    int8_available = os.path.exists(IMAGE_INT8) and os.path.exists(TEXT_INT8)

    if not fp32_available:
        print("Warning: FP32 ONNX models not found. Run export_onnx.py first.")
    if not int8_available:
        print("Warning: INT8 ONNX models not found. Run quantize_local.py first.")
    if not fp32_available and not int8_available:
        sys.exit(1)

    needs_fp32 = args.mode in ("all", "fp32", "fp32_img_int8_txt", "int8_img_fp32_txt")
    needs_int8 = args.mode in ("all", "int8", "fp32_img_int8_txt", "int8_img_fp32_txt")
    if needs_fp32 and not fp32_available:
        print(f"Error: mode '{args.mode}' requires FP32 models but they were not found.")
        sys.exit(1)
    if needs_int8 and not int8_available:
        print(f"Error: mode '{args.mode}' requires INT8 models but they were not found.")
        sys.exit(1)

    combos = []
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

    # Summary
    print("\n" + "=" * 50)
    print(f"Recall@10 Results — {model_name}")
    print("=" * 50)
    for variant, recall in results.items():
        print(f"  {variant:<26} {recall:.4f}")

    if "FP32" in results and "INT8" in results:
        delta = results["INT8"] - results["FP32"]
        sign = "+" if delta >= 0 else ""
        print(f"  {'Delta (INT8 vs FP32)':<26} {sign}{delta:.4f}")
    print()
