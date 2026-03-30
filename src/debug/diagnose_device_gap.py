"""
Diagnose the local-vs-device Recall@10 gap for CLIP on QAI Hub.

Compares local ONNX FP32 embeddings with on-device embeddings to identify
the source of the ~15% Recall@10 gap (0.8728 local vs 0.7299 on-device).

Usage:
  # Step 1: Run local ONNX inference and save embeddings
  python diagnose_device_gap.py --save-local

  # Step 2: Download device embeddings and compare
  python diagnose_device_gap.py --compare --image-job-id XXX --text-job-id YYY

  # Both in one shot
  python diagnose_device_gap.py --save-local --compare --image-job-id XXX --text-job-id YYY
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import clip as clip_lib

from src.common.eval import evaluate_track1
from src.common.config import (
    ONNX_DIR, DIAGNOSTICS_DIR as DIAG_DIR,
    IMAGE_DIR, IMG_LIST, TXT_LIST,
    ensure_output_dirs,
)

ensure_output_dirs()

LOCAL_IMG_PATH = os.path.join(DIAG_DIR, "local_img_embeddings.npy")
LOCAL_TXT_PATH = os.path.join(DIAG_DIR, "local_txt_embeddings.npy")


def load_images():
    """Load all images with competition-style /255 preprocessing.

    Competition sends images resized to 224x224, divided by 255 only.
    CLIP normalization is baked into the model's ImageEncoderWrapper.
    """
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
    """Tokenize all text prompts using CLIP tokenizer."""
    df = pd.read_csv(TXT_LIST)
    prompts = df.iloc[:, 1].dropna().tolist()
    import torch
    tokens = clip_lib.tokenize(prompts)  # (M, 77) int64
    return [tokens[i : i + 1].numpy().astype(np.int64) for i in range(len(tokens))]


def run_local_onnx(images, text_tokens):
    """Run local ONNX FP32 inference and return (img_embeddings, txt_embeddings).

    Returns:
        img_output: list of np.ndarray, each shape (1, 512)
        txt_output: list of np.ndarray, each shape (1, 512)
    """
    import onnxruntime as ort

    img_model = os.path.join(ONNX_DIR, "image_encoder.onnx")
    txt_model = os.path.join(ONNX_DIR, "text_encoder.onnx")

    for path in [img_model, txt_model]:
        if not os.path.exists(path):
            print(f"Error: ONNX model not found: {path}")
            print("Run export_onnx.py first.")
            sys.exit(1)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print("Loading ONNX sessions...")
    img_sess = ort.InferenceSession(img_model, sess_options=sess_options)
    txt_sess = ort.InferenceSession(txt_model, sess_options=sess_options)

    img_input_name = img_sess.get_inputs()[0].name
    txt_input_name = txt_sess.get_inputs()[0].name

    print(f"Running image encoder on {len(images)} images...")
    img_output = []
    for arr in images:
        out = img_sess.run(None, {img_input_name: arr})
        img_output.append(out[0])

    print(f"Running text encoder on {len(text_tokens)} prompts...")
    txt_output = []
    for arr in text_tokens:
        out = txt_sess.run(None, {txt_input_name: arr})
        txt_output.append(out[0])

    return img_output, txt_output


def save_local_embeddings(img_output, txt_output):
    """Save local embeddings as .npy files in diagnostics/ directory."""
    os.makedirs(DIAG_DIR, exist_ok=True)

    # Stack into (N, 512) and (M, 512) arrays
    img_embeds = np.vstack(img_output)
    txt_embeds = np.vstack(txt_output)

    np.save(LOCAL_IMG_PATH, img_embeds)
    np.save(LOCAL_TXT_PATH, txt_embeds)

    print(f"Saved local image embeddings: {LOCAL_IMG_PATH}  shape={img_embeds.shape}")
    print(f"Saved local text embeddings:  {LOCAL_TXT_PATH}  shape={txt_embeds.shape}")

    return img_embeds, txt_embeds


def load_local_embeddings():
    """Load previously saved local embeddings."""
    for path in [LOCAL_IMG_PATH, LOCAL_TXT_PATH]:
        if not os.path.exists(path):
            print(f"Error: Saved embeddings not found: {path}")
            print("Run with --save-local first.")
            sys.exit(1)

    img_embeds = np.load(LOCAL_IMG_PATH)
    txt_embeds = np.load(LOCAL_TXT_PATH)
    print(f"Loaded local image embeddings: shape={img_embeds.shape}")
    print(f"Loaded local text embeddings:  shape={txt_embeds.shape}")
    return img_embeds, txt_embeds


def download_device_embeddings(image_job_id, text_job_id):
    """Download on-device embeddings from QAI Hub inference jobs.

    Args:
        image_job_id: QAI Hub inference job ID for image encoder
        text_job_id: QAI Hub inference job ID for text encoder

    Returns:
        device_img: np.ndarray shape (N, 512)
        device_txt: np.ndarray shape (M, 512)
    """
    import qai_hub

    print(f"Downloading image encoder output from job: {image_job_id}")
    img_job = qai_hub.get_job(image_job_id)
    img_data = img_job.download_output_data()
    # Output format: {'output_0': list_of_arrays} where each is shape (1, 512)
    device_img = np.vstack(img_data["output_0"])

    print(f"Downloading text encoder output from job: {text_job_id}")
    txt_job = qai_hub.get_job(text_job_id)
    txt_data = txt_job.download_output_data()
    device_txt = np.vstack(txt_data["output_0"])

    print(f"Device image embeddings: shape={device_img.shape}  dtype={device_img.dtype}")
    print(f"Device text embeddings:  shape={device_txt.shape}  dtype={device_txt.dtype}")

    # Save device embeddings for future reference
    os.makedirs(DIAG_DIR, exist_ok=True)
    device_img_path = os.path.join(DIAG_DIR, "device_img_embeddings.npy")
    device_txt_path = os.path.join(DIAG_DIR, "device_txt_embeddings.npy")
    np.save(device_img_path, device_img)
    np.save(device_txt_path, device_txt)
    print(f"Saved device embeddings to {DIAG_DIR}/")

    return device_img, device_txt


def cosine_sim_per_sample(a, b):
    """Compute per-sample cosine similarity between two (N, D) arrays.

    Returns an array of shape (N,) with cosine similarity for each row.
    """
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a_norm * b_norm, axis=1)


def compare_embeddings(local_embeds, device_embeds, name):
    """Compare local vs device embeddings for one encoder.

    Prints per-sample cosine similarity stats and flags problem samples.

    Args:
        local_embeds: np.ndarray shape (N, D) — local ONNX FP32 embeddings
        device_embeds: np.ndarray shape (N, D) — on-device embeddings
        name: str — "image" or "text" (for display)
    """
    print(f"\n{'=' * 60}")
    print(f"  {name.upper()} ENCODER — Local vs Device Comparison")
    print(f"{'=' * 60}")

    n_local, n_device = len(local_embeds), len(device_embeds)
    if n_local != n_device:
        print(f"  WARNING: Sample count mismatch! local={n_local}, device={n_device}")
        n = min(n_local, n_device)
        local_embeds = local_embeds[:n]
        device_embeds = device_embeds[:n]
    else:
        n = n_local

    # --- Per-sample cosine similarity ---
    cos_sims = cosine_sim_per_sample(local_embeds, device_embeds)

    print(f"\n  Per-sample cosine similarity (local vs device):")
    print(f"    Mean:   {np.mean(cos_sims):.6f}")
    print(f"    Std:    {np.std(cos_sims):.6f}")
    print(f"    Min:    {np.min(cos_sims):.6f}  (sample {np.argmin(cos_sims)})")
    print(f"    Max:    {np.max(cos_sims):.6f}  (sample {np.argmax(cos_sims)})")
    print(f"    Median: {np.median(cos_sims):.6f}")

    # --- Flag problem samples (cosine sim < 0.95) ---
    threshold = 0.95
    problem_mask = cos_sims < threshold
    n_problems = np.sum(problem_mask)
    print(f"\n  Samples with cosine sim < {threshold}: {n_problems}/{n}")

    if n_problems > 0:
        problem_indices = np.where(problem_mask)[0]
        # Show up to 20 worst samples
        sorted_by_sim = problem_indices[np.argsort(cos_sims[problem_indices])]
        show_n = min(20, len(sorted_by_sim))
        print(f"  Worst {show_n} samples:")
        for idx in sorted_by_sim[:show_n]:
            print(f"    Sample {idx:>4d}: cos_sim={cos_sims[idx]:.6f}")

    # --- L2 norm comparison ---
    local_norms = np.linalg.norm(local_embeds, axis=1)
    device_norms = np.linalg.norm(device_embeds, axis=1)
    norm_ratios = device_norms / (local_norms + 1e-8)

    print(f"\n  L2 norm comparison:")
    print(f"    Local  — mean={np.mean(local_norms):.4f}  "
          f"std={np.std(local_norms):.4f}  "
          f"min={np.min(local_norms):.4f}  max={np.max(local_norms):.4f}")
    print(f"    Device — mean={np.mean(device_norms):.4f}  "
          f"std={np.std(device_norms):.4f}  "
          f"min={np.min(device_norms):.4f}  max={np.max(device_norms):.4f}")
    print(f"    Ratio (device/local) — mean={np.mean(norm_ratios):.4f}  "
          f"std={np.std(norm_ratios):.4f}  "
          f"min={np.min(norm_ratios):.4f}  max={np.max(norm_ratios):.4f}")

    # --- Element-wise absolute error ---
    abs_errors = np.abs(local_embeds - device_embeds)
    print(f"\n  Element-wise absolute error:")
    print(f"    Mean: {np.mean(abs_errors):.6f}")
    print(f"    Max:  {np.max(abs_errors):.6f}")
    print(f"    P95:  {np.percentile(abs_errors, 95):.6f}")
    print(f"    P99:  {np.percentile(abs_errors, 99):.6f}")

    return cos_sims


def run_comparison(local_img, local_txt, device_img, device_txt):
    """Run the full comparison between local and device embeddings."""

    print("\n" + "#" * 60)
    print("#  EMBEDDING COMPARISON: Local ONNX FP32 vs On-Device")
    print("#" * 60)

    # Compare each encoder
    img_cos_sims = compare_embeddings(local_img, device_img, "image")
    txt_cos_sims = compare_embeddings(local_txt, device_txt, "text")

    # --- Recall@10 for both ---
    print(f"\n{'=' * 60}")
    print(f"  RECALL@10 COMPARISON")
    print(f"{'=' * 60}")

    # Convert to list-of-arrays format expected by evaluate_track1
    local_img_list = [local_img[i : i + 1] for i in range(len(local_img))]
    local_txt_list = [local_txt[i : i + 1] for i in range(len(local_txt))]
    device_img_list = [device_img[i : i + 1] for i in range(len(device_img))]
    device_txt_list = [device_txt[i : i + 1] for i in range(len(device_txt))]

    recall_local = evaluate_track1(local_img_list, local_txt_list, TXT_LIST, IMG_LIST)
    recall_device = evaluate_track1(device_img_list, device_txt_list, TXT_LIST, IMG_LIST)

    # Cross-combinations to isolate which encoder causes the gap
    recall_local_img_device_txt = evaluate_track1(
        local_img_list, device_txt_list, TXT_LIST, IMG_LIST
    )
    recall_device_img_local_txt = evaluate_track1(
        device_img_list, local_txt_list, TXT_LIST, IMG_LIST
    )

    print(f"\n  {'Config':<40} {'Recall@10':>10}")
    print(f"  {'-' * 40} {'-' * 10}")
    print(f"  {'Local img + Local txt':<40} {recall_local:>10.4f}")
    print(f"  {'Device img + Device txt':<40} {recall_device:>10.4f}")
    print(f"  {'Local img + Device txt':<40} {recall_local_img_device_txt:>10.4f}")
    print(f"  {'Device img + Local txt':<40} {recall_device_img_local_txt:>10.4f}")

    delta = recall_device - recall_local
    print(f"\n  Gap (device - local): {delta:+.4f}")

    # --- Diagnosis ---
    print(f"\n{'=' * 60}")
    print(f"  DIAGNOSIS SUMMARY")
    print(f"{'=' * 60}")

    img_mean_sim = np.mean(img_cos_sims) if img_cos_sims is not None else 0
    txt_mean_sim = np.mean(txt_cos_sims) if txt_cos_sims is not None else 0

    print(f"\n  Image encoder agreement (mean cos sim): {img_mean_sim:.6f}")
    print(f"  Text encoder agreement  (mean cos sim): {txt_mean_sim:.6f}")

    if img_mean_sim < txt_mean_sim:
        print(f"\n  --> Image encoder has LARGER divergence from local.")
        print(f"      Likely culprit: image encoder quantization/compilation.")
    elif txt_mean_sim < img_mean_sim:
        print(f"\n  --> Text encoder has LARGER divergence from local.")
        print(f"      Likely culprit: text encoder quantization/compilation.")
    else:
        print(f"\n  --> Both encoders diverge equally.")

    # Check if it's a preprocessing issue (uniform shift) vs model issue (variable)
    if img_cos_sims is not None:
        img_std = np.std(img_cos_sims)
        if img_mean_sim > 0.99 and img_std < 0.005:
            print(f"\n  Image embeddings are very close — gap likely NOT from image encoder.")
        elif img_mean_sim < 0.90:
            print(f"\n  WARNING: Image embeddings diverge significantly (mean cos sim < 0.90).")
            print(f"  Possible causes: preprocessing mismatch, normalization issue, or"
                  f" quantization degradation.")

    if txt_cos_sims is not None:
        txt_std = np.std(txt_cos_sims)
        if txt_mean_sim > 0.99 and txt_std < 0.005:
            print(f"\n  Text embeddings are very close — gap likely NOT from text encoder.")
        elif txt_mean_sim < 0.90:
            print(f"\n  WARNING: Text embeddings diverge significantly (mean cos sim < 0.90).")
            print(f"  Possible causes: int64->int32 truncation, tokenization mismatch, or"
                  f" quantization degradation.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose local-vs-device Recall@10 gap for CLIP on QAI Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python diagnose_device_gap.py --save-local
  python diagnose_device_gap.py --compare --image-job-id jXXXXXX --text-job-id jYYYYYY
  python diagnose_device_gap.py --save-local --compare --image-job-id jXXXXXX --text-job-id jYYYYYY
""",
    )
    parser.add_argument(
        "--save-local",
        action="store_true",
        help="Run local ONNX FP32 inference and save embeddings to diagnostics/",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Download device embeddings and compare with saved local embeddings",
    )
    parser.add_argument(
        "--image-job-id",
        type=str,
        default=None,
        help="QAI Hub inference job ID for the image encoder",
    )
    parser.add_argument(
        "--text-job-id",
        type=str,
        default=None,
        help="QAI Hub inference job ID for the text encoder",
    )
    args = parser.parse_args()

    if not args.save_local and not args.compare:
        parser.print_help()
        print("\nError: Specify --save-local, --compare, or both.")
        sys.exit(1)

    if args.compare and (not args.image_job_id or not args.text_job_id):
        parser.print_help()
        print("\nError: --compare requires both --image-job-id and --text-job-id.")
        sys.exit(1)

    # --- Save local embeddings ---
    local_img, local_txt = None, None

    if args.save_local:
        print("=" * 60)
        print("  STEP 1: Local ONNX FP32 Inference")
        print("=" * 60)

        print("\nLoading data...")
        images = load_images()
        text_tokens = load_text_tokens()
        print(f"  {len(images)} images, {len(text_tokens)} text prompts loaded.\n")

        img_output, txt_output = run_local_onnx(images, text_tokens)
        local_img, local_txt = save_local_embeddings(img_output, txt_output)

        # Print local Recall@10 as sanity check
        recall_local = evaluate_track1(img_output, txt_output, TXT_LIST, IMG_LIST)
        print(f"\nLocal ONNX FP32 Recall@10: {recall_local:.4f}")

    # --- Compare with device ---
    if args.compare:
        print("\n" + "=" * 60)
        print("  STEP 2: Download Device Embeddings & Compare")
        print("=" * 60)

        # Load local embeddings if not already in memory
        if local_img is None or local_txt is None:
            print("\nLoading saved local embeddings...")
            local_img, local_txt = load_local_embeddings()

        # Download device embeddings
        print()
        device_img, device_txt = download_device_embeddings(
            args.image_job_id, args.text_job_id
        )

        # Run full comparison
        run_comparison(local_img, local_txt, device_img, device_txt)


if __name__ == "__main__":
    main()
