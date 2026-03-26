"""
COCO Karpathy 1K Val — Image-to-Text Retrieval Evaluation

Standard benchmark for measuring Recall@1/5/10 on COCO Karpathy 1K validation split.
Uses first 1000 images from split=="val" with all 5 captions each (5000 texts total).

This gives much more stable Recall estimates than the competition's 57-image sample dataset.

Usage:
  python evaluate_coco1k.py                                        # PyTorch FP32 (default)
  python evaluate_coco1k.py --onnx                                 # ONNX FP32 model
  python evaluate_coco1k.py --checkpoint lora_checkpoints/epoch_3  # LoRA checkpoint
  python evaluate_coco1k.py --merged-weights model.pt              # Merged weights file
  python evaluate_coco1k.py --batch-size 32                        # Custom batch size
"""

import sys
import os
import json
import argparse
import re
import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode,
)

sys.path.insert(0, "clip_model")
import clip as clip_lib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COCO_JSON = r"C:\rama\projects\data\coco\dataset_coco.json"
COCO_IMAGE_ROOT = r"C:\rama\projects\data\coco"
ONNX_DIR = "exported_onnx"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Standard CLIP preprocessing (for PyTorch mode)
pytorch_transform = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize(CLIP_MEAN, CLIP_STD),
])

# Competition-style preprocessing (for ONNX mode — norm baked into model)
onnx_transform = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    ToTensor(),  # /255 only, no normalization
])


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_karpathy_val(coco_json_path, image_root, max_images=1000):
    """
    Load COCO Karpathy 1K val split.

    Args:
        coco_json_path: Path to dataset_coco.json (Karpathy splits).
        image_root: Root directory containing val2017/ and train2017/ subdirs.
        max_images: Number of val images to use (standard: 1000).

    Returns:
        image_paths: List of absolute image file paths.
        all_captions: List of all caption strings (5 per image, ~5000 total).
        img2txt: List of lists — img2txt[i] = [indices into all_captions for image i].
    """
    print(f"Loading Karpathy splits from: {coco_json_path}")
    with open(coco_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter to val split, take first max_images
    val_images = [img for img in data["images"] if img["split"] == "val"]
    val_images = val_images[:max_images]
    print(f"  Found {len(val_images)} val images (using first {max_images})")

    image_paths = []
    all_captions = []
    img2txt = []

    for img_entry in val_images:
        # Resolve image path: Karpathy filenames use COCO 2014 naming
        # (e.g., COCO_val2014_000000XXXXX.jpg) but actual files on disk in
        # val2017/train2017 are just the numeric ID (e.g., 000000XXXXX.jpg).
        filepath_dir = img_entry.get("filepath", "val2017")
        karpathy_filename = img_entry["filename"]

        # Extract numeric ID from the Karpathy filename
        # Handles: COCO_val2014_000000391895.jpg -> 000000391895.jpg
        #          or just 000000391895.jpg
        match = re.search(r"(\d{12})\.jpg$", karpathy_filename)
        if match:
            actual_filename = match.group(1) + ".jpg"
        else:
            # Fallback: use filename as-is
            actual_filename = karpathy_filename

        img_path = os.path.join(image_root, filepath_dir, actual_filename)

        # Verify file exists; try alternate directory if not found
        if not os.path.exists(img_path):
            # Try the other common directory
            alt_dir = "train2017" if filepath_dir == "val2017" else "val2017"
            alt_path = os.path.join(image_root, alt_dir, actual_filename)
            if os.path.exists(alt_path):
                img_path = alt_path
            else:
                print(f"  WARNING: Image not found: {img_path} (skipping)")
                continue

        image_paths.append(img_path)

        # Collect all captions for this image
        caption_indices = []
        for sent in img_entry["sentences"]:
            caption_indices.append(len(all_captions))
            all_captions.append(sent["raw"])
        img2txt.append(caption_indices)

    print(f"  Loaded {len(image_paths)} images, {len(all_captions)} captions")
    return image_paths, all_captions, img2txt


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------
def compute_recall(img_embeddings, txt_embeddings, img2txt_mapping, ks=(1, 5, 10)):
    """
    Compute standard Recall@K for image-to-text retrieval.

    For each image query, checks if ANY of its ground-truth captions appear
    in the top-K most similar texts (binary hit per query, then averaged).

    Args:
        img_embeddings: (N, D) L2-normalized image embeddings.
        txt_embeddings: (M, D) L2-normalized text embeddings.
        img2txt_mapping: List of lists — img2txt_mapping[i] = ground-truth text indices.
        ks: Tuple of K values for Recall@K.

    Returns:
        Dict mapping "R@k" -> float recall value.
    """
    # Cosine similarity matrix (embeddings are already normalized)
    sim = img_embeddings @ txt_embeddings.T  # (N, M)

    results = {}
    for k in ks:
        hits = 0
        for i in range(len(img_embeddings)):
            top_k = np.argsort(-sim[i])[:k]
            gt_texts = set(img2txt_mapping[i])
            if len(set(top_k.tolist()) & gt_texts) > 0:
                hits += 1
        results[f"R@{k}"] = hits / len(img_embeddings)
    return results


# ---------------------------------------------------------------------------
# PyTorch inference
# ---------------------------------------------------------------------------
def run_pytorch(image_paths, captions, device, model_name="ViT-B/16",
                checkpoint=None, merged_weights=None, batch_size=64):
    """Run inference using PyTorch CLIP model."""
    print(f"\nLoading CLIP model ({model_name})...")
    model, _ = clip_lib.load(model_name, device=device)
    model.eval()

    # Load LoRA checkpoint if specified
    if checkpoint is not None:
        print(f"Loading LoRA checkpoint from: {checkpoint}")
        from peft import PeftModel
        model = model.float()
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()
        model.eval()

    # Load merged weights if specified
    if merged_weights is not None:
        print(f"Loading merged weights from: {merged_weights}")
        model = model.float()
        state_dict = torch.load(merged_weights, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    # --- Encode images ---
    print(f"Encoding {len(image_paths)} images (batch_size={batch_size})...")
    # Preprocess all images first
    preprocessed = []
    skipped = 0
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            preprocessed.append(pytorch_transform(img))
        except Exception as e:
            print(f"  WARNING: Failed to load {path}: {e}")
            skipped += 1
            preprocessed.append(None)

    if skipped > 0:
        print(f"  Skipped {skipped} images due to errors")

    # Filter out failed images (keep index mapping consistent)
    valid_indices = [i for i, p in enumerate(preprocessed) if p is not None]
    valid_tensors = [preprocessed[i] for i in valid_indices]

    img_embeddings = []
    t0 = time.time()
    for i in range(0, len(valid_tensors), batch_size):
        batch = torch.stack(valid_tensors[i:i + batch_size]).to(device)
        with torch.no_grad():
            emb = model.encode_image(batch)
        img_embeddings.append(emb.cpu())
    img_embeddings = torch.cat(img_embeddings, dim=0)
    t_img = time.time() - t0
    print(f"  Image encoding: {t_img:.1f}s ({t_img / len(valid_tensors) * 1000:.1f}ms/image)")

    # --- Encode text ---
    print(f"Tokenizing and encoding {len(captions)} captions...")
    tokens = clip_lib.tokenize(captions, truncate=True)  # (M, 77)

    txt_embeddings = []
    t0 = time.time()
    for i in range(0, len(tokens), batch_size):
        batch = tokens[i:i + batch_size].to(device)
        with torch.no_grad():
            emb = model.encode_text(batch)
        txt_embeddings.append(emb.cpu())
    txt_embeddings = torch.cat(txt_embeddings, dim=0)
    t_txt = time.time() - t0
    print(f"  Text encoding: {t_txt:.1f}s ({t_txt / len(tokens) * 1000:.1f}ms/text)")

    # L2 normalize
    img_embeddings = img_embeddings.numpy().astype(np.float32)
    txt_embeddings = txt_embeddings.numpy().astype(np.float32)
    img_embeddings = img_embeddings / np.linalg.norm(img_embeddings, axis=1, keepdims=True)
    txt_embeddings = txt_embeddings / np.linalg.norm(txt_embeddings, axis=1, keepdims=True)

    return img_embeddings, txt_embeddings, valid_indices


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------
def run_onnx(image_paths, captions, batch_size=64):
    """Run inference using ONNX Runtime with exported FP32 models."""
    import onnxruntime as ort

    img_model = os.path.join(ONNX_DIR, "image_encoder.onnx")
    txt_model = os.path.join(ONNX_DIR, "text_encoder.onnx")

    if not os.path.exists(img_model):
        print(f"ERROR: Image encoder not found: {img_model}")
        print("Run export_onnx.py first.")
        sys.exit(1)
    if not os.path.exists(txt_model):
        print(f"ERROR: Text encoder not found: {txt_model}")
        print("Run export_onnx.py first.")
        sys.exit(1)

    print(f"\nLoading ONNX sessions...")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    img_sess = ort.InferenceSession(img_model, sess_opts)
    txt_sess = ort.InferenceSession(txt_model, sess_opts)

    img_input_name = img_sess.get_inputs()[0].name
    txt_input_name = txt_sess.get_inputs()[0].name

    # --- Encode images (one at a time — ONNX models are batch=1) ---
    print(f"Encoding {len(image_paths)} images...")
    img_embeddings = []
    valid_indices = []
    t0 = time.time()
    for idx, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            tensor = onnx_transform(img)  # (3, 224, 224) — /255 only
            arr = tensor.numpy()[np.newaxis, :].astype(np.float32)  # (1, 3, 224, 224)
            out = img_sess.run(None, {img_input_name: arr})
            img_embeddings.append(out[0])  # (1, D)
            valid_indices.append(idx)
        except Exception as e:
            print(f"  WARNING: Failed on {path}: {e}")

        if (idx + 1) % 200 == 0:
            print(f"  ... {idx + 1}/{len(image_paths)} images done")

    t_img = time.time() - t0
    print(f"  Image encoding: {t_img:.1f}s ({t_img / len(valid_indices) * 1000:.1f}ms/image)")

    # --- Encode text (one at a time) ---
    print(f"Tokenizing and encoding {len(captions)} captions...")
    tokens = clip_lib.tokenize(captions, truncate=True)  # (M, 77)

    txt_embeddings = []
    t0 = time.time()
    for i in range(len(tokens)):
        token_arr = tokens[i:i + 1].numpy().astype(np.int64)  # (1, 77)
        out = txt_sess.run(None, {txt_input_name: token_arr})
        txt_embeddings.append(out[0])

        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{len(tokens)} captions done")

    t_txt = time.time() - t0
    print(f"  Text encoding: {t_txt:.1f}s ({t_txt / len(tokens) * 1000:.1f}ms/text)")

    img_embeddings = np.concatenate(img_embeddings, axis=0).astype(np.float32)
    txt_embeddings = np.concatenate(txt_embeddings, axis=0).astype(np.float32)

    # L2 normalize
    img_embeddings = img_embeddings / np.linalg.norm(img_embeddings, axis=1, keepdims=True)
    txt_embeddings = txt_embeddings / np.linalg.norm(txt_embeddings, axis=1, keepdims=True)

    return img_embeddings, txt_embeddings, valid_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="COCO Karpathy 1K Val — Image-to-Text Retrieval Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--onnx", action="store_true",
        help="Use ONNX FP32 models instead of PyTorch",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to LoRA checkpoint directory (PyTorch mode only)",
    )
    parser.add_argument(
        "--merged-weights", type=str, default=None,
        help="Path to merged weights .pt file (PyTorch mode only)",
    )
    parser.add_argument(
        "--model", default="ViT-B/16", choices=["ViT-B/16", "ViT-L/14"],
        help="CLIP model variant (PyTorch mode only, default: ViT-B/16)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for encoding (default: 64)",
    )
    parser.add_argument(
        "--max-images", type=int, default=1000,
        help="Number of val images to evaluate (default: 1000)",
    )
    parser.add_argument(
        "--coco-json", type=str, default=COCO_JSON,
        help=f"Path to dataset_coco.json (default: {COCO_JSON})",
    )
    parser.add_argument(
        "--coco-root", type=str, default=COCO_IMAGE_ROOT,
        help=f"Path to COCO image root dir (default: {COCO_IMAGE_ROOT})",
    )
    args = parser.parse_args()

    if args.onnx and (args.checkpoint or args.merged_weights):
        print("ERROR: --onnx cannot be combined with --checkpoint or --merged-weights")
        sys.exit(1)

    # Load dataset
    image_paths, all_captions, img2txt = load_karpathy_val(
        args.coco_json, args.coco_root, max_images=args.max_images,
    )

    if len(image_paths) == 0:
        print("ERROR: No images loaded. Check paths and dataset_coco.json.")
        sys.exit(1)

    # Run inference
    if args.onnx:
        mode_label = "ONNX FP32"
        img_emb, txt_emb, valid_indices = run_onnx(
            image_paths, all_captions, batch_size=args.batch_size,
        )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        mode_label = f"{args.model} (PyTorch FP32)"
        if args.checkpoint:
            mode_label = f"{args.model} (LoRA: {args.checkpoint})"
        elif args.merged_weights:
            mode_label = f"{args.model} (merged: {os.path.basename(args.merged_weights)})"

        img_emb, txt_emb, valid_indices = run_pytorch(
            image_paths, all_captions, device,
            model_name=args.model,
            checkpoint=args.checkpoint,
            merged_weights=args.merged_weights,
            batch_size=args.batch_size,
        )

    # Remap img2txt if some images were skipped
    if len(valid_indices) < len(img2txt):
        img2txt_filtered = [img2txt[i] for i in valid_indices]
    else:
        img2txt_filtered = img2txt

    # Compute recall
    print("\nComputing Recall@K...")
    results = compute_recall(img_emb, txt_emb, img2txt_filtered, ks=[1, 5, 10])

    # Print results
    n_imgs = len(valid_indices)
    n_txts = len(all_captions)
    print()
    print("=" * 56)
    print("  COCO Karpathy 1K Val -- Image-to-Text Retrieval")
    print("=" * 56)
    print(f"  Model:   {mode_label}")
    print(f"  Images:  {n_imgs}, Texts: {n_txts}")
    print()
    for k in [1, 5, 10]:
        val = results[f"R@{k}"]
        print(f"  Recall@{k:<3} {val * 100:6.2f}%")
    print("=" * 56)


if __name__ == "__main__":
    main()
