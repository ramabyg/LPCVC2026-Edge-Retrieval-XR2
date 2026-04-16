"""
Shared calibration data loader for QAI Hub compile-time INT8 quantization.

Supports three dataset sources:
  sample    — the 57-image competition sample dataset (fast, always available)
  coco      — COCO captions (COCO-style JSON: {images, annotations})
  flickr30k — Flickr30k captions (same COCO-style JSON format)

Usage:
    from src.common.calibration import load_calibration_data
    img_calib = load_calibration_data("image", source="coco", n_samples=500)
    txt_calib = load_calibration_data("text",  source="coco", n_samples=500)
"""

import os
import sys
import numpy as np


def load_calibration_data(encoder, source="sample", n_samples=None):
    """Load calibration samples for the given encoder.

    Args:
        encoder:   "image" or "text"
        source:    "sample" | "coco" | "flickr30k"
        n_samples: max samples to load (None = use all available)

    Returns:
        dict {input_name: list[np.ndarray]}  — QAI Hub calibration format
        Each array has shape (1, ...) as required by submit_compile_job.
    """
    import pandas as pd
    from PIL import Image
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Compose
    from torchvision.transforms import InterpolationMode

    from src.common.config import (
        IMAGE_DIR, IMG_LIST, TXT_LIST,
        COCO_DATA_DIR, COCO_JSON,
        FLICKR30K_IMG_DIR, FLICKR30K_JSON,
    )

    preprocess = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor(),  # uint8 → float32, /255 — CLIP norm is baked into the ONNX model
    ])

    # ------------------------------------------------------------------
    # Resolve image paths and caption texts per source
    # ------------------------------------------------------------------
    if source == "sample":
        df = pd.read_csv(IMG_LIST)
        img_paths = [os.path.join(IMAGE_DIR, f) for f in df.iloc[:, 0].tolist()]
        df_txt = pd.read_csv(TXT_LIST)
        texts = df_txt.iloc[:, 1].tolist()

    elif source in ("coco", "flickr30k"):
        import json
        json_path = COCO_JSON if source == "coco" else FLICKR30K_JSON
        img_dir   = COCO_DATA_DIR if source == "coco" else FLICKR30K_IMG_DIR

        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # COCO-style JSON: {images: [{id, file_name}], annotations: [{image_id, caption}]}
        id2file = {img["id"]: img["file_name"] for img in data["images"]}
        anns    = data["annotations"]  # list of {image_id, caption}
        img_paths = [os.path.join(img_dir, id2file[a["image_id"]]) for a in anns]
        texts     = [a["caption"] for a in anns]

    else:
        raise ValueError(f"Unknown calibration source: {source!r}  (use sample | coco | flickr30k)")

    # Apply n_samples cap
    if n_samples is not None:
        img_paths = img_paths[:n_samples]
        texts     = texts[:n_samples]

    print(f"  Calibration: source={source}  images={len(img_paths)}  texts={len(texts)}")

    # ------------------------------------------------------------------
    # Build encoder-specific payload
    # ------------------------------------------------------------------
    if encoder == "image":
        samples = []
        for path in img_paths:
            img = Image.open(path).convert("RGB")
            samples.append(preprocess(img).numpy()[np.newaxis])  # (1, 3, 224, 224)
        return {"image": samples}

    elif encoder == "text":
        clip_path = os.path.join(os.path.dirname(__file__), "..", "..", "clip_model")
        sys.path.insert(0, clip_path)
        import clip as clip_module
        tokens = clip_module.tokenize(texts, truncate=True).numpy().astype(np.int64)  # int64
        return {"text": [tok[np.newaxis] for tok in tokens]}  # list of (1, 77)

    else:
        raise ValueError(f"Unknown encoder: {encoder!r}  (use image | text)")
