"""
Local INT8 Static Quantization for CLIP encoders using ONNXRuntime.

Steps:
  1. Build calibration datasets from sample images/text
  2. Quantize image encoder (all ops: Conv, MatMul, Gemm) — full INT8
  3. Quantize text encoder (MatMul/Gemm only) — conservative, preserves accuracy
  4. Output quantized ONNX files ready for local inference or QAI Hub recompile

Output:
  exported_onnx/image_encoder_int8.onnx
  exported_onnx/text_encoder_int8.onnx

Usage:
  python quantize_local.py
"""

import sys
import os
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "clip_model")
import clip as clip_lib

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)
from onnxruntime.quantization.preprocess import quant_pre_process

# --- Configuration ---
ONNX_DIR = "exported_onnx"
IMAGE_ONNX_PATH = os.path.join(ONNX_DIR, "image_encoder.onnx")
TEXT_ONNX_PATH  = os.path.join(ONNX_DIR, "text_encoder.onnx")
IMAGE_INT8_PATH = os.path.join(ONNX_DIR, "image_encoder_int8.onnx")
TEXT_INT8_PATH  = os.path.join(ONNX_DIR, "text_encoder_int8.onnx")
IMAGE_PREP_PATH = os.path.join(ONNX_DIR, "image_encoder_prep.onnx")
TEXT_PREP_PATH  = os.path.join(ONNX_DIR, "text_encoder_prep.onnx")

DATA_DIR  = r"C:\rama\projects\data\lpcvc_track1_sample_data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
IMG_LIST  = os.path.join(DATA_DIR, "img_list.csv")
TXT_LIST  = os.path.join(DATA_DIR, "txt_list.csv")
# ---------------------


class ImageCalibrationReader(CalibrationDataReader):
    """Feeds competition-style /255 images (no extra normalization — baked into model)."""

    def __init__(self):
        df = pd.read_csv(IMG_LIST)
        filenames = df.iloc[:, 0].tolist()
        print(f"  Loading {len(filenames)} calibration images...")

        self._data = []
        for f in filenames:
            img = Image.open(os.path.join(IMAGE_DIR, f)).convert("RGB").resize((224, 224))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))[np.newaxis, :]  # (1, 3, 224, 224)
            self._data.append({"image": arr})

        self._iter = iter(self._data)

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._data)


class TextCalibrationReader(CalibrationDataReader):
    """Feeds tokenized text prompts as int32 (ONNXRuntime quantization uses int32 internally)."""

    def __init__(self):
        df = pd.read_csv(TXT_LIST)
        prompts = df.iloc[:, 1].dropna().tolist()
        print(f"  Tokenizing {len(prompts)} calibration prompts...")

        import torch
        tokens = clip_lib.tokenize(prompts)  # (M, 77) int64
        self._data = [
            {"text": tokens[i:i+1].numpy().astype(np.int64)}
            for i in range(len(tokens))
        ]
        self._iter = iter(self._data)

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._data)


def quantize_image_encoder():
    print("\n[Image Encoder] Preprocessing (shape inference + graph optimization)...")
    quant_pre_process(IMAGE_ONNX_PATH, IMAGE_PREP_PATH)
    print("\n[Image Encoder] Quantizing (full INT8 — Conv, MatMul, Gemm)...")
    reader = ImageCalibrationReader()
    quantize_static(
        model_input=IMAGE_PREP_PATH,
        model_output=IMAGE_INT8_PATH,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,          # QDQ is QAI Hub-compatible
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,                       # per-channel weights: better accuracy for ViT
    )
    size_mb = os.path.getsize(IMAGE_INT8_PATH) / 1e6
    print(f"  Saved: {IMAGE_INT8_PATH}  ({size_mb:.1f} MB)")


def quantize_text_encoder():
    print("\n[Text Encoder] Preprocessing (shape inference + graph optimization)...")
    quant_pre_process(TEXT_ONNX_PATH, TEXT_PREP_PATH)
    print("\n[Text Encoder] Quantizing (conservative — MatMul/Gemm only, skip embeddings)...")
    reader = TextCalibrationReader()
    quantize_static(
        model_input=TEXT_PREP_PATH,
        model_output=TEXT_INT8_PATH,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],  # skip Gather (embeddings) and layer norm
    )
    size_mb = os.path.getsize(TEXT_INT8_PATH) / 1e6
    print(f"  Saved: {TEXT_INT8_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    # Validate FP32 ONNX inputs exist
    for path in [IMAGE_ONNX_PATH, TEXT_ONNX_PATH]:
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run export_onnx.py first.")
            sys.exit(1)

    print("=== Local ONNXRuntime INT8 Static Quantization ===")
    print(f"Input:  {ONNX_DIR}/image_encoder.onnx + text_encoder.onnx")
    print(f"Output: {ONNX_DIR}/image_encoder_int8.onnx + text_encoder_int8.onnx")

    quantize_image_encoder()
    quantize_text_encoder()

    print("\n=== Quantization complete ===")
    print("Next: python inference_onnx_local.py")
