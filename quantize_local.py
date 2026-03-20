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
  python quantize_local.py                     # default: QOperator, QUInt8 activations, Percentile
  python quantize_local.py --format qdq        # use QDQ format instead (for QAI Hub export)
  python quantize_local.py --activation qint8  # signed int8 activations
  python quantize_local.py --calibration minmax  # MinMax calibration instead of Percentile

Quantization format notes:
  QOperator (default) — fuses Q/DQ into ops, always outputs float32. Best for local ORT inference.
  QDQ              — inserts paired Q/DQ nodes. Designed for hardware compilers (QAI Hub, TensorRT).
                     ORT may return raw int8 output values when using QDQ, causing ~random Recall@10.
"""

import sys
import os
import argparse
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


def quantize_image_encoder(quant_format, activation_type, calibrate_method):
    print("\n[Image Encoder] Preprocessing (shape inference + graph optimization)...")
    quant_pre_process(IMAGE_ONNX_PATH, IMAGE_PREP_PATH)
    print(f"\n[Image Encoder] Quantizing (full INT8 — Conv, MatMul, Gemm) "
          f"[format={quant_format.name}, act={activation_type.name}, cal={calibrate_method.name}]...")
    reader = ImageCalibrationReader()
    quantize_static(
        model_input=IMAGE_PREP_PATH,
        model_output=IMAGE_INT8_PATH,
        calibration_data_reader=reader,
        quant_format=quant_format,
        weight_type=QuantType.QInt8,
        activation_type=activation_type,
        calibrate_method=calibrate_method,
        per_channel=True,                       # per-channel weights: better accuracy for ViT
    )
    size_mb = os.path.getsize(IMAGE_INT8_PATH) / 1e6
    print(f"  Saved: {IMAGE_INT8_PATH}  ({size_mb:.1f} MB)")


def quantize_text_encoder(quant_format, activation_type, calibrate_method):
    print("\n[Text Encoder] Preprocessing (shape inference + graph optimization)...")
    quant_pre_process(TEXT_ONNX_PATH, TEXT_PREP_PATH)
    print(f"\n[Text Encoder] Quantizing (conservative — MatMul/Gemm only, skip embeddings) "
          f"[format={quant_format.name}, act={activation_type.name}, cal={calibrate_method.name}]...")
    reader = TextCalibrationReader()
    quantize_static(
        model_input=TEXT_PREP_PATH,
        model_output=TEXT_INT8_PATH,
        calibration_data_reader=reader,
        quant_format=quant_format,
        weight_type=QuantType.QInt8,
        activation_type=activation_type,
        calibrate_method=calibrate_method,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],  # skip Gather (embeddings) and layer norm
    )
    size_mb = os.path.getsize(TEXT_INT8_PATH) / 1e6
    print(f"  Saved: {TEXT_INT8_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local INT8 Static Quantization for CLIP")
    parser.add_argument(
        "--format", default="qoperator", choices=["qoperator", "qdq"],
        help=(
            "Quantization format:\n"
            "  qoperator (default) — fuses Q/DQ into ops, float32 output. Best for local ORT.\n"
            "  qdq                 — QAI Hub / TensorRT compatible, may output raw int8 in ORT."
        ),
    )
    parser.add_argument(
        "--activation", default="quint8", choices=["quint8", "qint8"],
        help=(
            "Activation quantization type:\n"
            "  quint8 (default) — broader ORT CPU kernel coverage, fewer silent fallbacks.\n"
            "  qint8            — signed int8, required for some hardware targets."
        ),
    )
    parser.add_argument(
        "--calibration", default="percentile", choices=["percentile", "minmax"],
        help=(
            "Calibration method:\n"
            "  percentile (default) — trims outliers, better for transformer attention distributions.\n"
            "  minmax               — captures absolute worst-case range."
        ),
    )
    args = parser.parse_args()

    quant_format = QuantFormat.QOperator if args.format == "qoperator" else QuantFormat.QDQ
    activation_type = QuantType.QUInt8 if args.activation == "quint8" else QuantType.QInt8
    calibrate_method = (CalibrationMethod.Percentile if args.calibration == "percentile"
                        else CalibrationMethod.MinMax)

    # Validate FP32 ONNX inputs exist
    for path in [IMAGE_ONNX_PATH, TEXT_ONNX_PATH]:
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run export_onnx.py first.")
            sys.exit(1)

    print("=== Local ONNXRuntime INT8 Static Quantization ===")
    print(f"Input:  {ONNX_DIR}/image_encoder.onnx + text_encoder.onnx")
    print(f"Output: {ONNX_DIR}/image_encoder_int8.onnx + text_encoder_int8.onnx")
    print(f"Config: format={quant_format.name}  activation={activation_type.name}  "
          f"calibration={calibrate_method.name}")

    quantize_image_encoder(quant_format, activation_type, calibrate_method)
    quantize_text_encoder(quant_format, activation_type, calibrate_method)

    print("\n=== Quantization complete ===")
    print("Next: python inference_onnx_local.py --inspect-embeddings")
