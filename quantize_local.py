"""
Local INT8 Static Quantization for CLIP encoders using ONNXRuntime.

Steps:
  1. Build calibration datasets from sample images/text
  2. Quantize image encoder (conservative — Conv, MatMul, Gemm only, skips Softmax + LayerNorm)
  3. Quantize text encoder (MatMul/Gemm only) — conservative, preserves accuracy
  4. Output quantized ONNX files ready for local inference or QAI Hub recompile

Output:
  exported_onnx/image_encoder_int8.onnx
  exported_onnx/text_encoder_int8.onnx

Usage:
  python quantize_local.py                       # default: QOperator, QUInt8 activations, Percentile
  python quantize_local.py --format qdq          # use QDQ format instead (for QAI Hub export)
  python quantize_local.py --activation qint8    # signed int8 activations
  python quantize_local.py --calibration minmax  # MinMax calibration instead of Percentile
  python quantize_local.py --optimize-graph      # apply transformer graph fusion before quantization

Quantization format notes:
  QOperator (default) — fuses Q/DQ into ops, always outputs float32. Best for local ORT inference.
  QDQ              — inserts paired Q/DQ nodes. Designed for hardware compilers (QAI Hub, TensorRT).
                     ORT may return raw int8 output values when using QDQ, causing ~random Recall@10.

Image encoder quantization notes:
  Only Conv, MatMul, and Gemm are quantized (same conservative approach as text encoder).
  Softmax and LayerNormalization are left in FP32 — INT8 quantization of these ops in ViT
  corrupts attention distributions and the residual stream across all transformer layers,
  causing catastrophic Recall@10 collapse (~0.07 observed vs 0.87 FP32 baseline).
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

# --- Configuration (set after argparse below) ---
ONNX_DIR = "exported_onnx"

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


def maybe_optimize_graph(input_path, output_path, model_type, num_heads, hidden_size):
    """Apply onnxruntime.transformers graph optimization (fuses attention, LayerNorm, GELU)."""
    from onnxruntime.transformers import optimizer
    opt = optimizer.optimize_model(
        input_path,
        model_type=model_type,
        num_heads=num_heads,
        hidden_size=hidden_size,
    )
    opt.save_model_to_file(output_path)
    print(f"  Graph optimized -> {output_path}")


def quantize_image_encoder(quant_format, activation_type, calibrate_method, optimize_graph=False,
                           num_heads=12, hidden_size=768):
    print("\n[Image Encoder] Preprocessing (shape inference + graph optimization)...")
    quant_pre_process(IMAGE_ONNX_PATH, IMAGE_PREP_PATH)

    model_input = IMAGE_PREP_PATH
    if optimize_graph:
        opt_path = IMAGE_PREP_PATH.replace("_prep.onnx", "_opt.onnx")
        print(f"\n[Image Encoder] Applying transformer graph optimization "
              f"(model_type=vit, num_heads={num_heads}, hidden_size={hidden_size})...")
        maybe_optimize_graph(IMAGE_PREP_PATH, opt_path, model_type="vit",
                             num_heads=num_heads, hidden_size=hidden_size)
        model_input = opt_path

    print(f"\n[Image Encoder] Quantizing (conservative — Conv, MatMul, Gemm; skips Softmax, LayerNorm) "
          f"[format={quant_format.name}, act={activation_type.name}, cal={calibrate_method.name}]...")
    reader = ImageCalibrationReader()
    quantize_static(
        model_input=model_input,
        model_output=IMAGE_INT8_PATH,
        calibration_data_reader=reader,
        quant_format=quant_format,
        weight_type=QuantType.QInt8,
        activation_type=activation_type,
        calibrate_method=calibrate_method,
        per_channel=True,                           # per-channel weights: better accuracy for ViT
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],  # excludes Softmax and LayerNorm
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
        "--model", default="ViT-B/16", choices=["ViT-B/16", "ViT-L/14"],
        help="CLIP model variant to quantize (default: ViT-B/16)",
    )
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
    parser.add_argument(
        "--optimize-graph", action="store_true",
        help=(
            "Apply onnxruntime.transformers optimizer before quantization (fuses attention, "
            "LayerNorm, GELU subgraphs). Off by default — try this if accuracy is still poor "
            "after the standard run."
        ),
    )
    args = parser.parse_args()

    quant_format = QuantFormat.QOperator if args.format == "qoperator" else QuantFormat.QDQ
    activation_type = QuantType.QUInt8 if args.activation == "quint8" else QuantType.QInt8
    calibrate_method = (CalibrationMethod.Percentile if args.calibration == "percentile"
                        else CalibrationMethod.MinMax)

    # Derive file paths and model params from --model
    if args.model == "ViT-B/16":
        slug = ""
        num_heads, hidden_size = 12, 768
    else:  # ViT-L/14
        slug = "_" + args.model.lower().replace("/", "").replace("-", "")  # "_vitl14"
        num_heads, hidden_size = 16, 1024

    IMAGE_ONNX_PATH = os.path.join(ONNX_DIR, f"image_encoder{slug}.onnx")
    TEXT_ONNX_PATH  = os.path.join(ONNX_DIR, f"text_encoder{slug}.onnx")
    IMAGE_INT8_PATH = os.path.join(ONNX_DIR, f"image_encoder{slug}_int8.onnx")
    TEXT_INT8_PATH  = os.path.join(ONNX_DIR, f"text_encoder{slug}_int8.onnx")
    IMAGE_PREP_PATH = os.path.join(ONNX_DIR, f"image_encoder{slug}_prep.onnx")
    TEXT_PREP_PATH  = os.path.join(ONNX_DIR, f"text_encoder{slug}_prep.onnx")

    # Validate FP32 ONNX inputs exist
    for path in [IMAGE_ONNX_PATH, TEXT_ONNX_PATH]:
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run export_onnx.py --model {args.model} first.")
            sys.exit(1)

    print("=== Local ONNXRuntime INT8 Static Quantization ===")
    print(f"Model:  {args.model}")
    print(f"Input:  {IMAGE_ONNX_PATH} + {TEXT_ONNX_PATH}")
    print(f"Output: {IMAGE_INT8_PATH} + {TEXT_INT8_PATH}")
    print(f"Config: format={quant_format.name}  activation={activation_type.name}  "
          f"calibration={calibrate_method.name}")

    quantize_image_encoder(quant_format, activation_type, calibrate_method,
                           optimize_graph=args.optimize_graph,
                           num_heads=num_heads, hidden_size=hidden_size)
    quantize_text_encoder(quant_format, activation_type, calibrate_method)

    print("\n=== Quantization complete ===")
    print("Next: python inference_onnx_local.py --inspect-embeddings")
