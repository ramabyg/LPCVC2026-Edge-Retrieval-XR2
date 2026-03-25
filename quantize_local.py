"""
Local INT8 Static Quantization for CLIP encoders using ONNXRuntime.

Steps:
  1. Build calibration datasets from sample images/text
  2. Quantize image encoder (conservative — Conv, MatMul, Gemm only, skips Softmax + LayerNorm)
  3. Quantize text encoder (MatMul/Gemm only) — conservative, preserves accuracy
  4. Output quantized ONNX files ready for local inference or QAI Hub recompile

Output filenames include format + activation for easy benchmarking:
  exported_onnx/image_encoder{slug}_int8_{format}_{activation}.onnx
  exported_onnx/text_encoder{slug}_int8_{format}_{activation}.onnx

Examples:
  image_encoder_int8_qoperator_quint8.onnx
  image_encoder_vitl14_int8_qdq_qint8.onnx

Usage:
  python quantize_local.py                                    # all 4 combos × both models (8 per encoder)
  python quantize_local.py --model ViT-B/16                  # all 4 combos for ViT-B/16 only
  python quantize_local.py --model ViT-B/16 --format qdq --activation qint8  # single targeted combo
  python quantize_local.py --optimize-graph                  # apply transformer graph fusion before quant

Quantization format notes:
  QOperator — fuses Q/DQ into ops, always outputs float32. Best for local ORT inference.
  QDQ       — inserts paired Q/DQ nodes. Designed for hardware compilers (QAI Hub, TensorRT).
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

# --- Configuration ---
ONNX_DIR = "exported_onnx"

DATA_DIR  = r"C:\rama\projects\data\lpcvc_track1_sample_data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
IMG_LIST  = os.path.join(DATA_DIR, "img_list.csv")
TXT_LIST  = os.path.join(DATA_DIR, "txt_list.csv")

# All 4 quantization combinations
COMBOS = [
    (QuantFormat.QOperator, QuantType.QUInt8, "qoperator", "quint8"),
    (QuantFormat.QOperator, QuantType.QInt8,  "qoperator", "qint8"),
    (QuantFormat.QDQ,       QuantType.QUInt8, "qdq",       "quint8"),
    (QuantFormat.QDQ,       QuantType.QInt8,  "qdq",       "qint8"),
]
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
    """Feeds tokenized text prompts as int64."""

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


def preprocess_image(onnx_path, prep_path, optimize_graph=False, num_heads=12, hidden_size=768):
    """Run quant_pre_process (and optional graph optimization) once before quantization loops."""
    print(f"  Preprocessing (shape inference + graph optimization)...")
    quant_pre_process(onnx_path, prep_path)
    model_input = prep_path
    if optimize_graph:
        opt_path = prep_path.replace("_prep.onnx", "_opt.onnx")
        print(f"  Applying transformer graph optimization (vit, heads={num_heads}, hidden={hidden_size})...")
        maybe_optimize_graph(prep_path, opt_path, model_type="vit",
                             num_heads=num_heads, hidden_size=hidden_size)
        model_input = opt_path
    return model_input


def preprocess_text(onnx_path, prep_path):
    """Run quant_pre_process once before quantization loops."""
    print(f"  Preprocessing (shape inference + graph optimization)...")
    quant_pre_process(onnx_path, prep_path)
    return prep_path


def quantize_image(prep_path, output_path, quant_format, activation_type, calibrate_method):
    """Quantize image encoder from a preprocessed ONNX file."""
    reader = ImageCalibrationReader()
    quantize_static(
        model_input=prep_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=quant_format,
        weight_type=QuantType.QInt8,
        activation_type=activation_type,
        calibrate_method=calibrate_method,
        per_channel=True,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],  # excludes Softmax and LayerNorm
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved: {output_path}  ({size_mb:.1f} MB)")


def quantize_text(prep_path, output_path, quant_format, activation_type, calibrate_method):
    """Quantize text encoder from a preprocessed ONNX file."""
    reader = TextCalibrationReader()
    quantize_static(
        model_input=prep_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=quant_format,
        weight_type=QuantType.QInt8,
        activation_type=activation_type,
        calibrate_method=calibrate_method,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],  # skip Gather (embeddings) and layer norm
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved: {output_path}  ({size_mb:.1f} MB)")


def run_for_model(model_name, calibrate_method, optimize_graph, combos_to_run):
    """Run all quantization combos for a single model variant."""
    if model_name == "ViT-B/16":
        slug = ""
        num_heads, hidden_size = 12, 768
    else:
        slug = "_" + model_name.lower().replace("/", "").replace("-", "")
        num_heads, hidden_size = 16, 1024

    image_onnx = os.path.join(ONNX_DIR, f"image_encoder{slug}.onnx")
    text_onnx  = os.path.join(ONNX_DIR, f"text_encoder{slug}.onnx")
    image_prep = os.path.join(ONNX_DIR, f"image_encoder{slug}_prep.onnx")
    text_prep  = os.path.join(ONNX_DIR, f"text_encoder{slug}_prep.onnx")

    for path in [image_onnx, text_onnx]:
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run export_onnx.py --model {model_name} first.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}  ({len(combos_to_run)} combo(s)  ×  2 encoders)")
    print(f"{'='*60}")

    # Preprocess once — shared across all combos for this model
    print(f"\n[Image Encoder] Preprocess...")
    image_model_input = preprocess_image(image_onnx, image_prep, optimize_graph,
                                          num_heads, hidden_size)
    print(f"\n[Text Encoder] Preprocess...")
    text_model_input = preprocess_text(text_onnx, text_prep)

    for quant_format, activation_type, fmt_name, act_name in combos_to_run:
        label = f"{fmt_name}_{act_name}"
        print(f"\n--- {model_name} | {fmt_name} | {act_name} ---")

        img_out = os.path.join(ONNX_DIR, f"image_encoder{slug}_int8_{fmt_name}_{act_name}.onnx")
        txt_out = os.path.join(ONNX_DIR, f"text_encoder{slug}_int8_{fmt_name}_{act_name}.onnx")

        print(f"[Image Encoder] Quantizing (Conv/MatMul/Gemm, {label})...")
        quantize_image(image_model_input, img_out, quant_format, activation_type, calibrate_method)

        print(f"[Text Encoder] Quantizing (MatMul/Gemm, {label})...")
        quantize_text(text_model_input, txt_out, quant_format, activation_type, calibrate_method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local INT8 Static Quantization for CLIP")
    parser.add_argument(
        "--model", default=None, choices=["ViT-B/16", "ViT-L/14"],
        help="CLIP model variant. If omitted, runs for both ViT-B/16 and ViT-L/14.",
    )
    parser.add_argument(
        "--format", default=None, choices=["qoperator", "qdq"],
        help="Quantization format. If omitted (default), all formats are run.",
    )
    parser.add_argument(
        "--activation", default=None, choices=["quint8", "qint8"],
        help="Activation quantization type. If omitted (default), all types are run.",
    )
    parser.add_argument(
        "--calibration", default="percentile", choices=["percentile", "minmax"],
        help=(
            "Calibration method:\n"
            "  percentile (default) — trims outliers, better for transformer attention.\n"
            "  minmax               — captures absolute worst-case range."
        ),
    )
    parser.add_argument(
        "--optimize-graph", action="store_true",
        help="Apply onnxruntime.transformers optimizer before quantization (fuses attention, LayerNorm, GELU).",
    )
    args = parser.parse_args()

    calibrate_method = (CalibrationMethod.Percentile if args.calibration == "percentile"
                        else CalibrationMethod.MinMax)

    # Determine which combos to run based on --format / --activation filters
    combos_to_run = []
    for quant_format, activation_type, fmt_name, act_name in COMBOS:
        if args.format and fmt_name != args.format:
            continue
        if args.activation and act_name != args.activation:
            continue
        combos_to_run.append((quant_format, activation_type, fmt_name, act_name))

    if not combos_to_run:
        print("Error: no combos match the specified --format / --activation filters.")
        sys.exit(1)

    models_to_run = [args.model] if args.model else ["ViT-B/16", "ViT-L/14"]

    print("=== Local ONNXRuntime INT8 Static Quantization ===")
    print(f"Models:      {models_to_run}")
    print(f"Combos:      {[(f, a) for _, _, f, a in combos_to_run]}")
    print(f"Calibration: {args.calibration}")
    total = len(models_to_run) * len(combos_to_run)
    print(f"Total runs:  {total} image encoder + {total} text encoder = {total*2} files")

    for model_name in models_to_run:
        run_for_model(model_name, calibrate_method, args.optimize_graph, combos_to_run)

    print("\n=== Quantization complete ===")
    print("Next: python inference_onnx_local.py --sweep")
