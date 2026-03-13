"""
INT8 Quantization + Compilation + Profiling for CLIP ViT-B/16 on QAI Hub.

Steps:
  1. Prepare calibration data from sample dataset
  2. Submit INT8 quantize jobs (image + text encoders)
  3. Compile quantized models to QNN DLC for XR2 Gen 2
  4. Profile latency on device
"""

import sys
import os
import qai_hub
import onnx
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "clip_model")
import clip as clip_lib

# --- Configuration ---
ONNX_DIR = "exported_onnx"
# ONNX_DIR = r"C:\rama\projects\LPCVC2026-Edge-Retrieval-XR2\exported_onnx"
IMAGE_ONNX_PATH = os.path.join(ONNX_DIR, "image_encoder.onnx")
TEXT_ONNX_PATH = os.path.join(ONNX_DIR, "text_encoder.onnx")

DATA_DIR = r"C:\rama\projects\data\lpcvc_track1_sample_data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
IMG_LIST = os.path.join(DATA_DIR, "img_list.csv")
TXT_LIST = os.path.join(DATA_DIR, "txt_list.csv")

TARGET_DEVICE = qai_hub.Device("XR2 Gen 2 (Proxy)")
COMPILE_OPTIONS = "--target_runtime qnn_dlc --truncate_64bit_io --quantize_io"
PROFILE_OPTIONS = "--max_profiler_iterations 100"
# ---------------------


def prepare_image_calibration_data():
    """Load sample images as calibration data for INT8 quantization.
    Preprocessing: resize 224x224, /255 only (normalization baked into model).
    """
    df_img = pd.read_csv(IMG_LIST)
    filenames = df_img.iloc[:, 0].tolist()
    print(f"Preparing image calibration data: {len(filenames)} images...")

    images = []
    for f in filenames:
        img = Image.open(os.path.join(IMAGE_DIR, f)).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, :]  # (1, 3, 224, 224)
        images.append(arr)

    return {"image": images}


def prepare_text_calibration_data():
    """Load and tokenize text prompts as calibration data for INT8 quantization."""
    df_txt = pd.read_csv(TXT_LIST)
    prompts = df_txt.iloc[:, 1].dropna().tolist()
    print(f"Preparing text calibration data: {len(prompts)} prompts...")

    import torch
    tokens = clip_lib.tokenize(prompts)  # shape: [M, 77], int64
    # Split into list of (1, 77) int64 numpy arrays
    text_arrays = [tokens[i:i+1].numpy().astype(np.int64) for i in range(len(tokens))]

    return {"text": text_arrays}


# --- Main ---
if __name__ == "__main__":
    # Validate ONNX files exist
    if not os.path.exists(IMAGE_ONNX_PATH) or not os.path.exists(TEXT_ONNX_PATH):
        print(f"Error: ONNX files not found in '{ONNX_DIR}'. Run export_onnx.py first.")
        sys.exit(1)

    # Load ONNX models
    print(f"Loading image encoder from {IMAGE_ONNX_PATH}...")
    onnx_img_model = onnx.load(IMAGE_ONNX_PATH)
    print(f"Loading text encoder from {TEXT_ONNX_PATH}...")
    onnx_txt_model = onnx.load(TEXT_ONNX_PATH)

    # Prepare calibration data
    image_cal_data = prepare_image_calibration_data()
    text_cal_data = prepare_text_calibration_data()

    # --- Step 1: Quantize ---
    print("\n=== Submitting INT8 quantize jobs ===")

    print("Quantizing image encoder...")
    img_quantize_job = qai_hub.submit_quantize_job(
        model=onnx_img_model,
        calibration_data=image_cal_data,
        weights_dtype=qai_hub.QuantizeDtype.INT8,
        activations_dtype=qai_hub.QuantizeDtype.INT8,
    )
    print(f"  Image quantize job: {img_quantize_job.job_id}")
    quantized_img = img_quantize_job.get_target_model()
    print("  Image encoder quantized.")

    print("Quantizing text encoder...")
    txt_quantize_job = qai_hub.submit_quantize_job(
        model=onnx_txt_model,
        calibration_data=text_cal_data,
        weights_dtype=qai_hub.QuantizeDtype.INT8,
        activations_dtype=qai_hub.QuantizeDtype.INT8,
    )
    print(f"  Text quantize job: {txt_quantize_job.job_id}")
    quantized_txt = txt_quantize_job.get_target_model()
    print("  Text encoder quantized.")

    # --- Step 2: Compile ---
    print("\n=== Submitting compile jobs ===")

    print("Compiling quantized image encoder...")
    img_compile_job = qai_hub.submit_compile_job(
        model=quantized_img,
        device=TARGET_DEVICE,
        input_specs={"image": (1, 3, 224, 224)},
        options=COMPILE_OPTIONS,
    )
    print(f"  Image compile job: {img_compile_job.job_id}")

    print("Compiling quantized text encoder...")
    txt_compile_job = qai_hub.submit_compile_job(
        model=quantized_txt,
        device=TARGET_DEVICE,
        input_specs={"text": ((1, 77), "int64")},
        options=COMPILE_OPTIONS,
    )
    print(f"  Text compile job: {txt_compile_job.job_id}")

    # --- Step 3: Profile ---
    print("\n=== Submitting profile jobs ===")

    print("Profiling quantized image encoder...")
    img_profile_job = qai_hub.submit_profile_job(
        model=qai_hub.get_job(img_compile_job.job_id).get_target_model(),
        device=TARGET_DEVICE,
        options=PROFILE_OPTIONS,
    )
    print(f"  Image profile job: {img_profile_job.job_id}")

    print("Profiling quantized text encoder...")
    txt_profile_job = qai_hub.submit_profile_job(
        model=qai_hub.get_job(txt_compile_job.job_id).get_target_model(),
        device=TARGET_DEVICE,
        options=PROFILE_OPTIONS,
    )
    print(f"  Text profile job: {txt_profile_job.job_id}")

    # --- Summary ---
    print("\n" + "=" * 50)
    print("INT8 Quantization Results")
    print("=" * 50)
    print(f"Image Encoder:")
    print(f"  Quantize job:  {img_quantize_job.job_id}")
    print(f"  Compile job:   {img_compile_job.job_id}")
    print(f"  Profile job:   {img_profile_job.job_id}")
    print(f"Text Encoder:")
    print(f"  Quantize job:  {txt_quantize_job.job_id}")
    print(f"  Compile job:   {txt_compile_job.job_id}")
    print(f"  Profile job:   {txt_profile_job.job_id}")
    print()
    print("To run inference, update inference.py with:")
    print(f'  image compiled_id: "{img_compile_job.job_id}"')
    print(f'  text  compiled_id: "{txt_compile_job.job_id}"')
