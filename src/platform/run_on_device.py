"""
Unified pipeline: Compile → Profile → Inference on QAI Hub.

Compiles ONNX encoders, profiles latency, runs inference, and prints Recall@10.
Compile job IDs are wired automatically — no manual updates needed.

Usage:
    python run_on_device.py                                      # ViT-B/16 FP32
    python run_on_device.py --precision fp16                    # ViT-B/16 FP16 native
    python run_on_device.py --precision int8-compile            # ViT-B/16 INT8 (compile-time)
    python run_on_device.py --precision int8-hub                # ViT-B/16 INT8 (QAI Hub quantizer, W8A8)
    python run_on_device.py --precision w8a16                   # ViT-B/16 W8A16 (QAI Hub quantizer)
    python run_on_device.py --precision int8-local              # ViT-B/16 INT8 (local QDQ)
    python run_on_device.py --model ViT-L/14                    # ViT-L/14 FP32
    python run_on_device.py --image-dataset-id dXXX --text-dataset-id dXXX  # custom datasets
"""

import sys
import os
import argparse
import onnx
import qai_hub
import numpy as np

from src.common.eval import evaluate_track1
from src.common.config import (
    ONNX_DIR, TXT_LIST, IMG_LIST, DEVICE_NAME,
    DEFAULT_IMAGE_DATASET_ID, DEFAULT_TEXT_DATASET_ID,
    CALIBRATION_SOURCE, CALIBRATION_N_SAMPLES,
    ensure_output_dirs,
)

ensure_output_dirs()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compile + Profile + Inference on QAI Hub")
parser.add_argument(
    "--model", default="ViT-B/16", choices=["ViT-B/16", "ViT-L/14"],
    help="CLIP model variant (default: ViT-B/16)",
)
parser.add_argument(
    "--int8", action="store_true",
    help="(Legacy) Use INT8 QDQ ONNX models. Prefer --precision int8-local instead.",
)
parser.add_argument(
    "--precision", default=None,
    choices=["fp32", "fp16", "int8-compile", "int8-hub", "w8a16", "int8-local"],
    help=(
        "Precision mode for compilation:\n"
        "  fp32         — default, no extra flags\n"
        "  fp16         — native FP16 on Hexagon HTP\n"
        "  int8-compile — QAI Hub handles quantization at compile time (all ops)\n"
        "  int8-hub     — 2-step: QAI Hub quantize job (W8A8) then compile\n"
        "  w8a16        — 2-step: QAI Hub quantize job (W8A16) then compile\n"
        "  int8-local   — use locally quantized QDQ ONNX (same as --int8)"
    ),
)
parser.add_argument(
    "--image-dataset-id", default=None,
    help="QAI Hub dataset ID for image inputs (uses last known ID if omitted)",
)
parser.add_argument(
    "--text-dataset-id", default=None,
    help="QAI Hub dataset ID for text inputs (uses last known ID if omitted)",
)
parser.add_argument(
    "--calib-source", default=None, choices=["sample", "coco", "flickr30k"],
    help="Calibration data source for int8-compile (default: config.py CALIBRATION_SOURCE)",
)
parser.add_argument(
    "--calib-samples", type=int, default=None,
    help="Max calibration samples to use (default: config.py CALIBRATION_N_SAMPLES)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Configuration (paths and defaults from src/common/config.py)
# ---------------------------------------------------------------------------
TARGET_DEVICE = qai_hub.Device(DEVICE_NAME)

# Resolve --int8 legacy flag into --precision
if args.precision is None:
    args.precision = "int8-local" if args.int8 else "fp32"

# Calibration settings: CLI args take priority, fall back to config.py defaults
calib_source  = args.calib_source  or CALIBRATION_SOURCE
calib_samples = args.calib_samples or CALIBRATION_N_SAMPLES

slug       = "" if args.model == "ViT-B/16" else "_" + args.model.lower().replace("/", "").replace("-", "")
int8_suffix = "_int8" if args.precision == "int8-local" else ""

IMAGE_ONNX_PATH = os.path.join(ONNX_DIR, f"image_encoder{slug}{int8_suffix}.onnx")
TEXT_ONNX_PATH  = os.path.join(ONNX_DIR, f"text_encoder{slug}{int8_suffix}.onnx")

image_dataset_id = args.image_dataset_id or DEFAULT_IMAGE_DATASET_ID
text_dataset_id  = args.text_dataset_id  or DEFAULT_TEXT_DATASET_ID

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_value_info(model):
    """Remove tensors from value_info that also appear in model I/O (ONNX spec violation).

    quant_pre_process can leave output tensors (e.g. 'embedding') in value_info,
    which QAI Hub's compiler rejects with: 'Tensors occur in value_info but also in model IO'.
    """
    io_names = {t.name for t in model.graph.input} | {t.name for t in model.graph.output}
    culprits = [vi.name for vi in model.graph.value_info if vi.name in io_names]
    if culprits:
        new_vi = [vi for vi in model.graph.value_info if vi.name not in io_names]
        del model.graph.value_info[:]
        model.graph.value_info.extend(new_vi)
        print(f"  Cleaned value_info: removed {culprits}")
    return model


def get_compile_options(precision):
    """Return QAI Hub compile options based on precision mode."""
    base = "--target_runtime qnn_dlc --truncate_64bit_io"
    if precision == "fp16":
        return f"{base} --qnn_options default_graph_htp_precision=FLOAT16"
    elif precision == "int8-compile":
        return f"{base} --quantize_full_type int8"
    else:  # fp32, int8-local (QDQ ONNX already quantized)
        return base


def compile_and_wait(model, input_specs, precision, calibration_data=None):
    options = get_compile_options(precision)
    print(f"  Compile options: {options}")
    job = qai_hub.submit_compile_job(
        model=model,
        device=TARGET_DEVICE,
        input_specs=input_specs,
        options=options,
        calibration_data=calibration_data,
    )
    print(f"  Compile job submitted: {job.job_id}  (waiting...)")
    job.wait()
    status = job.get_status()
    if status.failure:
        print(f"  Compile FAILED: {status.message}")
        sys.exit(1)
    print(f"  Compile done: {job.job_id}")
    return job


NEEDS_QUANTIZE_JOB = {"int8-hub", "w8a16"}
NEEDS_CALIBRATION = {"int8-compile", "int8-hub", "w8a16"}

QUANTIZE_DTYPES = {
    "int8-hub": (qai_hub.QuantizeDtype.INT8, qai_hub.QuantizeDtype.INT8),
    "w8a16":   (qai_hub.QuantizeDtype.INT8, qai_hub.QuantizeDtype.INT16),
}


def quantize_and_compile(model, input_specs, precision, calibration_data):
    """2-step pipeline: submit_quantize_job → submit_compile_job."""
    weights_dtype, activations_dtype = QUANTIZE_DTYPES[precision]
    print(f"  Quantize: weights={weights_dtype.name}  activations={activations_dtype.name}")
    quant_job = qai_hub.submit_quantize_job(
        model=model,
        calibration_data=calibration_data,
        weights_dtype=weights_dtype,
        activations_dtype=activations_dtype,
    )
    print(f"  Quantize job submitted: {quant_job.job_id}  (waiting...)")
    quant_job.wait()
    status = quant_job.get_status()
    if status.failure:
        print(f"  Quantize FAILED: {status.message}")
        sys.exit(1)
    print(f"  Quantize done: {quant_job.job_id}")
    quantized_model = quant_job.get_target_model()

    compile_job = qai_hub.submit_compile_job(
        model=quantized_model,
        device=TARGET_DEVICE,
        input_specs=input_specs,
        options="--target_runtime qnn_dlc --truncate_64bit_io",
    )
    print(f"  Compile job submitted: {compile_job.job_id}  (waiting...)")
    compile_job.wait()
    status = compile_job.get_status()
    if status.failure:
        print(f"  Compile FAILED: {status.message}")
        sys.exit(1)
    print(f"  Compile done: {compile_job.job_id}")
    return compile_job


def submit_profile(compiled_model):
    job = qai_hub.submit_profile_job(
        model=compiled_model,
        device=TARGET_DEVICE,
        options="--max_profiler_iterations 100",
    )
    print(f"  Profile job submitted: {job.job_id}")
    return job


def run_inference(compiled_model, dataset):
    job = qai_hub.submit_inference_job(
        model=compiled_model,
        device=TARGET_DEVICE,
        inputs=dataset,
    )
    print(f"  Inference job submitted: {job.job_id}  (waiting...)")
    job.wait()
    status = job.get_status()
    if status.failure:
        print(f"  Inference FAILED: {status.message}")
        return None
    return job.download_output_data()["output_0"]


def print_profile_summary(name, profile_job):
    print(f"\n--- {name} profile ---")
    profile_job.wait()
    status = profile_job.get_status()
    if status.failure:
        print(f"  Profile job failed: {status.message}")
        return
    try:
        data = profile_job.download_profile()
        # QAI Hub SDK may return a list (one entry per graph); unwrap if so
        if isinstance(data, list):
            data = data[0] if data else {}
        # Total execution time is under execution_summary
        summary = data.get("execution_summary", {})
        total_ms = summary.get("estimated_inference_time_ms") or summary.get("inference_time_ms")
        if total_ms is not None:
            print(f"  Estimated latency: {total_ms:.2f} ms")
        else:
            # Fallback: sum layer times
            layers = data.get("execution_detail", {}).get("layers", [])
            if layers:
                total_us = sum(l.get("execution_time", 0) for l in layers)
                print(f"  Total layer time:  {total_us / 1000:.2f} ms  ({len(layers)} layers)")
            else:
                print(f"  Profile job: {profile_job.job_id} (check QAI Hub dashboard)")
    except Exception as e:
        print(f"  Could not parse profile data: {e}")
        print(f"  Profile job ID: {profile_job.job_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print(f"\nModel: {args.model}  Precision: {args.precision}")
print(f"Image ONNX: {IMAGE_ONNX_PATH}")
print(f"Text  ONNX: {TEXT_ONNX_PATH}")
print(f"Image dataset: {image_dataset_id}")
print(f"Text  dataset: {text_dataset_id}")

# Validate ONNX files
for path in [IMAGE_ONNX_PATH, TEXT_ONNX_PATH]:
    if not os.path.exists(path):
        print(f"\nError: {path} not found.")
        hint = f"export_onnx.py --model {args.model}"
        if args.int8:
            hint += f"  then  quantize_local.py --format qdq"
        print(f"Run: {hint}")
        sys.exit(1)

# Load ONNX
print("\nLoading ONNX models...")
onnx_img = clean_value_info(onnx.load(IMAGE_ONNX_PATH))
onnx_txt = clean_value_info(onnx.load(TEXT_ONNX_PATH))

# --- Step 1: Compile (or Quantize + Compile) ---
img_calib, txt_calib = None, None
if args.precision in NEEDS_CALIBRATION:
    from src.common.calibration import load_calibration_data
    print(f"\nLoading calibration data ({calib_source}, max {calib_samples} samples)...")
    img_calib = load_calibration_data("image", calib_source, calib_samples)
    txt_calib = load_calibration_data("text",  calib_source, calib_samples)

print(f"\n=== {'Quantizing + Compiling' if args.precision in NEEDS_QUANTIZE_JOB else 'Compiling'} ===")
if args.precision in NEEDS_QUANTIZE_JOB:
    print("Image encoder:")
    img_compile_job = quantize_and_compile(onnx_img, {"image": (1, 3, 224, 224)}, args.precision, img_calib)
    print("Text encoder:")
    txt_compile_job = quantize_and_compile(onnx_txt, {"text": ((1, 77), "int64")}, args.precision, txt_calib)
else:
    print("Image encoder:")
    img_compile_job = compile_and_wait(onnx_img, {"image": (1, 3, 224, 224)}, args.precision, img_calib)
    print("Text encoder:")
    txt_compile_job = compile_and_wait(onnx_txt, {"text": ((1, 77), "int64")}, args.precision, txt_calib)

img_compiled_model = img_compile_job.get_target_model()
txt_compiled_model = txt_compile_job.get_target_model()

# --- Step 2: Profile (submit now, read results after inference) ---
print("\n=== Submitting profile jobs ===")
print("Image encoder:")
img_profile_job = submit_profile(img_compiled_model)
print("Text encoder:")
txt_profile_job = submit_profile(txt_compiled_model)

# --- Step 3: Inference ---
print("\n=== Running inference ===")
image_dataset = qai_hub.get_dataset(image_dataset_id)
text_dataset  = qai_hub.get_dataset(text_dataset_id)

print("Image encoder:")
image_output = run_inference(img_compiled_model, image_dataset)
print("Text encoder:")
text_output  = run_inference(txt_compiled_model, text_dataset)

if image_output is None or text_output is None:
    print("\nInference failed — cannot compute Recall@10.")
    sys.exit(1)

# --- Step 4: Recall@10 ---
print("\n=== Recall@10 ===")
recall = evaluate_track1(image_output, text_output, TXT_LIST, IMG_LIST)
print(f"Recall@10: {recall:.4f}")

# --- Step 5: Profile results (jobs have been running in parallel with inference) ---
print_profile_summary("Image encoder", img_profile_job)
print_profile_summary("Text encoder",  txt_profile_job)

# --- Summary ---
print("\n" + "=" * 50)
print(f"Model:        {args.model}  {args.precision}")
print(f"Recall@10:    {recall:.4f}")
print(f"Image compile job: {img_compile_job.job_id}")
print(f"Text  compile job: {txt_compile_job.job_id}")
print(f"Image profile job: {img_profile_job.job_id}")
print(f"Text  profile job: {txt_profile_job.job_id}")
print("=" * 50)
