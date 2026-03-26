"""
Unified pipeline: Compile → Profile → Inference on QAI Hub.

Compiles ONNX encoders, profiles latency, runs inference, and prints Recall@10.
Compile job IDs are wired automatically — no manual updates needed.

Usage:
    python run_on_device.py                                      # ViT-B/16 FP32
    python run_on_device.py --model ViT-L/14                    # ViT-L/14 FP32
    python run_on_device.py --int8                               # ViT-B/16 INT8
    python run_on_device.py --model ViT-L/14 --int8             # ViT-L/14 INT8
    python run_on_device.py --image-dataset-id dXXX --text-dataset-id dXXX  # custom datasets
"""

import sys
import os
import argparse
import onnx
import qai_hub
import numpy as np

from src.common.eval import evaluate_track1

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
    help="Use INT8 QDQ ONNX models (run quantize_local.py --format qdq first)",
)
parser.add_argument(
    "--image-dataset-id", default=None,
    help="QAI Hub dataset ID for image inputs (uses last known ID if omitted)",
)
parser.add_argument(
    "--text-dataset-id", default=None,
    help="QAI Hub dataset ID for text inputs (uses last known ID if omitted)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ONNX_DIR = "exported_onnx"
DATA_DIR = r"C:\rama\projects\data\lpcvc_track1_sample_data"
TXT_LIST = os.path.join(DATA_DIR, "txt_list.csv")
IMG_LIST = os.path.join(DATA_DIR, "img_list.csv")

TARGET_DEVICE = qai_hub.Device("XR2 Gen 2 (Proxy)")

# Last known dataset IDs — override with --image-dataset-id / --text-dataset-id
DEFAULT_IMAGE_DATASET_ID = "d2ne8er12"
DEFAULT_TEXT_DATASET_ID  = "d70krkm59"

slug       = "" if args.model == "ViT-B/16" else "_" + args.model.lower().replace("/", "").replace("-", "")
int8_suffix = "_int8" if args.int8 else ""

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


def compile_and_wait(model, input_specs):
    job = qai_hub.submit_compile_job(
        model=model,
        device=TARGET_DEVICE,
        input_specs=input_specs,
        options="--target_runtime qnn_dlc --truncate_64bit_io",
    )
    print(f"  Compile job submitted: {job.job_id}  (waiting...)")
    job.wait()
    status = job.get_status()
    if status.failure:
        print(f"  Compile FAILED: {status.message}")
        sys.exit(1)
    print(f"  Compile done: {job.job_id}")
    return job


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
print(f"\nModel: {args.model}  INT8: {args.int8}")
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

# --- Step 1: Compile ---
print("\n=== Compiling ===")
print("Image encoder:")
img_compile_job = compile_and_wait(onnx_img, {"image": (1, 3, 224, 224)})
print("Text encoder:")
txt_compile_job = compile_and_wait(onnx_txt, {"text": ((1, 77), "int64")})

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
print(f"Model:        {args.model}  {'INT8' if args.int8 else 'FP32'}")
print(f"Recall@10:    {recall:.4f}")
print(f"Image compile job: {img_compile_job.job_id}")
print(f"Text  compile job: {txt_compile_job.job_id}")
print(f"Image profile job: {img_profile_job.job_id}")
print(f"Text  profile job: {txt_profile_job.job_id}")
print("=" * 50)
