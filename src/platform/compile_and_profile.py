import qai_hub
import onnx
import os
import sys
import argparse
import numpy as np
import time

# Flush stdout immediately on every print — required on Windows where stdout is block-buffered
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="Compile and profile CLIP encoders on QAI Hub")
parser.add_argument(
    "--model", default="ViT-B/16", choices=["ViT-B/16", "ViT-L/14"],
    help="CLIP model variant to compile (default: ViT-B/16)",
)
parser.add_argument(
    "--int8", action="store_true",
    help="(Legacy) Compile INT8 QDQ ONNX models. Prefer --precision int8-local instead.",
)
parser.add_argument(
    "--precision", default=None,
    choices=["fp32", "fp16", "int8-compile", "int8-hub", "w8a16", "int8-local"],
    help=(
        "Precision mode for compilation:\n"
        "  fp32         — default, no extra flags\n"
        "  fp16         — native FP16 on Hexagon HTP (--qnn_options default_graph_htp_precision=FLOAT16)\n"
        "  int8-compile — QAI Hub handles quantization at compile time (--quantize_full_type int8)\n"
        "  int8-hub     — 2-step: QAI Hub quantize job (W8A8) then compile\n"
        "  w8a16        — 2-step: QAI Hub quantize job (W8A16) then compile\n"
        "  int8-local   — use locally quantized QDQ ONNX (same as --int8)"
    ),
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

# Resolve --int8 legacy flag into --precision
if args.precision is None:
    args.precision = "int8-local" if args.int8 else "fp32"

from src.common.config import ONNX_DIR, DEVICE_NAME, ensure_output_dirs
from src.common.config import CALIBRATION_SOURCE, CALIBRATION_N_SAMPLES
from src.common.calibration import load_calibration_data

ensure_output_dirs()

def run_profile(model, device):
    """Submit a profile job for the model."""
    profile_job = qai_hub.submit_profile_job(
        model=model,
        device=device,
        options="--max_profiler_iterations 100"
    )
    return profile_job.job_id

def get_compile_options(precision):
    """Return QAI Hub compile options based on precision mode."""
    base = "--target_runtime qnn_dlc --truncate_64bit_io"
    if precision == "fp16":
        return f"{base} --qnn_options default_graph_htp_precision=FLOAT16"
    elif precision == "int8-compile":
        return f"{base} --quantize_full_type int8"
    else:  # fp32, int8-local (QDQ ONNX already quantized)
        return base



def wait_with_status(job, label, poll_interval=15):
    """Poll job status every poll_interval seconds, printing updates until done."""
    start = time.time()
    last_code = None
    while True:
        job_status = job.get_status()
        code = job_status.code  # e.g. "CREATED", "OPTIMIZING_MODEL", "SUCCESS", "FAILED"
        elapsed = int(time.time() - start)
        if code != last_code:
            print(f"  [{label}] {code} ({elapsed}s elapsed)")
            last_code = code
        if job_status.finished:
            break
        time.sleep(poll_interval)
    if not job_status.success:
        print(f"  [{label}] Job ended with status: {code} — {job_status.message}")
        sys.exit(1)


NEEDS_QUANTIZE_JOB = {"int8-hub", "w8a16"}
NEEDS_CALIBRATION = {"int8-compile", "int8-hub", "w8a16"}

QUANTIZE_DTYPES = {
    "int8-hub": (qai_hub.QuantizeDtype.INT8, qai_hub.QuantizeDtype.INT8),
    "w8a16":   (qai_hub.QuantizeDtype.INT8, qai_hub.QuantizeDtype.INT16),
}


def compile_model(model, device, input_specs, precision="fp32", calibration_data=None):
    """Submits a compile job for the model and waits for completion."""
    options = get_compile_options(precision)
    print(f"  Compile options: {options}")
    compile_job = qai_hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=input_specs,
        options=options,
        calibration_data=calibration_data,
    )
    print(f"  Job submitted: {compile_job.job_id}")
    wait_with_status(compile_job, label=compile_job.job_id)
    return compile_job


def quantize_and_compile_model(model, device, input_specs, precision, calibration_data):
    """2-step pipeline: submit_quantize_job → submit_compile_job."""
    weights_dtype, activations_dtype = QUANTIZE_DTYPES[precision]
    print(f"  Quantize: weights={weights_dtype.name}  activations={activations_dtype.name}")
    quant_job = qai_hub.submit_quantize_job(
        model=model,
        calibration_data=calibration_data,
        weights_dtype=weights_dtype,
        activations_dtype=activations_dtype,
    )
    print(f"  Quantize job submitted: {quant_job.job_id}")
    wait_with_status(quant_job, label=f"quant-{quant_job.job_id}")
    quantized_model = quant_job.get_target_model()

    compile_job = qai_hub.submit_compile_job(
        model=quantized_model,
        device=device,
        input_specs=input_specs,
        options="--target_runtime qnn_dlc --truncate_64bit_io",
    )
    print(f"  Compile job submitted: {compile_job.job_id}")
    wait_with_status(compile_job, label=compile_job.job_id)
    return compile_job

# Derive paths from --model (matches export_onnx.py naming convention)
if args.model == "ViT-B/16":
    slug = ""
else:
    slug = "_" + args.model.lower().replace("/", "").replace("-", "")  # "_vitl14"

# For int8-local, use locally quantized QDQ ONNX; all others use FP32 ONNX
int8_suffix = "_int8" if args.precision == "int8-local" else ""
IMAGE_ONNX_PATH = os.path.join(ONNX_DIR, f"image_encoder{slug}{int8_suffix}.onnx")
TEXT_ONNX_PATH  = os.path.join(ONNX_DIR, f"text_encoder{slug}{int8_suffix}.onnx")

for path in [IMAGE_ONNX_PATH, TEXT_ONNX_PATH]:
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run export_onnx.py --model {args.model} first.")
        sys.exit(1)

print(f"Model: {args.model}  Precision: {args.precision}")


# Load the ONNX models from the new location

print(f"Loading ONNX Image Encoder from {IMAGE_ONNX_PATH}...")
onnx_img_model = onnx.load(IMAGE_ONNX_PATH)

# Check the model for errors
try:
    onnx.checker.check_model(onnx_img_model)
    print("Image ONNX model is valid ✅")
except onnx.checker.ValidationError as e:
    print("Image ONNX model validation failed ❌")
    print(e)

print(f"\nLoading ONNX Text Encoder from {TEXT_ONNX_PATH}...")
onnx_txt_model = onnx.load(TEXT_ONNX_PATH)

# Check the model for errors
try:
    onnx.checker.check_model(onnx_txt_model)
    print("Text ONNX model is valid ✅")
except onnx.checker.ValidationError as e:
    print("Text ONNX model validation failed ❌")
    print(e)

target_device = qai_hub.Device(DEVICE_NAME)

# Resolve calibration settings: CLI args take priority, fall back to config.py defaults
calib_source  = args.calib_source  or CALIBRATION_SOURCE
calib_samples = args.calib_samples or CALIBRATION_N_SAMPLES

# Load calibration data for precision modes that need it
img_calib, txt_calib = None, None
if args.precision in NEEDS_CALIBRATION:
    img_calib = load_calibration_data("image", calib_source, calib_samples)
    txt_calib = load_calibration_data("text",  calib_source, calib_samples)

# Submit compilation jobs
print(f"\nSubmitting {'quantize + compile' if args.precision in NEEDS_QUANTIZE_JOB else 'compile'} jobs to QAI Hub...")
if args.precision in NEEDS_QUANTIZE_JOB:
    img_job = quantize_and_compile_model(
        model=onnx_img_model,
        device=target_device,
        input_specs={"image": (1, 3, 224, 224)},
        precision=args.precision,
        calibration_data=img_calib,
    )
    txt_job = quantize_and_compile_model(
        model=onnx_txt_model,
        device=target_device,
        input_specs={"text": ((1, 77), "int64")},
        precision=args.precision,
        calibration_data=txt_calib,
    )
else:
    img_job = compile_model(
        model=onnx_img_model,
        device=target_device,
        input_specs={"image": (1, 3, 224, 224)},
        precision=args.precision,
        calibration_data=img_calib,
    )
    txt_job = compile_model(
        model=onnx_txt_model,
        device=target_device,
        input_specs={"text": ((1, 77), "int64")},
        precision=args.precision,
        calibration_data=txt_calib,
    )

print(f"\n=== Compile Job IDs ===")
print(f"Image compile job: {img_job.job_id}")
print(f"Text compile  job: {txt_job.job_id}")

# Submit profiling jobs
print("\nSubmitting profiling jobs to QAI Hub...")
img_profile_id = run_profile(model=img_job.get_target_model(), device=target_device)
txt_profile_id = run_profile(model=txt_job.get_target_model(), device=target_device)
print(f"\n=== Profile Job IDs ===")
print(f"Image profile job: {img_profile_id}")
print(f"Text profile  job: {txt_profile_id}")
