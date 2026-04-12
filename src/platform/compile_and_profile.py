import qai_hub
import onnx
import os
import sys
import argparse

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
    choices=["fp32", "fp16", "int8-compile", "int8-local"],
    help=(
        "Precision mode for compilation:\n"
        "  fp32         — default, no extra flags\n"
        "  fp16         — native FP16 on Hexagon HTP (--qnn_options default_graph_htp_precision=FLOAT16)\n"
        "  int8-compile — QAI Hub handles quantization at compile time (--quantize_full_type int8)\n"
        "  int8-local   — use locally quantized QDQ ONNX (same as --int8)"
    ),
)
args = parser.parse_args()

# Resolve --int8 legacy flag into --precision
if args.precision is None:
    args.precision = "int8-local" if args.int8 else "fp32"

from src.common.config import ONNX_DIR, DEVICE_NAME, ensure_output_dirs

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


def compile_model(model, device, input_specs, precision="fp32"):
    """Submits a compile job for the model and waits for completion."""
    options = get_compile_options(precision)
    print(f"  Compile options: {options}")
    compile_job = qai_hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=input_specs,
        options=options,
    )
    compile_job.wait()
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

# Submit compilation jobs
print("\nSubmitting compilation jobs to QAI Hub...")
img_job = compile_model(
    model=onnx_img_model,
    device=target_device,
    input_specs={"image": (1, 3, 224, 224)},
    precision=args.precision,
)
txt_job = compile_model(
    model=onnx_txt_model,
    device=target_device,
    input_specs={"text": ((1, 77), "int64")},
    precision=args.precision,
)

print(f"\n=== Job IDs ===")
print(f"Image compile job: {img_job.job_id}")
print(f"Text compile  job: {txt_job.job_id}")


# Submit profiling jobs
print("\nSubmitting profiling jobs to QAI Hub...")
run_profile(
    model=img_job.get_target_model(),
    device=target_device
)
run_profile(
    model=txt_job.get_target_model(),
    device=target_device
)
print("Profiling jobs submitted for both models.")
