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
    help="Compile INT8 QDQ ONNX models (run quantize_local.py --format qdq first)",
)
args = parser.parse_args()

# --- Configuration ---
ONNX_DIR = "exported_onnx"
# ---------------------

def run_profile(model, device):
    """Submit a profile job for the model."""
    profile_job = qai_hub.submit_profile_job(
        model=model,
        device=device,
        options="--max_profiler_iterations 100"
    )
    return profile_job.job_id

def compile_model(model, device, input_specs):
    """Submits a compile job for the model and waits for completion."""
    compile_job = qai_hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=input_specs,
        options="--target_runtime qnn_dlc --truncate_64bit_io"
    )
    compile_job.wait()
    return compile_job

# Derive paths from --model (matches export_onnx.py naming convention)
if args.model == "ViT-B/16":
    slug = ""
else:
    slug = "_" + args.model.lower().replace("/", "").replace("-", "")  # "_vitl14"

int8_suffix = "_int8" if args.int8 else ""
IMAGE_ONNX_PATH = os.path.join(ONNX_DIR, f"image_encoder{slug}{int8_suffix}.onnx")
TEXT_ONNX_PATH  = os.path.join(ONNX_DIR, f"text_encoder{slug}{int8_suffix}.onnx")

for path in [IMAGE_ONNX_PATH, TEXT_ONNX_PATH]:
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run export_onnx.py --model {args.model} first.")
        sys.exit(1)

print(f"Model: {args.model}  INT8: {args.int8}")


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

target_device = qai_hub.Device("XR2 Gen 2 (Proxy)")

# Submit compilation jobs
print("\nSubmitting compilation jobs to QAI Hub...")
img_job = compile_model(
    model=onnx_img_model,
    device=target_device,
    input_specs={"image": (1, 3, 224, 224)}
)
txt_job = compile_model(
    model=onnx_txt_model,
    device=target_device,
    input_specs={"text": ((1, 77), "int64")}
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
