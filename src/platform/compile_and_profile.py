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
from src.common.config import IMAGE_DIR, IMG_LIST, TXT_LIST

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


def load_calibration_data(encoder):
    """Load sample dataset as calibration data for compile-time INT8 quantization.

    Returns a dict {input_name: list[np.ndarray]} with one (1, ...) array per sample.
    QAI Hub requires this format when --quantize_full_type is used.
    """
    import pandas as pd
    from PIL import Image
    from torchvision.transforms import Resize, CenterCrop, ToTensor, Compose
    from torchvision.transforms import InterpolationMode

    if encoder == "image":
        preprocess = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            ToTensor(),  # uint8 → float32, /255 — normalization is baked into the ONNX model
        ])
        df = pd.read_csv(IMG_LIST)
        samples = []
        for fname in df.iloc[:, 0].tolist():
            img = Image.open(os.path.join(IMAGE_DIR, fname)).convert("RGB")
            samples.append(preprocess(img).numpy()[np.newaxis])  # (1, 3, 224, 224)
        return {"image": samples}

    elif encoder == "text":
        # Use CLIP tokenizer — same tokenization as competition pipeline
        clip_path = os.path.join(os.path.dirname(__file__), "..", "..", "clip_model")
        sys.path.insert(0, clip_path)
        import clip as clip_module
        df = pd.read_csv(TXT_LIST)
        texts = df.iloc[:, 1].tolist()
        tokens = clip_module.tokenize(texts, truncate=True).numpy().astype(np.int64)  # (N, 77) int64 — matches ONNX input spec
        return {"text": [tok[np.newaxis] for tok in tokens]}  # list of (1, 77)

    else:
        raise ValueError(f"Unknown encoder: {encoder}")


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

# For compile-time INT8, calibration data is required — without it QAI Hub silently ignores
# the --quantize_full_type flag and compiles as FP32.
img_calib = load_calibration_data("image") if args.precision == "int8-compile" else None
txt_calib = load_calibration_data("text")  if args.precision == "int8-compile" else None

# Submit compilation jobs
print("\nSubmitting compilation jobs to QAI Hub...")
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
