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

AIMET INT8 (Phase 2 scope-restricted builds):
    # one mixed-tag pair
    python run_on_device.py --precision int8-aimet \
        --image-int8-tag defscope --text-int8-tag defscope

    # all four accuracy-tied pairs, one comparison table (the latency decision)
    python run_on_device.py --precision int8-aimet --sweep-pairs

Device defaults to config.DEVICE_NAME (Samsung Galaxy S22 family); override with
--device or the QAI_HUB_DEVICE env var.
"""

import sys
import os
import argparse
from datetime import datetime

import onnx
import qai_hub
import numpy as np

from src.common.eval import evaluate_track1
from src.common.config import (
    ONNX_DIR, TXT_LIST, IMG_LIST, DEVICE_NAME, PROFILE_LOG_DIR,
    DEFAULT_IMAGE_DATASET_ID, DEFAULT_TEXT_DATASET_ID,
    CALIBRATION_SOURCE, CALIBRATION_N_SAMPLES, LATENCY_BUDGET_MS,
    ensure_output_dirs,
)

ensure_output_dirs()

# ---------------------------------------------------------------------------
# AIMET scope pairs (Phase 2, plans_notes/aimet_quantization_review_2026-08-04.md)
#
# Local ORT Recall@10 for each pair, both encoders INT8, measured 2026-08-11
# against an in-process FP32/FP32 reference of 0.8728. The four pairs span
# 0.8496-0.8610; one image is worth ~0.018 on a 56-image set, so they are
# ACCURACY-TIED and the device latency decides which one ships.
#
# DEVICE RUNTIME (2026-08-12): these builds MUST be compiled to a QNN context
# binary, not a DLC. Every scope-restricted build is a MIXED float/int8 graph
# (LayerNorm / Add / Softmax stay float), and the HTP cannot compose a mixed
# graph out of a DLC: the compile job succeeds, then every profile and inference
# job dies with
#     Failed to call QnnModel_composeGraphsFromDlc: MODEL_GRAPH_ERROR
# which is what made the 2026-08-12 sweep return four bare "job failed" rows.
# Measured on the image encoder, Galaxy S22:
#   defscope        + qnn_dlc            -> MODEL_GRAPH_ERROR   (jp437xy25)
#   defscope        + FLOAT16 fallback   -> MODEL_GRAPH_ERROR   (jpv78z7rp)
#   defscope        + qnn_context_binary -> 55.93 ms            (jpeyq4685)
#   all-ops int8    + qnn_dlc            -> 21.26 ms            (jgo8xe8kp)
# Per-channel MatMul weights were ruled out: folding them to per-tensor still
# gives MODEL_GRAPH_ERROR under qnn_dlc (jgnk4r6rg, jglldev8g). The float ops
# are also expensive — see the 55.93 ms above against 20.16 ms for FP32.
#
# defscope    = Conv, ConvTranspose, MatMul, Gemm
# addsoftmax  = defscope + Softmax        (image only; best solo image scope)
# bestscope   = defscope + LayerNormalization, Softmax  (text only)
# LayerNormalization must stay float in the IMAGE encoder under any scope.
# ---------------------------------------------------------------------------
DEFAULT_AIMET_IMAGE_TAG = "defscope"
DEFAULT_AIMET_TEXT_TAG  = "defscope"

# (image_tag, text_tag) -> local ORT Recall@10 for the shippable INT8/INT8 pair
PAIR_MATRIX = {
    ("defscope",   "defscope"):  0.8610,
    ("defscope",   "bestscope"): 0.8592,
    ("addsoftmax", "defscope"):  0.8589,
    ("addsoftmax", "bestscope"): 0.8496,
}

# Local reference points for the comparison table.
LOCAL_FP32_ONNX_RECALL = 0.8728
LOCAL_PYTORCH_RECALL   = 0.8805

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
    choices=["fp32", "fp16", "int8-compile", "int8-hub", "w8a16", "int8-local", "int8-aimet"],
    help=(
        "Precision mode for compilation:\n"
        "  fp32         — default, no extra flags\n"
        "  fp16         — native FP16 on Hexagon HTP\n"
        "  int8-compile — QAI Hub handles quantization at compile time (all ops)\n"
        "  int8-hub     — 2-step: QAI Hub quantize job (W8A8) then compile\n"
        "  w8a16        — 2-step: QAI Hub quantize job (W8A16) then compile\n"
        "  int8-local   — use locally quantized QDQ ONNX (same as --int8)\n"
        "  int8-aimet   — AIMET scope-restricted QDQ ONNX, tag-selected per encoder"
    ),
)
parser.add_argument(
    "--device", default=None,
    help=f"QAI Hub device name (default: {DEVICE_NAME})",
)
parser.add_argument(
    "--image-int8-tag", default=DEFAULT_AIMET_IMAGE_TAG,
    help=f"AIMET variant tag for the image encoder (default: {DEFAULT_AIMET_IMAGE_TAG})",
)
parser.add_argument(
    "--text-int8-tag", default=DEFAULT_AIMET_TEXT_TAG,
    help=f"AIMET variant tag for the text encoder (default: {DEFAULT_AIMET_TEXT_TAG})",
)
parser.add_argument(
    "--aimet-runtime", default="qnn_context_binary",
    choices=["qnn_context_binary", "qnn_dlc"],
    help=(
        "Target runtime for --precision int8-aimet (default: qnn_context_binary). "
        "The scope-restricted AIMET builds are MIXED float/int8 graphs, and a "
        "mixed graph cannot be composed from a DLC on the HTP — every profile and "
        "inference job dies with 'QnnModel_composeGraphsFromDlc: MODEL_GRAPH_ERROR' "
        "even though the compile job succeeds. See the note above run_once()."
    ),
)
parser.add_argument(
    "--sweep-pairs", action="store_true",
    help=(
        "Run every (image, text) AIMET tag pair in PAIR_MATRIX and print one "
        "comparison table. These four scopes are accuracy-tied locally, so the "
        "device latency is what actually selects between them."
    ),
)
parser.add_argument(
    "--dry-run", action="store_true",
    help=(
        "Validate artifacts and datasets and render the results table with "
        "placeholder numbers, without submitting any QAI Hub job. A sweep is "
        "12+ device jobs — check the plumbing first."
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
device_name = args.device or DEVICE_NAME
TARGET_DEVICE = qai_hub.Device(device_name)
aimet_runtime = args.aimet_runtime

# Resolve --int8 legacy flag into --precision
if args.precision is None:
    args.precision = "int8-local" if args.int8 else "fp32"

# Calibration settings: CLI args take priority, fall back to config.py defaults
calib_source  = args.calib_source  or CALIBRATION_SOURCE
calib_samples = args.calib_samples or CALIBRATION_N_SAMPLES

slug       = "" if args.model == "ViT-B/16" else "_" + args.model.lower().replace("/", "").replace("-", "")
int8_suffix = "_int8" if args.precision == "int8-local" else ""


def aimet_paths(image_tag, text_tag):
    """Artifact paths for an AIMET (image_tag, text_tag) scope pair.

    Tags are per-encoder because the best scope differs between towers, so the
    shippable build is generally a MIXED-tag pair.
    """
    return (
        os.path.join(ONNX_DIR, f"image_encoder{slug}_int8_aimet_{image_tag}.onnx"),
        os.path.join(ONNX_DIR, f"text_encoder{slug}_int8_aimet_{text_tag}.onnx"),
    )


if args.precision == "int8-aimet":
    IMAGE_ONNX_PATH, TEXT_ONNX_PATH = aimet_paths(args.image_int8_tag, args.text_int8_tag)
else:
    IMAGE_ONNX_PATH = os.path.join(ONNX_DIR, f"image_encoder{slug}{int8_suffix}.onnx")
    TEXT_ONNX_PATH  = os.path.join(ONNX_DIR, f"text_encoder{slug}{int8_suffix}.onnx")

if args.sweep_pairs and args.precision != "int8-aimet":
    print("--sweep-pairs only applies to --precision int8-aimet")
    sys.exit(2)

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
    base = f"--target_runtime {aimet_runtime if precision == 'int8-aimet' else 'qnn_dlc'}"
    base += " --truncate_64bit_io"
    if precision == "fp16":
        return f"{base} --qnn_options default_graph_htp_precision=FLOAT16"
    elif precision == "int8-compile":
        return f"{base} --quantize_full_type int8"
    else:  # fp32, int8-local, int8-aimet (QDQ ONNX already quantized)
        return base


class JobFailure(Exception):
    """A QAI Hub job failed. Carries the message so the results table can show it.

    Previously every failure path called sys.exit(1) and --sweep-pairs recorded a
    bare 'job failed', which threw away the only useful part: WHY. That is how a
    whole sweep came back with four identical 'job failed' rows and no diagnosis.
    """

    def __init__(self, stage, job, message):
        super().__init__(f"{stage} {job.job_id}: {message}")
        self.stage = stage
        self.job_id = job.job_id
        self.message = message

    def short(self):
        return f"{self.stage} failed ({self.job_id}): {self.message.splitlines()[0][:70]}"


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
        raise JobFailure("compile", job, status.message)
    print(f"  Compile done: {job.job_id}")
    return job


NEEDS_QUANTIZE_JOB = {"int8-hub", "w8a16"}
NEEDS_CALIBRATION = {"int8-compile", "int8-hub", "w8a16"}

QUANTIZE_DTYPES = {
    "int8-hub": (qai_hub.QuantizeDtype.INT8, qai_hub.QuantizeDtype.INT8),
    "w8a16":   (qai_hub.QuantizeDtype.INT8, qai_hub.QuantizeDtype.INT16),
}


def quantize_and_compile(model, input_specs, precision, calibration_data):
    """3-job pipeline per QAI Hub docs: optimize ONNX → quantize → compile to QNN DLC.

    The first compile-to-ONNX pass runs the Hub optimizer (fuses LayerNorm /
    Attention / GELU) before quantization, which materially improves quantized
    accuracy versus feeding the raw exported ONNX into submit_quantize_job.
    """
    # Step 1: pre-quantize optimization compile (ONNX → optimized ONNX)
    optimize_job = qai_hub.submit_compile_job(
        model=model,
        device=TARGET_DEVICE,
        input_specs=input_specs,
        options="--target_runtime onnx",
    )
    print(f"  Optimize (ONNX) job submitted: {optimize_job.job_id}  (waiting...)")
    optimize_job.wait()
    status = optimize_job.get_status()
    if status.failure:
        print(f"  Optimize FAILED: {status.message}")
        raise JobFailure("optimize", optimize_job, status.message)
    print(f"  Optimize done: {optimize_job.job_id}")
    optimized_model = optimize_job.get_target_model()

    # Step 2: quantize the optimized ONNX
    weights_dtype, activations_dtype = QUANTIZE_DTYPES[precision]
    print(f"  Quantize: weights={weights_dtype.name}  activations={activations_dtype.name}")
    quant_job = qai_hub.submit_quantize_job(
        model=optimized_model,
        calibration_data=calibration_data,
        weights_dtype=weights_dtype,
        activations_dtype=activations_dtype,
    )
    print(f"  Quantize job submitted: {quant_job.job_id}  (waiting...)")
    quant_job.wait()
    status = quant_job.get_status()
    if status.failure:
        print(f"  Quantize FAILED: {status.message}")
        raise JobFailure("quantize", quant_job, status.message)
    print(f"  Quantize done: {quant_job.job_id}")
    quantized_model = quant_job.get_target_model()

    # Step 3: final compile to QNN DLC. --quantize_io intentionally OFF to
    # preserve float32 IO (competition sends fp32 images; we read fp32 embeddings).
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
        raise JobFailure("compile", compile_job, status.message)
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
        raise JobFailure("inference", job, status.message)
    return job.download_output_data()["output_0"]


def print_profile_summary(name, profile_job):
    """Print the profile result and return latency in ms.

    Returns (ms, error). A profile failure is not fatal — Recall@10 is still
    worth having — but the reason travels back so the table can show it.
    """
    print(f"\n--- {name} profile ---")
    profile_job.wait()
    status = profile_job.get_status()
    if status.failure:
        print(f"  Profile job failed: {status.message}")
        return None, JobFailure("profile", profile_job, status.message).short()
    try:
        data = profile_job.download_profile()
        # QAI Hub SDK may return a list (one entry per graph); unwrap if so
        if isinstance(data, list):
            data = data[0] if data else {}
        # execution_summary.estimated_inference_time is in MICROSECONDS — that is
        # the key the SDK itself reads (qai_hub/client.py divides it by 1000).
        # The older `estimated_inference_time_ms` spelling never existed, so this
        # branch always fell through to the layer-sum fallback below.
        summary = data.get("execution_summary", {})
        total_us = summary.get("estimated_inference_time")
        if total_us is not None:
            total_ms = total_us / 1000.0
            print(f"  Estimated latency: {total_ms:.2f} ms")
            return total_ms, None
        # Fallback: sum layer times
        layers = data.get("execution_detail", {}).get("layers", [])
        if layers:
            total_us = sum(l.get("execution_time", 0) for l in layers)
            print(f"  Total layer time:  {total_us / 1000:.2f} ms  ({len(layers)} layers)")
            return total_us / 1000.0, None
        print(f"  Profile job: {profile_job.job_id} (check QAI Hub dashboard)")
    except Exception as e:
        print(f"  Could not parse profile data: {e}")
        print(f"  Profile job ID: {profile_job.job_id}")
    return None, "profile data unparseable"


# ---------------------------------------------------------------------------
# One end-to-end run: compile -> profile -> inference -> Recall@10
# ---------------------------------------------------------------------------
def run_once(image_onnx_path, text_onnx_path, label):
    """Compile, profile and score one (image, text) encoder pair.

    Returns a dict of results. Raises JobFailure on a compile/inference failure,
    which --sweep-pairs catches (with the message) so one bad pair does not
    abandon the others. A profile failure is recorded, not raised.
    """
    print("\n" + "#" * 70)
    print(f"# {label}")
    print("#" * 70)
    print(f"Model: {args.model}  Precision: {args.precision}  Device: {device_name}")
    print(f"Image ONNX: {image_onnx_path}")
    print(f"Text  ONNX: {text_onnx_path}")
    print(f"Image dataset: {image_dataset_id}")
    print(f"Text  dataset: {text_dataset_id}")

    # Load ONNX
    print("\nLoading ONNX models...")
    onnx_img = clean_value_info(onnx.load(image_onnx_path))
    onnx_txt = clean_value_info(onnx.load(text_onnx_path))

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

    # --- Step 4: Recall@10 ---
    print("\n=== Recall@10 ===")
    recall = evaluate_track1(image_output, text_output, TXT_LIST, IMG_LIST)
    print(f"Recall@10: {recall:.4f}")

    # --- Step 5: Profile results (ran in parallel with inference) ---
    img_ms, img_profile_err = print_profile_summary("Image encoder", img_profile_job)
    txt_ms, txt_profile_err = print_profile_summary("Text encoder",  txt_profile_job)
    total_ms = None if img_ms is None or txt_ms is None else img_ms + txt_ms
    profile_err = img_profile_err or txt_profile_err

    # --- Summary ---
    print("\n" + "=" * 50)
    print(f"Model:        {args.model}  {args.precision}")
    print(f"Device:       {device_name}")
    print(f"Recall@10:    {recall:.4f}")
    if total_ms is not None:
        verdict = "PASS" if total_ms <= LATENCY_BUDGET_MS else "OVER BUDGET"
        print(f"Latency:      {img_ms:.2f} + {txt_ms:.2f} = {total_ms:.2f} ms  "
              f"[{verdict}, budget {LATENCY_BUDGET_MS:.0f} ms]")
    print(f"Image compile job: {img_compile_job.job_id}")
    print(f"Text  compile job: {txt_compile_job.job_id}")
    print(f"Image profile job: {img_profile_job.job_id}")
    print(f"Text  profile job: {txt_profile_job.job_id}")
    print("=" * 50)

    return {
        "label": label,
        "recall": recall,
        "img_ms": img_ms,
        "txt_ms": txt_ms,
        "total_ms": total_ms,
        "img_compile_job": img_compile_job.job_id,
        "txt_compile_job": txt_compile_job.job_id,
        "img_profile_job": img_profile_job.job_id,
        "txt_profile_job": txt_profile_job.job_id,
        "profile_error": profile_err,
    }


def require_files(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    for p in missing:
        print(f"\nError: {p} not found.")
    if missing:
        if args.precision == "int8-aimet":
            print("Run: src/local/quantize_aimet.py --encoder image|text --variant <tag>")
        else:
            hint = f"export_onnx.py --model {args.model}"
            if args.int8:
                hint += "  then  quantize_local.py --format qdq"
            print(f"Run: {hint}")
        sys.exit(1)


def write_results_table(results, path):
    """Device results next to the local ORT numbers they are meant to check."""
    lines = []
    lines.append(f"# On-device results — {args.model} {args.precision}")
    lines.append(f"# Device: {device_name}")
    lines.append(f"# Datasets: image={image_dataset_id}  text={text_dataset_id}")
    lines.append(f"# Latency budget: {LATENCY_BUDGET_MS:.0f} ms combined")
    lines.append("")
    lines.append(f"  {'pair':<30} {'local R@10':>10} {'device R@10':>12} "
                 f"{'img ms':>8} {'txt ms':>8} {'total ms':>9}  verdict")
    lines.append("  " + "-" * 88)
    lines.append(f"  {'PyTorch FP32 (local)':<30} {LOCAL_PYTORCH_RECALL:>10.4f} "
                 f"{'—':>12} {'—':>8} {'—':>8} {'—':>9}")
    lines.append(f"  {'ONNX FP32 (local ORT)':<30} {LOCAL_FP32_ONNX_RECALL:>10.4f} "
                 f"{'—':>12} {'—':>8} {'—':>8} {'—':>9}")
    for r in results:
        local = r.get("local_recall")
        local_s = f"{local:.4f}" if local is not None else "—"
        if r.get("error"):
            lines.append(f"  {r['label']:<30} {local_s:>10} {'FAILED':>12} "
                         f"{'—':>8} {'—':>8} {'—':>9}  {r['error']}")
            continue
        img_s = f"{r['img_ms']:.2f}" if r["img_ms"] is not None else "—"
        txt_s = f"{r['txt_ms']:.2f}" if r["txt_ms"] is not None else "—"
        tot_s = f"{r['total_ms']:.2f}" if r["total_ms"] is not None else "—"
        verdict = ""
        if r["total_ms"] is not None:
            verdict = "PASS" if r["total_ms"] <= LATENCY_BUDGET_MS else "OVER BUDGET"
        if not verdict and r.get("profile_error"):
            verdict = r["profile_error"]
        lines.append(f"  {r['label']:<30} {local_s:>10} {r['recall']:>12.4f} "
                     f"{img_s:>8} {txt_s:>8} {tot_s:>9}  {verdict}")
    lines.append("")
    lines.append("  Job IDs")
    for r in results:
        if r.get("error"):
            continue
        lines.append(f"    {r['label']:<30} compile img={r['img_compile_job']} txt={r['txt_compile_job']}"
                     f"  profile img={r['img_profile_job']} txt={r['txt_profile_job']}")
    table = "\n".join(lines)
    print("\n" + table)
    with open(path, "w") as f:
        f.write(table + "\n")
    print(f"\nWritten to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def dry_run(pairs):
    """Validate everything the real run depends on; submit nothing."""
    print(f"\nDRY RUN — device {device_name}, precision {args.precision}")
    require_files([p for it, tt in pairs for p in aimet_paths(*(it, tt))]
                  if args.precision == "int8-aimet"
                  else [IMAGE_ONNX_PATH, TEXT_ONNX_PATH])
    print("  artifacts: OK")

    qai_hub.Device(device_name)
    print(f"  device {device_name!r}: OK")
    for kind, did in [("image", image_dataset_id), ("text", text_dataset_id)]:
        qai_hub.get_dataset(did)
        print(f"  {kind} dataset {did}: OK")

    fake = []
    for i, (image_tag, text_tag) in enumerate(pairs):
        label = (f"img={image_tag} txt={text_tag}" if args.precision == "int8-aimet"
                 else f"{args.model} {args.precision}")
        fake.append({
            "label": label, "recall": 0.0, "img_ms": 10.0 + i, "txt_ms": 4.0,
            "total_ms": 14.0 + i,
            "img_compile_job": "jDRY", "txt_compile_job": "jDRY",
            "img_profile_job": "jDRY", "txt_profile_job": "jDRY",
            "local_recall": (PAIR_MATRIX.get((image_tag, text_tag))
                             if args.precision == "int8-aimet" else None),
        })
    fake.append({"label": "failure-path check", "error": "job failed", "local_recall": None})
    write_results_table(fake, os.path.join(PROFILE_LOG_DIR, f"dryrun_{stamp}.txt"))
    print("\nDry run OK — numbers above are placeholders, nothing was submitted.")


if args.dry_run:
    dry_run(list(PAIR_MATRIX) if args.sweep_pairs
            else [(args.image_int8_tag, args.text_int8_tag)])
    sys.exit(0)

if args.sweep_pairs:
    pairs = list(PAIR_MATRIX)
    require_files([p for it, tt in pairs for p in aimet_paths(it, tt)])
    print(f"\nSweeping {len(pairs)} AIMET scope pairs on {device_name}.")
    print("These are accuracy-tied locally (0.8496-0.8610); latency is the decision.\n")

    results = []
    for image_tag, text_tag in pairs:
        img_path, txt_path = aimet_paths(image_tag, text_tag)
        label = f"img={image_tag} txt={text_tag}"
        try:
            r = run_once(img_path, txt_path, label)
        except JobFailure as e:
            # One failed pair must not abandon the remaining three — but keep the
            # reason, which is the whole point of running the sweep unattended.
            print(f"\n!! Pair {label} failed: {e}  — continuing with the rest.")
            r = {"label": label, "error": e.short()}
        r["local_recall"] = PAIR_MATRIX[(image_tag, text_tag)]
        results.append(r)

    out = os.path.join(PROFILE_LOG_DIR, f"aimet_pair_sweep_{stamp}.txt")
    write_results_table(results, out)
else:
    require_files([IMAGE_ONNX_PATH, TEXT_ONNX_PATH])
    if args.precision == "int8-aimet":
        label = f"img={args.image_int8_tag} txt={args.text_int8_tag}"
        local_recall = PAIR_MATRIX.get((args.image_int8_tag, args.text_int8_tag))
    else:
        label = f"{args.model} {args.precision}"
        local_recall = None
    try:
        result = run_once(IMAGE_ONNX_PATH, TEXT_ONNX_PATH, label)
    except JobFailure as e:
        print(f"\n!! {label} failed: {e}")
        result = {"label": label, "error": e.short()}
    result["local_recall"] = local_recall
    out = os.path.join(PROFILE_LOG_DIR, f"on_device_{args.precision}_{stamp}.txt")
    write_results_table([result], out)
