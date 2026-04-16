"""
On-device benchmark sweep: Compile → Profile → Inference for all ONNX models.

Auto-discovers all FP32 and INT8 ONNX pairs in exported_onnx/, then for each:
  1. Compile to QNN DLC on QAI Hub (waits for completion)
  2. Profile on XR2 Gen 2 (waits for completion)
  3. Run inference (waits for completion)
  4. Compute Recall@10

Prints a summary table and writes detailed profile data to a log file.

Usage:
  python sweep_on_device.py                              # all discovered models
  python sweep_on_device.py --model ViT-B/16            # one model only
  python sweep_on_device.py --fp32-only                 # skip INT8 models
  python sweep_on_device.py --int8-only                 # skip FP32 models
  python sweep_on_device.py --model ViT-B/16 --fp32-only   # quick smoke test
  python sweep_on_device.py --image-dataset-id dXXX --text-dataset-id dXXX
"""

import sys
import os
import re
import json
import argparse
import onnx
import qai_hub
import numpy as np
from datetime import datetime

from src.common.eval import evaluate_track1
from src.common.config import (
    ONNX_DIR, PROFILE_LOG_DIR as LOG_DIR, TXT_LIST, IMG_LIST, DEVICE_NAME,
    DEFAULT_IMAGE_DATASET_ID, DEFAULT_TEXT_DATASET_ID, SLUG_TO_MODEL,
    CALIBRATION_SOURCE, CALIBRATION_N_SAMPLES,
    ensure_output_dirs,
)

ensure_output_dirs()

# ---------------------------------------------------------------------------
# Configuration (paths and defaults from src/common/config.py)
# ---------------------------------------------------------------------------
TARGET_DEVICE = qai_hub.Device(DEVICE_NAME)

# ---------------------------------------------------------------------------
# Helpers (self-contained — not imported from run_on_device.py)
# ---------------------------------------------------------------------------

def clean_value_info(model):
    """Remove tensors from value_info that also appear in model I/O (ONNX spec violation).

    quant_pre_process can leave output tensors (e.g. 'embedding') in value_info,
    which QAI Hub's compiler rejects.
    """
    io_names = {t.name for t in model.graph.input} | {t.name for t in model.graph.output}
    culprits = [vi.name for vi in model.graph.value_info if vi.name in io_names]
    if culprits:
        new_vi = [vi for vi in model.graph.value_info if vi.name not in io_names]
        del model.graph.value_info[:]
        model.graph.value_info.extend(new_vi)
        print(f"    Cleaned value_info: removed {culprits}")
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


def compile_and_wait(model, input_specs, label, precision="fp32", calibration_data=None):
    """Submit compile job and block until complete. Returns job or raises on failure."""
    job = qai_hub.submit_compile_job(
        model=model,
        device=TARGET_DEVICE,
        input_specs=input_specs,
        options=get_compile_options(precision),
        calibration_data=calibration_data,
    )
    print(f"    [{label}] Compile job: {job.job_id}  (waiting...)")
    job.wait()
    status = job.get_status()
    if status.failure:
        raise RuntimeError(f"Compile failed: {status.message}")
    print(f"    [{label}] Compile done.")
    return job


def profile_and_wait(compiled_model, label):
    """Submit profile job and block until complete. Returns (job, latency_ms)."""
    job = qai_hub.submit_profile_job(
        model=compiled_model,
        device=TARGET_DEVICE,
        options="--max_profiler_iterations 100",
    )
    print(f"    [{label}] Profile job: {job.job_id}  (waiting...)")
    job.wait()
    status = job.get_status()
    if status.failure:
        raise RuntimeError(f"Profile failed: {status.message}")

    latency_ms = None
    mem_mb = None
    profile_data = None
    try:
        profile_data = job.download_profile()
        summary = profile_data.get("execution_summary", {})
        # QAI Hub returns estimated_inference_time in microseconds
        latency_us = summary.get("estimated_inference_time")
        if latency_us is not None:
            latency_ms = latency_us / 1000.0
        else:
            # Fallback: sum layer execution times (also in microseconds)
            layers = profile_data.get("execution_detail", {}).get("layers", [])
            if layers:
                latency_ms = sum(l.get("execution_time", 0) for l in layers) / 1000.0
        peak_bytes = summary.get("estimated_inference_peak_memory")
        if peak_bytes is not None:
            mem_mb = peak_bytes / (1024 * 1024)
    except Exception as e:
        print(f"    [{label}] Warning: could not parse profile data: {e}")

    ms_str  = f"{latency_ms:.2f} ms" if latency_ms is not None else "n/a"
    mem_str = f"{mem_mb:.0f} MiB"    if mem_mb    is not None else "n/a"
    print(f"    [{label}] Latency: {ms_str}  Peak memory: {mem_str}")
    return job, latency_ms, mem_mb, profile_data


def run_inference(compiled_model, dataset, label):
    """Submit inference job, wait, return output array or raise on failure."""
    job = qai_hub.submit_inference_job(
        model=compiled_model,
        device=TARGET_DEVICE,
        inputs=dataset,
    )
    print(f"    [{label}] Inference job: {job.job_id}  (waiting...)")
    job.wait()
    status = job.get_status()
    if status.failure:
        raise RuntimeError(f"Inference failed: {status.message}")
    return job.download_output_data()["output_0"]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_pairs(model_filter=None, fp32_only=False, int8_only=False):
    """
    Returns list of dicts:
      {model, fmt, act, img_path, txt_path}
    Discovers both FP32 and INT8 ONNX pairs in ONNX_DIR.
    """
    if not os.path.isdir(ONNX_DIR):
        print(f"Error: {ONNX_DIR}/ not found.")
        sys.exit(1)

    pairs = []

    # FP32 pairs
    if not int8_only:
        for slug, model_name in SLUG_TO_MODEL.items():
            if model_filter and model_name != model_filter:
                continue
            img = os.path.join(ONNX_DIR, f"image_encoder{slug}.onnx")
            txt = os.path.join(ONNX_DIR, f"text_encoder{slug}.onnx")
            if os.path.exists(img) and os.path.exists(txt):
                pairs.append({"model": model_name, "fmt": "fp32", "act": "—",
                               "img_path": img, "txt_path": txt})
            else:
                if os.path.exists(img) or os.path.exists(txt):
                    print(f"Warning: incomplete FP32 pair for {model_name} — skipping.")

    # INT8 pairs
    if not fp32_only:
        pattern = re.compile(
            r"^image_encoder((?:_vitl14)?)_int8_([a-z]+)_([a-z0-9]+)\.onnx$"
        )
        for fname in sorted(os.listdir(ONNX_DIR)):
            m = pattern.match(fname)
            if not m:
                continue
            slug, fmt_name, act_name = m.group(1), m.group(2), m.group(3)
            model_name = SLUG_TO_MODEL.get(slug, slug or "ViT-B/16")
            if model_filter and model_name != model_filter:
                continue
            txt_fname = f"text_encoder{slug}_int8_{fmt_name}_{act_name}.onnx"
            img_path  = os.path.join(ONNX_DIR, fname)
            txt_path  = os.path.join(ONNX_DIR, txt_fname)
            if not os.path.exists(txt_path):
                print(f"Warning: found {fname} but missing {txt_fname} — skipping.")
                continue
            pairs.append({"model": model_name, "fmt": fmt_name, "act": act_name,
                           "img_path": img_path, "txt_path": txt_path})

    return pairs


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results):
    w_model, w_fmt, w_act = 10, 12, 12
    print("\n" + "=" * 100)
    print("On-Device Sweep Results — XR2 Gen 2")
    print("=" * 100)
    print(f"  {'Model':<{w_model}} {'Format':<{w_fmt}} {'Activation':<{w_act}} "
          f"{'Img(ms)':>9} {'Txt(ms)':>9} {'Total(ms)':>10} "
          f"{'ImgMem(MiB)':>12} {'TxtMem(MiB)':>12} {'Recall@10':>10} {'vs FP32':>9}")
    print(f"  {'-'*w_model} {'-'*w_fmt} {'-'*w_act} "
          f"{'-'*9} {'-'*9} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*9}")

    # Collect FP32 baselines per model for delta calculation
    fp32_recalls = {}
    for r in results:
        if r["fmt"] == "fp32" and r.get("recall") is not None:
            fp32_recalls[r["model"]] = r["recall"]

    for r in results:
        img_ms   = f"{r['img_ms']:.1f}"    if r.get("img_ms")    is not None else "FAIL"
        txt_ms   = f"{r['txt_ms']:.1f}"    if r.get("txt_ms")    is not None else "FAIL"
        img_mem  = f"{r['img_mem_mb']:.0f}" if r.get("img_mem_mb") is not None else "n/a"
        txt_mem  = f"{r['txt_mem_mb']:.0f}" if r.get("txt_mem_mb") is not None else "n/a"
        recall   = f"{r['recall']:.4f}"    if r.get("recall")    is not None else "FAIL"

        total_ms = ""
        if r.get("img_ms") is not None and r.get("txt_ms") is not None:
            total_ms = f"{r['img_ms'] + r['txt_ms']:.1f}"

        delta_str = ""
        if r["fmt"] != "fp32" and r.get("recall") is not None:
            baseline = fp32_recalls.get(r["model"])
            if baseline is not None:
                delta_str = f"{r['recall'] - baseline:+.4f}"

        print(f"  {r['model']:<{w_model}} {r['fmt']:<{w_fmt}} {r['act']:<{w_act}} "
              f"{img_ms:>9} {txt_ms:>9} {total_ms:>10} "
              f"{img_mem:>12} {txt_mem:>12} {recall:>10} {delta_str:>9}")

    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="On-device benchmark sweep for all ONNX models")
    parser.add_argument(
        "--model", default=None, choices=["ViT-B/16", "ViT-L/14"],
        help="Filter to a single model variant (default: all discovered)",
    )
    parser.add_argument("--fp32-only",  action="store_true", help="Only run FP32 models")
    parser.add_argument("--int8-only",  action="store_true", help="Only run INT8 models")
    parser.add_argument("--image-dataset-id", default=None)
    parser.add_argument("--text-dataset-id",  default=None)
    parser.add_argument(
        "--precision", default="fp32", choices=["fp32", "fp16", "int8-compile"],
        help="Compile precision applied to FP32 pairs (int8-local pairs always compile as fp32 — they are already quantized)",
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

    image_dataset_id = args.image_dataset_id or DEFAULT_IMAGE_DATASET_ID
    text_dataset_id  = args.text_dataset_id  or DEFAULT_TEXT_DATASET_ID
    calib_source     = args.calib_source     or CALIBRATION_SOURCE
    calib_samples    = args.calib_samples    or CALIBRATION_N_SAMPLES

    # Load calibration data once before the loop (only needed for int8-compile)
    img_calib, txt_calib = None, None
    if args.precision == "int8-compile":
        from src.common.calibration import load_calibration_data
        print(f"Loading calibration data ({calib_source}, max {calib_samples} samples)...")
        img_calib = load_calibration_data("image", calib_source, calib_samples)
        txt_calib = load_calibration_data("text",  calib_source, calib_samples)
        print()

    # Log file
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"on_device_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    print(f"Profile log: {log_path}\n")

    # Discover ONNX pairs
    pairs = discover_pairs(
        model_filter=args.model,
        fp32_only=args.fp32_only,
        int8_only=args.int8_only,
    )
    if not pairs:
        print("No ONNX pairs found. Run export_onnx.py and/or quantize_local.py first.")
        sys.exit(1)

    print(f"Discovered {len(pairs)} model pair(s) to benchmark.\n")

    # Load datasets once
    image_dataset = qai_hub.get_dataset(image_dataset_id)
    text_dataset  = qai_hub.get_dataset(text_dataset_id)

    results = []

    for i, pair in enumerate(pairs):
        model_name = pair["model"]
        fmt        = pair["fmt"]
        act        = pair["act"]
        label      = f"{model_name} {fmt} {act}"

        print(f"\n[{i+1}/{len(pairs)}] {label}")
        print(f"  Image: {pair['img_path']}")
        print(f"  Text:  {pair['txt_path']}")

        row = {"model": model_name, "fmt": fmt, "act": act,
               "img_ms": None, "txt_ms": None,
               "img_mem_mb": None, "txt_mem_mb": None,
               "recall": None,
               "img_compile_job": None, "txt_compile_job": None,
               "img_profile_job": None, "txt_profile_job": None,
               "img_profile_detail": None, "txt_profile_detail": None}

        try:
            # Step 1: Load + clean ONNX
            print("  Loading ONNX models...")
            onnx_img = clean_value_info(onnx.load(pair["img_path"]))
            onnx_txt = clean_value_info(onnx.load(pair["txt_path"]))

            # Step 2: Compile
            # int8-local pairs (fmt = "qoperator"/"qdq") have quantization baked in —
            # compile as fp32 to avoid double-quantization; only apply --precision to FP32 ONNX
            effective_precision = args.precision if fmt == "fp32" else "fp32"
            pair_img_calib = img_calib if effective_precision == "int8-compile" else None
            pair_txt_calib = txt_calib if effective_precision == "int8-compile" else None
            print(f"  Compiling (precision={effective_precision})...")
            img_compile = compile_and_wait(onnx_img, {"image": (1, 3, 224, 224)}, "image", effective_precision, pair_img_calib)
            txt_compile = compile_and_wait(onnx_txt, {"text": ((1, 77), "int64")}, "text",  effective_precision, pair_txt_calib)
            row["img_compile_job"] = img_compile.job_id
            row["txt_compile_job"] = txt_compile.job_id

            img_compiled = img_compile.get_target_model()
            txt_compiled = txt_compile.get_target_model()

            # Step 3: Profile (sequential — one at a time)
            print("  Profiling...")
            img_prof_job, img_ms, img_mem_mb, img_prof_data = profile_and_wait(img_compiled, "image")
            txt_prof_job, txt_ms, txt_mem_mb, txt_prof_data = profile_and_wait(txt_compiled, "text")
            row["img_profile_job"]    = img_prof_job.job_id
            row["txt_profile_job"]    = txt_prof_job.job_id
            row["img_ms"]             = img_ms
            row["txt_ms"]             = txt_ms
            row["img_mem_mb"]         = img_mem_mb
            row["txt_mem_mb"]         = txt_mem_mb
            row["img_profile_detail"] = img_prof_data
            row["txt_profile_detail"] = txt_prof_data

            # Step 4: Inference
            print("  Running inference...")
            img_out = run_inference(img_compiled, image_dataset, "image")
            txt_out = run_inference(txt_compiled, text_dataset,  "text")

            # Step 5: Recall@10
            recall = evaluate_track1(img_out, txt_out, TXT_LIST, IMG_LIST)
            row["recall"] = float(recall)
            print(f"  Recall@10: {recall:.4f}")

        except Exception as e:
            print(f"  [SKIP] {label} — {e}")

        results.append(row)

        # Write log entry immediately after each combo
        log_entry = {
            "timestamp":        datetime.now().isoformat(),
            "model":            row["model"],
            "format":           row["fmt"],
            "activation":       row["act"],
            "img_compile_job":  row["img_compile_job"],
            "txt_compile_job":  row["txt_compile_job"],
            "img_profile_job":  row["img_profile_job"],
            "txt_profile_job":  row["txt_profile_job"],
            "img_latency_ms":    row["img_ms"],
            "txt_latency_ms":    row["txt_ms"],
            "img_peak_mem_mib":  row["img_mem_mb"],
            "txt_peak_mem_mib":  row["txt_mem_mb"],
            "recall_at_10":     row["recall"],
            "img_profile_detail": row["img_profile_detail"],
            "txt_profile_detail": row["txt_profile_detail"],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, indent=2, default=str))
            f.write("\n---\n")

    # Final summary
    print_summary_table(results)
    print(f"\nDetailed profile data saved to: {log_path}")
