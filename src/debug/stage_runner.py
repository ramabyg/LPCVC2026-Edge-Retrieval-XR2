"""
Stage runner — local Recall@10 across the QAI Hub quantize pipeline.

For a chosen --precision (int8-hub or w8a16), this script collects three
artifacts and evaluates each one locally on the sample dataset:

  Stage 0 — raw exported ONNX from `src/platform/export_onnx.py`
  Stage 1 — Hub-optimized ONNX  (output of submit_compile_job --target_runtime onnx)
  Stage 2 — Hub-quantized  ONNX (output of submit_quantize_job)

Stage 3 (compiled QNN DLC running on-device) is intentionally out of scope —
that path is owned by `src/platform/run_on_device.py`. The point of this tool
is to localize accuracy regressions inside the locally-runnable portion of
the pipeline, before spending any device cycles.

Outputs:
  diagnostics/stage_artifacts/{precision}/stage{1,2}/{image,text}.onnx
  diagnostics/stage_artifacts/{precision}/job_ids.json   # for --reuse-jobs

Examples:
  # First run — submits 2 optimize jobs + 2 quantize jobs, downloads, evaluates.
  python -m src.debug.stage_runner --precision int8-hub

  # Reuse jobs from a previous run (no Hub credits used).
  python -m src.debug.stage_runner --precision int8-hub --reuse-jobs

  # Reuse jobs explicitly:
  python -m src.debug.stage_runner --precision int8-hub \
      --img-optimize-job jXXX --txt-optimize-job jXXX \
      --img-quantize-job jXXX --txt-quantize-job jXXX

  # Skip Hub work entirely — just re-evaluate already-downloaded artifacts.
  python -m src.debug.stage_runner --precision int8-hub --eval-only
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort
import qai_hub

from src.common.calibration import load_calibration_data
from src.common.config import (
    DIAGNOSTICS_DIR, ONNX_DIR, TXT_LIST, IMG_LIST,
    CALIBRATION_SOURCE, CALIBRATION_N_SAMPLES, DEVICE_NAME,
    ensure_output_dirs,
)
from src.common.eval import evaluate_track1
from src.local.inference_onnx import (
    load_images, load_text_tokens, print_embedding_stats,
)


PRECISION_DTYPES = {
    "int8-hub": (qai_hub.QuantizeDtype.INT8, qai_hub.QuantizeDtype.INT8),
    "w8a16":    (qai_hub.QuantizeDtype.INT8, qai_hub.QuantizeDtype.INT16),
}

ENCODERS = [
    # (name, raw_onnx_basename, input_spec_for_compile_job)
    ("image", "image_encoder.onnx", {"image": (1, 3, 224, 224)}),
    ("text",  "text_encoder.onnx",  {"text":  ((1, 77), "int64")}),
]


def stage_dir(precision: str, stage: int) -> str:
    return os.path.join(DIAGNOSTICS_DIR, "stage_artifacts", precision, f"stage{stage}")


def job_ids_path(precision: str) -> str:
    return os.path.join(DIAGNOSTICS_DIR, "stage_artifacts", precision, "job_ids.json")


def load_job_ids(precision: str) -> dict:
    p = job_ids_path(precision)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_job_ids(precision: str, ids: dict) -> None:
    p = job_ids_path(precision)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(ids, f, indent=2)


def download_workbench_onnx(model: qai_hub.Model, dst_folder: str, name: str) -> str:
    """Download a Hub ONNX model, unzipping if needed, into dst_folder/{name}.onnx (+ {name}.onnx.data).

    The external-data companion file is renamed to {name}.onnx.data and the ONNX
    proto is patched to reference the new name.  This prevents silent overwrites
    when image and text encoders are downloaded to the same dst_folder — Hub often
    uses a generic internal name (e.g. model.onnx.data) for both, and the second
    download would clobber the first without the rename.

    Returns the path to the .onnx file inside dst_folder.
    """
    os.makedirs(dst_folder, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        downloaded = model.download(tmpdir)
        if downloaded.endswith(".zip"):
            shutil.unpack_archive(downloaded, tmpdir)
            os.remove(downloaded)
            entries = [e for e in os.listdir(tmpdir) if not e.endswith(".zip")]
            if len(entries) == 1 and os.path.isdir(os.path.join(tmpdir, entries[0])):
                bundle = os.path.join(tmpdir, entries[0])
            else:
                bundle = tmpdir
        else:
            bundle = os.path.dirname(downloaded) if os.path.isfile(downloaded) else downloaded

        onnx_files = [f for f in os.listdir(bundle) if f.endswith(".onnx")]
        if not onnx_files:
            raise RuntimeError(f"No .onnx file found in downloaded bundle {bundle!r}")
        onnx_src = os.path.join(bundle, onnx_files[0])
        data_files = [f for f in os.listdir(bundle) if not f.endswith(".onnx")]

        dst_onnx = os.path.join(dst_folder, f"{name}.onnx")
        new_location = f"{name}.onnx.data"

        if data_files:
            # Patch the proto's external-data location to the canonical {name}.onnx.data
            # filename before saving, so ORT can resolve it after both encoders land in
            # the same folder.
            model_proto = onnx.load(onnx_src, load_external_data=False)
            for init in model_proto.graph.initializer:
                if init.data_location == onnx.TensorProto.EXTERNAL:
                    for kv in init.external_data:
                        if kv.key == "location":
                            kv.value = new_location
            with open(dst_onnx, "wb") as f:
                f.write(model_proto.SerializeToString())
            shutil.copy(os.path.join(bundle, data_files[0]),
                        os.path.join(dst_folder, new_location))
        else:
            shutil.copy(onnx_src, dst_onnx)

        return dst_onnx


def submit_optimize_job(name: str, raw_onnx_path: str, input_specs: dict,
                        max_retries: int = 3) -> qai_hub.CompileJob:
    """Stage 1 — Hub's --target_runtime onnx pass (LN/Attention/GELU fusion etc.)."""
    print(f"  [{name}] submitting optimize compile job (--target_runtime onnx)...")
    onnx_basename = os.path.basename(raw_onnx_path)
    data_file = raw_onnx_path + ".data"
    # Hub requires a directory when the ONNX has external data. Both encoders share
    # ONNX_DIR, so create a per-encoder temp dir to avoid cross-contamination.
    for attempt in range(1, max_retries + 1):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                shutil.copy(raw_onnx_path, os.path.join(tmpdir, onnx_basename))
                if os.path.exists(data_file):
                    shutil.copy(data_file, os.path.join(tmpdir, onnx_basename + ".data"))
                job = qai_hub.submit_compile_job(
                    model=tmpdir,
                    device=qai_hub.Device(DEVICE_NAME),
                    input_specs=input_specs,
                    options="--target_runtime onnx",
                    name=f"stage1-{name}-optimize",
                )
            print(f"  [{name}] optimize job: {job.job_id}")
            return job
        except qai_hub.client.InternalError as e:
            if attempt == max_retries:
                raise
            wait = 15 * attempt
            print(f"  [{name}] upload failed (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)


def submit_quantize(
    name: str, optimized_model: qai_hub.Model, calib: dict, precision: str,
) -> qai_hub.QuantizeJob:
    weights_dtype, activations_dtype = PRECISION_DTYPES[precision]
    options = ""  # Tool 1 uses defaults so we measure the stock pipeline.
    print(
        f"  [{name}] submitting quantize job  weights={weights_dtype.name} "
        f"acts={activations_dtype.name}  options={options!r}..."
    )
    job = qai_hub.submit_quantize_job(
        model=optimized_model,
        calibration_data=calib,
        weights_dtype=weights_dtype,
        activations_dtype=activations_dtype,
        options=options,
        name=f"stage2-{name}-quantize-{precision}",
    )
    print(f"  [{name}] quantize job: {job.job_id}")
    return job


def wait_for(job, what: str):
    job.wait()
    status = job.get_status()
    if status.failure:
        print(f"  FAILED ({what}): {status.message}")
        sys.exit(1)


def evaluate_pair(label: str, image_onnx: str, text_onnx: str,
                  images, text_tokens, inspect_first: bool) -> float:
    """Run image+text encoders locally with onnxruntime CPU and return Recall@10."""
    print(f"\n[{label}]  image: {os.path.relpath(image_onnx)}")
    print(f"[{label}]   text: {os.path.relpath(text_onnx)}")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    img_sess = ort.InferenceSession(image_onnx, sess_options=sess_opts,
                                    providers=["CPUExecutionProvider"])
    txt_sess = ort.InferenceSession(text_onnx, sess_options=sess_opts,
                                    providers=["CPUExecutionProvider"])
    img_in = img_sess.get_inputs()[0].name
    txt_in = txt_sess.get_inputs()[0].name

    img_out = []
    for i, arr in enumerate(images):
        out = img_sess.run(None, {img_in: arr})[0]
        img_out.append(out)
        if inspect_first and i == 0:
            print_embedding_stats(out, f"[{label}] image embed[0]")

    txt_out = []
    for i, arr in enumerate(text_tokens):
        out = txt_sess.run(None, {txt_in: arr})[0]
        txt_out.append(out)
        if inspect_first and i == 0:
            print_embedding_stats(out, f"[{label}] text  embed[0]")

    recall = evaluate_track1(img_out, txt_out, TXT_LIST, IMG_LIST)
    print(f"[{label}] Recall@10 = {recall:.4f}")
    return recall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--precision", choices=list(PRECISION_DTYPES), default="int8-hub")
    parser.add_argument("--reuse-jobs", action="store_true",
                        help="Use job IDs from job_ids.json (skip Hub submissions for already-known jobs).")
    parser.add_argument("--img-optimize-job", default=None)
    parser.add_argument("--txt-optimize-job", default=None)
    parser.add_argument("--img-quantize-job", default=None)
    parser.add_argument("--txt-quantize-job", default=None)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip Hub work — only re-evaluate already-downloaded artifacts.")
    parser.add_argument("--calib-source", default=CALIBRATION_SOURCE,
                        choices=["sample", "coco", "flickr30k"])
    parser.add_argument("--calib-samples", type=int, default=CALIBRATION_N_SAMPLES)
    parser.add_argument("--no-inspect", action="store_true",
                        help="Skip first-sample embedding stat printout.")
    args = parser.parse_args()

    ensure_output_dirs()
    inspect_first = not args.no_inspect

    print(f"Stage runner — precision={args.precision}")
    print(f"  Stage artifacts dir: {os.path.join(DIAGNOSTICS_DIR, 'stage_artifacts', args.precision)}")

    raw_paths = {name: os.path.join(ONNX_DIR, raw) for name, raw, _ in ENCODERS}
    for name, p in raw_paths.items():
        if not os.path.exists(p):
            print(f"\nError: raw {name} ONNX not found at {p}")
            print("Run: python src/platform/export_onnx.py")
            sys.exit(1)

    # ---- Resolve job IDs from CLI or cache (used unless --eval-only) ----
    cached = load_job_ids(args.precision)
    job_ids = {
        "image_optimize": args.img_optimize_job or (cached.get("image_optimize") if args.reuse_jobs else None),
        "text_optimize":  args.txt_optimize_job or (cached.get("text_optimize")  if args.reuse_jobs else None),
        "image_quantize": args.img_quantize_job or (cached.get("image_quantize") if args.reuse_jobs else None),
        "text_quantize":  args.txt_quantize_job or (cached.get("text_quantize")  if args.reuse_jobs else None),
    }

    # ---- Stage 1 + Stage 2: submit/reuse jobs and download artifacts ----
    if not args.eval_only:
        print("\nLoading calibration data...")
        img_calib = load_calibration_data("image", args.calib_source, args.calib_samples)
        txt_calib = load_calibration_data("text",  args.calib_source, args.calib_samples)
        calib_by_encoder = {"image": img_calib, "text": txt_calib}

        print("\n=== Stage 1: Hub optimize compile jobs (--target_runtime onnx) ===")
        optimize_jobs = {}
        for name, raw_basename, specs in ENCODERS:
            jid = job_ids[f"{name}_optimize"]
            if jid:
                print(f"  [{name}] reusing optimize job {jid}")
                job = qai_hub.get_job(jid)
            else:
                job = submit_optimize_job(name, raw_paths[name], specs)
                job_ids[f"{name}_optimize"] = job.job_id
                save_job_ids(args.precision, job_ids)
            optimize_jobs[name] = job

        for name, job in optimize_jobs.items():
            wait_for(job, f"stage1-{name}")
        print("  Stage 1 jobs complete.")

        print("\n=== Stage 2: Hub quantize jobs ===")
        quantize_jobs = {}
        for name, _raw, _specs in ENCODERS:
            jid = job_ids[f"{name}_quantize"]
            if jid:
                print(f"  [{name}] reusing quantize job {jid}")
                job = qai_hub.get_job(jid)
            else:
                job = submit_quantize(name, optimize_jobs[name].get_target_model(),
                                      calib_by_encoder[name], args.precision)
                job_ids[f"{name}_quantize"] = job.job_id
                save_job_ids(args.precision, job_ids)
            quantize_jobs[name] = job

        for name, job in quantize_jobs.items():
            wait_for(job, f"stage2-{name}")
        print("  Stage 2 jobs complete.")

        # Download artifacts.
        print("\n=== Downloading Stage 1 + Stage 2 artifacts ===")
        for name, _raw, _specs in ENCODERS:
            s1 = stage_dir(args.precision, 1)
            s2 = stage_dir(args.precision, 2)
            os.makedirs(s1, exist_ok=True)
            os.makedirs(s2, exist_ok=True)
            print(f"  [{name}] downloading optimized ONNX to {s1} ...")
            download_workbench_onnx(optimize_jobs[name].get_target_model(), s1, name)
            print(f"  [{name}] downloading quantized ONNX to {s2} ...")
            download_workbench_onnx(quantize_jobs[name].get_target_model(), s2, name)

    # ---- Local evaluation ----
    print("\n=== Loading sample dataset ===")
    images = load_images()
    text_tokens = load_text_tokens()
    print(f"  {len(images)} images, {len(text_tokens)} text prompts.")

    s1 = stage_dir(args.precision, 1)
    s2 = stage_dir(args.precision, 2)
    plan = [
        ("Stage 0  raw fp32",     raw_paths["image"],            raw_paths["text"]),
        ("Stage 1  hub-optimized", os.path.join(s1, "image.onnx"), os.path.join(s1, "text.onnx")),
        ("Stage 2  hub-quantized", os.path.join(s2, "image.onnx"), os.path.join(s2, "text.onnx")),
    ]

    print("\n=== Local evaluation (onnxruntime CPU) ===")
    results = []
    for label, img_path, txt_path in plan:
        if not (os.path.exists(img_path) and os.path.exists(txt_path)):
            print(f"\n[{label}] SKIP — artifacts missing ({img_path}, {txt_path}).")
            results.append((label, None))
            continue
        recall = evaluate_pair(label, img_path, txt_path, images, text_tokens, inspect_first)
        results.append((label, recall))

    # ---- Summary ----
    baseline = next((r for label, r in results if label.startswith("Stage 0") and r is not None), None)
    print("\n" + "=" * 64)
    print(f"Summary — precision={args.precision}")
    print("=" * 64)
    print(f"  {'Stage':<26} {'Recall@10':>10}  {'Δ vs Stage 0':>14}")
    print(f"  {'-'*26} {'-'*10}  {'-'*14}")
    for label, recall in results:
        if recall is None:
            print(f"  {label:<26} {'—':>10}  {'—':>14}")
            continue
        delta = "" if baseline is None or label.startswith("Stage 0") else f"{recall - baseline:+.4f}"
        print(f"  {label:<26} {recall:>10.4f}  {delta:>14}")
    print("=" * 64)
    print(f"\nJob IDs (saved to {job_ids_path(args.precision)}):")
    for k, v in job_ids.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
