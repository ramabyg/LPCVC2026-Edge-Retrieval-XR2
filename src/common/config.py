"""
Central configuration for LPCVC 2026 Track 1.

All data paths, QAI Hub settings, output directories, and CLIP constants
live here. Import from any script instead of hardcoding values inline.

Paths support environment variable overrides — no code edits needed
when running on a different machine:

    set LPCVC_DATA_DIR=D:\\datasets\\lpcvc_sample
    set LPCVC_COCO_DIR=D:\\datasets\\coco
    set QAI_HUB_IMAGE_DATASET=d_new_upload_id
    python src/local/inference_pytorch.py

On Linux/Mac use export instead of set:
    export LPCVC_DATA_DIR=/data/lpcvc_sample
"""

import os

# ── Sample dataset (competition data) ────────────────────────────────────────
# SAMPLE_DATA_DIR = os.environ.get(
#     "LPCVC_DATA_DIR",
#     r"/mnt/sandbox/ryara/datasets/lpcvc_track1_sample_data",
# )
SAMPLE_DATA_DIR = os.environ.get(
    "LPCVC_DATA_DIR",
    r"C:\rama\projects\data\lpcvc_track1_sample_data",
)
IMAGE_DIR = os.path.join(SAMPLE_DATA_DIR, "images")
IMG_LIST  = os.path.join(SAMPLE_DATA_DIR, "img_list.csv")
TXT_LIST  = os.path.join(SAMPLE_DATA_DIR, "txt_list.csv")

# ── Training datasets ─────────────────────────────────────────────────────────
# COCO_DATA_DIR    = os.environ.get("LPCVC_COCO_DIR",     r"/mnt/sandbox/ryara/datasets/coco")
# FLICKR30K_DIR    = os.environ.get("LPCVC_FLICKR30K_DIR", r"/mnt/sandbox/ryara/datasets/flickr30k")
COCO_DATA_DIR    = os.environ.get("LPCVC_COCO_DIR",     r"C:\rama\projects\data\coco")
FLICKR30K_DIR    = os.environ.get("LPCVC_FLICKR30K_DIR", r"C:\rama\projects\data\flickr30k")

COCO_JSON         = os.path.join(COCO_DATA_DIR, "dataset_coco.json")
FLICKR30K_JSON    = os.path.join(FLICKR30K_DIR, "dataset_flickr30k.json")
FLICKR30K_IMG_DIR = os.path.join(FLICKR30K_DIR, "images")

# ── Output directories ────────────────────────────────────────────────────────
# Default to paths relative to this file's repo root so the code works on any
# machine (Windows, Linux GPU server) without code edits.  Override via env
# vars when a different storage location is needed (e.g. /mnt/artifacts/...).
# to override use something like: 
# export LPCVC_ONNX_DIR=/mnt/artifacts/ryara/lpcvc/exported_onnx 
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ONNX_DIR        = os.environ.get("LPCVC_ONNX_DIR",         os.path.join(_REPO_ROOT, "exported_onnx"))
PROFILE_LOG_DIR = os.environ.get("LPCVC_PROFILE_LOG_DIR",  os.path.join(_REPO_ROOT, "profile_logs"))
DIAGNOSTICS_DIR = os.environ.get("LPCVC_DIAGNOSTICS_DIR",  os.path.join(_REPO_ROOT, "diagnostics"))
CHECKPOINTS_DIR = os.environ.get("LPCVC_CHECKPOINTS_DIR",  os.path.join(_REPO_ROOT, "lora_checkpoints"))

# ── QAI Hub ───────────────────────────────────────────────────────────────────
DEVICE_NAME              = os.environ.get("QAI_HUB_DEVICE",         "XR2 Gen 2 (Proxy)")
DEFAULT_IMAGE_DATASET_ID = os.environ.get("QAI_HUB_IMAGE_DATASET",  "d91y6v3n9")
DEFAULT_TEXT_DATASET_ID  = os.environ.get("QAI_HUB_TEXT_DATASET",   "d74j38p07")

# ── Calibration data (for compile-time INT8 quantization) ────────────────────
# Override via env vars or CLI --calib-source / --calib-samples
CALIBRATION_SOURCE    = os.environ.get("LPCVC_CALIB_SOURCE",    "coco")  # sample | coco | flickr30k
CALIBRATION_N_SAMPLES = int(os.environ.get("LPCVC_CALIB_N_SAMPLES", "500"))

# ── CLIP model ────────────────────────────────────────────────────────────────
CLIP_MEAN     = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)
SLUG_TO_MODEL = {"": "ViT-B/16", "_vitl14": "ViT-L/14"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def ensure_output_dirs():
    """Create all standard output directories if they don't exist.

    Call once at the top of any script that reads/writes output folders.
    Safe to call multiple times (exist_ok=True).
    """
    for d in [ONNX_DIR, PROFILE_LOG_DIR, DIAGNOSTICS_DIR, CHECKPOINTS_DIR]:
        os.makedirs(d, exist_ok=True)
