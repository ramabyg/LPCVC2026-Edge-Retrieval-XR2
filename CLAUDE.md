# LPCVC 2026 Track 1 — Image-to-Text Retrieval on Qualcomm XR2 Gen 2

## Project Goal
Maximize **Recall@10** for image-to-text retrieval running on a Qualcomm Snapdragon XR2 Gen 2 device.
Recall@10: for each image, check if its ground-truth text appears in the top-10 most similar texts by cosine similarity.

## Competition Constraints

| Constraint | Value |
|-----------|-------|
| Latency budget | ≤ 35ms combined (image + text encoder) — hard threshold |
| Ranking metric | Recall@10 (higher is better, only scored if latency passes) |
| Input format | `float32 (1, 3, 224, 224)`, images pre-resized to 224×224, divided by 255 only |
| Normalization | NOT applied by competition — must be baked into the model |
| Target device | Snapdragon XR2 Gen 2 (Hexagon NPU, optimized for INT8) |

## Current Baseline (March 2026)

| Model | Image enc (ms) | Text enc (ms) | Total | Status |
|-------|---------------|--------------|-------|--------|
| ViT-B/16 FP32 (no norm baked) | 26.3 | 4.6 | ~31ms | Old baseline |
| ViT-B/16 FP32 (norm baked) | ~26.3 | 4.6 | ~31ms | **Current — norm fix applied** |

Already under 35ms with ViT-B/16 FP32 → primary goal is **maximizing Recall@10**, not shrinking the model.

## Pipeline (5 steps)

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `export_onnx.py` | Exports CLIP image + text encoders to ONNX format |
| 2 | `compile_and_profile.py` | Uploads ONNX to QAI Hub, compiles to QNN DLC for XR2 Gen 2, profiles latency |
| 3 | `upload_dataset.py` | Uploads images + tokenized text to QAI Hub (prints dataset IDs) |
| 4 | `inference.py` | Runs compiled models on QAI Hub device, computes Recall@10 |
| 5 | `inference_local.py` | Runs CLIP locally (no QAI Hub) for fast iteration — use this for experiments |

**Typical workflow for experiments:** edit model → run `inference_local.py` to validate → if good, run steps 1–4 to push to device.

## Architecture Decisions

- **Model:** `ViT-B/16` CLIP loaded from local `clip_model/` submodule (OpenAI's original repo)
  - Do NOT use `qai_hub_models.models.openai_clip` wrapper — we load directly via `clip.load("ViT-B/16")`
- **Normalization:** CLIP mean/std baked into `ImageEncoderWrapper.forward()` using `register_buffer`
  - mean: `[0.48145466, 0.4578275, 0.40821073]`, std: `[0.26862954, 0.26130258, 0.27577711]`
  - Competition sends `/255` images → wrapper applies CLIP normalization internally
- **ONNX export:** opset 18, fixed batch size 1, `dynamo=True`, float32 image, int64 text
- **Compile options:** `--target_runtime qnn_dlc --truncate_64bit_io` (int64 → int32 on device)
- **ONNX files:** `exported_onnx/image_encoder.onnx` + `.onnx.data`, `text_encoder.onnx` + `.onnx.data`
  - Both `.onnx` and `.onnx.data` must always stay together in the same directory

## Important File Paths

- Local dataset: `C:\rama\projects\data\lpcvc_track1_sample_data\`
  - `images/` — 57 images
  - `img_list.csv` — columns: image filename | semicolon-separated ground-truth text IDs
  - `txt_list.csv` — columns: text ID | text prompt (222 entries)
- CLIP weights cache: `~/.cache/clip/` (downloaded automatically on first run)
- ONNX exports: `exported_onnx/`

## Known Issues / Gotchas

- **`upload_dataset.py` normalization bug:** images are only divided by 255 — missing CLIP's
  mean/std normalization. Now that normalization is baked into `ImageEncoderWrapper`, this
  is correct behavior for the model — but `upload_dataset.py` must NOT add normalization
  (the model handles it). Fix needed: ensure preprocessing matches (just `/255`).
- **`inference.py` has hardcoded job IDs:** update `compiled_id` and `dataset_id` after each
  compile/upload run.
- **`.onnx.data` files are used:** they hold the model weights (~344 MB image, ~254 MB text).
  Do not delete them — the `.onnx` file references them by relative path.
- **After normalization fix:** re-export ONNX and re-compile before running on-device.
  The old compiled DLC does not have normalization baked in.

## Key Source Files

| File | Role |
|------|------|
| `clip_model/clip/clip.py` | `clip.load()`, `clip.tokenize()`, `_transform()` preprocessing |
| `clip_model/clip/model.py` | CLIP model class, `encode_image()`, `encode_text()` |
| `inference.py` | `evaluate_track1()`, `parse_ground_truth()` — reuse these functions |
| `export_onnx.py` | `ImageEncoderWrapper` (norm baked in), `TextEncoderWrapper` — modify for experiments |
| `inference_local.py` | Uses competition-style `/255` input + manual CLIP norm — matches on-device behavior |

## Evaluation Function

```python
from inference import evaluate_track1
# img_output: list of numpy arrays shape (1, 512)
# txt_output: list of numpy arrays shape (1, 512)
result = evaluate_track1(img_output, txt_output, TXT_LIST_PATH, IMG_LIST_PATH)
# returns float: mean Recall@10
```

## Optimization Strategy (Phase 0 complete — next steps)

1. **Run `inference_local.py`** → get true FP32 baseline Recall@10 (norm now correctly applied)
2. **Re-export ONNX + compile + run on-device** → verify local ≈ on-device Recall@10
3. **INT8 quantize ViT-B/16** on QAI Hub → profile latency, check Recall@10
4. **Profile ViT-L/14** (FP32 + INT8) → if fits under 35ms, switch to larger model
5. **Fine-tune on COCO + Flickr30k** → LoRA first (works on GTX 1650), full fine-tune with better GPU
6. **Knowledge distillation** from ViT-L/14 → ViT-B/16 if ViT-L/14 too slow on-device

Full plan: `CLIP_Optimization_Plan_v2.md`
