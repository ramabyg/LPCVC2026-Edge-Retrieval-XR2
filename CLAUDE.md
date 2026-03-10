# LPCVC 2026 Track 1 — Image-to-Text Retrieval on Qualcomm XR2 Gen 2

## Project Goal
Maximize **Recall@10** for image-to-text retrieval running on a Qualcomm Snapdragon XR2 Gen 2 device.
Recall@10: for each image, check if its ground-truth text appears in the top-10 most similar texts by cosine similarity.

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

- **Normalization bug in `upload_dataset.py`:** images are only divided by 255 — missing CLIP's
  ImageNet normalization (mean/std). `inference_local.py` uses the correct `preprocess` pipeline
  from `clip.load()`, so local Recall@10 will be higher than on-device score. Fix `upload_dataset.py`
  before final submission.
- **`inference.py` has hardcoded job IDs:** update `compiled_id` and `dataset_id` after each
  compile/upload run.
- **`.onnx.data` files are used:** they hold the model weights (~344 MB image, ~254 MB text).
  Do not delete them — the `.onnx` file references them by relative path.

## Key Source Files

| File | Role |
|------|------|
| `clip_model/clip/clip.py` | `clip.load()`, `clip.tokenize()`, `_transform()` preprocessing |
| `clip_model/clip/model.py` | CLIP model class, `encode_image()`, `encode_text()` |
| `inference.py` | `evaluate_track1()`, `parse_ground_truth()` — reuse these functions |
| `export_onnx.py` | `ImageEncoderWrapper`, `TextEncoderWrapper` — modify for model experiments |

## Evaluation Function

```python
from inference import evaluate_track1
# img_output: list of numpy arrays shape (1, 512)
# txt_output: list of numpy arrays shape (1, 512)
result = evaluate_track1(img_output, txt_output, TXT_LIST_PATH, IMG_LIST_PATH)
# returns float: mean Recall@10
```
