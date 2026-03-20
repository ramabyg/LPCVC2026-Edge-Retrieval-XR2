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

- Already under 35ms with ViT-B/16 FP32 → primary goal is **maximizing Recall@10**, not shrinking the model.
- Recall@10 for sample image data set is 0.8805

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


## Measured Baselines

| Variant | Recall@10 | Notes |
|---------|-----------|-------|
| PyTorch FP32 (local) | **0.8805** | Ground truth — `inference_local.py` |
| FP32 ONNX (local ORT) | **0.8728** | Small gap due to PIL box resize vs bicubic+centercrop |
| INT8 ONNX (first attempt) | **0.0527** | Catastrophic failure — ~random chance |

---

## Original Quantization Settings (that produced 0.0527)

Both image and text encoders quantized with `quantize_local.py` using:

| Setting | Value |
|---------|-------|
| **Format** | `QuantFormat.QDQ` |
| **Weight type** | `QuantType.QInt8` |
| **Activation type** | `QuantType.QInt8` |
| **Calibration method** | `CalibrationMethod.MinMax` |
| **Per-channel** | `True` |
| **Pre-processing** | `quant_pre_process()` (shape inference + generic graph optimization) |
| **Image encoder scope** | All ops (Conv, MatMul, Gemm) |
| **Text encoder scope** | MatMul/Gemm only (Gather/embeddings excluded) |
| **Calibration data** | 57 images, 222 text prompts (sample dataset) |

---

## Root Cause Identified

`QuantFormat.QDQ` is designed for **hardware compiler export** (QAI Hub, TensorRT). When run under **local ORT inference**:
- The model's last output node may remain in int8 (no trailing `DequantizeLinear`)
- ORT returns raw `±127`-range int8 values interpreted as float32
- Cosine similarity computed on garbage values
- Result: ~random Recall@10

---

## Code Changes Implemented (Ready to Test)

### `quantize_local.py`
Added CLI flags to make quantization settings configurable without code edits:

```bash
# Defaults changed to fix the QDQ bug:
--format qoperator         # Default: was QDQ → now QOperator (always outputs float32)
--activation quint8        # Default: was QInt8 → now QUInt8 (broader ORT kernel support)
--calibration percentile   # Default: was MinMax → now Percentile (handles attention outliers)
```

**Example usage:**
```bash
python quantize_local.py                           # Use new defaults (QOperator + QUInt8 + Percentile)
python quantize_local.py --format qdq              # Keep QDQ for QAI Hub export if needed
python quantize_local.py --activation qint8        # Use signed int8 if required
python quantize_local.py --calibration minmax      # Use MinMax calibration if needed
```

### `inference_onnx_local.py`
Added diagnostic flags for debugging:

```bash
--mode all|fp32|int8|fp32_img_int8_txt|int8_img_fp32_txt
# Cross-combination testing to isolate which encoder is broken
# fp32_img_int8_txt: FP32 image encoder + INT8 text encoder
# int8_img_fp32_txt: INT8 image encoder + FP32 text encoder

--inspect-embeddings
# Prints for first sample from each encoder:
#   - dtype, shape, min, max, norm
#   - cosine_sim(FP32_embed, INT8_embed)
# KEY DEBUG SIGNAL: if min/max ≈ ±127 → raw int8 leak (confirms QDQ bug)
```

**Example usage:**
```bash
python inference_onnx_local.py --inspect-embeddings
python inference_onnx_local.py --mode fp32_img_int8_txt
python inference_onnx_local.py --mode int8_img_fp32_txt
```

---

## Current State of Files

| File | Status | Last Produced |
|------|--------|---------------|
| `quantize_local.py` | ✅ Modified | n/a |
| `inference_onnx_local.py` | ✅ Modified | n/a |
| `exported_onnx/image_encoder_int8.onnx` | ⚠️ Stale | Old QDQ settings — needs re-run |
| `exported_onnx/text_encoder_int8.onnx` | ⚠️ Stale | Old QDQ settings — needs re-run |

---

## Next Steps (Not Yet Run)

### Phase 1: Run with new QOperator defaults

```bash
# Re-quantize both encoders with QOperator (not QDQ)
python quantize_local.py
```

**Expected:** Output float32 embeddings (not ±127 int8 garbage)

### Phase 2: Verify output type and embedding stats

```bash
# Check embedding dtype and range for first sample
python inference_onnx_local.py --inspect-embeddings
```

**Expected:** `dtype=float32`, `min/max` in reasonable range (not ±127)

### Phase 3: Isolate which encoder is broken (if INT8 still poor)

```bash
# Test with mixed encoders to find culprit
python inference_onnx_local.py --mode fp32_img_int8_txt
python inference_onnx_local.py --mode int8_img_fp32_txt
```

**Expected:** One will be ~0.87 (good), one will be ~0.05 (broken)

### Phase 4: Transformer graph optimization (planned, not yet implemented)

If QOperator + QUInt8 + Percentile still doesn't reach 0.85+ Recall@10:
- Add `onnxruntime.transformers.optimize_model` before `quant_pre_process`
- Fuses attention, LayerNorm, GELU subgraphs → cleaner int8 boundaries
- Image encoder: `model_type='vit'`, num_heads=12, hidden_size=768
- Text encoder: `model_type='bert'`, num_heads=12, hidden_size=768
- See plan file: `C:\Users\ursra\.claude\plans\luminous-zooming-cookie.md`

### Phase 5: Layer-by-layer error analysis (if still needed)

Use `onnxruntime.quantization.qdq_loss_debug` to identify worst-performing layers and exclude them from quantization.

---

## ViT-B/16 Model Parameters (for reference)

| Parameter | Value |
|-----------|-------|
| Image encoder type | ViT (Vision Transformer) |
| Text encoder type | BERT-style transformer |
| Hidden dimension | 768 |
| Num attention heads | 12 |
| Projection dimension | 512 (embedding output) |
| Image input | (1, 3, 224, 224) float32 |
| Text input | (1, 77) int64 tokens |

---

## Key Learnings

1. **QDQ vs QOperator:** QDQ is for hardware export; QOperator is for local ORT inference
2. **Output type matters:** If the last output node isn't explicitly dequantized, ORT returns raw int8
3. **Transformer activation distributions:** Attention softmax produces sharp, outlier-prone distributions that MinMax calibration can't handle well → Percentile is better
4. **ORT kernel support:** Unsigned int8 (QUInt8) has broader CPU kernel support than signed int8 (QInt8)
5. **Graph optimization:** `dynamo=True` ONNX exports may use non-standard patterns that transformer-specific optimizers don't recognize — need graceful fallback

## Related Documentation

- **Quantization debug plan:** `quantization_debug_pllan.md`
- **Transformer optimization plan:** `working_with_claude/transformer_graph_optimization_issue_debug.md`
- **Detailed optimization implementation plan:** `C:\Users\ursra\.claude\plans\luminous-zooming-cookie.md`
- **Main project spec:** `CLAUDE.md`

## Measured Baselines (Updated)

| Variant | Recall@10 | Notes |
|---------|-----------|-------|
| PyTorch FP32 (local) | **0.8805** | Ground truth — `inference_local.py` |
| FP32 ONNX (local ORT) | **0.8728** | Small gap from PIL box resize vs bicubic+centercrop |
| INT8 ONNX — attempt 1 (QDQ + all ops) | **0.0527** | Catastrophic — raw int8 output leak |
| INT8 ONNX — attempt 2 (QOperator, all ops) | **0.1003** | Still catastrophic — Softmax/LayerNorm poisoned |
| **INT8 ONNX — fixed (QOperator, Conv/MatMul/Gemm only)** | **0.8256** | ✅ Production-ready |

## Embedding Stats (First Sample)

| Encoder | dtype | norm | min | max |
|---------|-------|------|-----|-----|
| FP32 image | float32 | 10.655 | -2.100 | 7.341 |
| INT8 image (fixed) | float32 | 9.756 | -1.003 | 6.564 |
| FP32 text | float32 | 8.953 | -1.409 | 6.441 |
| INT8 text | float32 | 10.048 | -2.223 | 7.759 |

Cosine similarity between img[0] and txt[0]:
- FP32/FP32: 0.2565
- INT8_img/FP32_txt: **0.2671** (slightly higher than FP32 — mild beneficial regularization)
- FP32_img/INT8_txt: 0.2453

---

## Final Recall@10 Results

| Config | Recall@10 | vs FP32 baseline |
|--------|-----------|-----------------|
| FP32 (ONNX baseline) | 0.8728 | — |
| **INT8 (both encoders)** | **0.8256** | **-0.0472** |
| FP32_img + INT8_txt | 0.8524 | -0.0204 |
| INT8_img + FP32_txt | 0.8619 | -0.0109 |

**Interesting:** INT8_img + FP32_txt (0.8619) > FP32_img + INT8_txt (0.8524).
The INT8 image encoder is performing better in isolation than INT8 text encoder.

---
