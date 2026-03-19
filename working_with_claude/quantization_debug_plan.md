# Quantization Debugging Plan — CLIP ViT-B/16 INT8

## Problem Statement

Local ONNXRuntime INT8 static quantization produces catastrophic accuracy loss:

| Variant | Recall@10 |
|---------|-----------|
| FP32 ONNX | 0.8728 |
| INT8 ONNX | 0.0527 |
| Delta | **-0.8201** |

**0.0527 ≈ 10/222 ≈ random chance** → embeddings are completely garbage, not just degraded.

## Quantization Settings That Produced This Result

- Format: `QuantFormat.QDQ`
- Weight type: `QInt8`, Activation type: `QInt8`
- Calibration: `MinMax`, per-channel=True
- Image encoder: all ops quantized
- Text encoder: MatMul/Gemm only (conservative)
- Preprocessing: `quant_pre_process()` applied before quantization

## Debugging Steps

### Step 1 — Isolate Which Encoder Is Broken

**Goal:** Determine if the problem is in the image encoder, text encoder, or both.

**Method:** Run cross-combinations in `inference_onnx_local.py`:

| Run | Image Model | Text Model | What it tells you |
|-----|-------------|------------|-------------------|
| A | FP32 | FP32 | Baseline (0.8728) |
| B | FP32 | INT8 | Is text encoder broken? |
| C | INT8 | FP32 | Is image encoder broken? |
| D | INT8 | INT8 | Current result (0.0527) |

**Expected outcome:** One or both of B/C will show ~random recall, identifying the culprit.

**Status:** [ ] Not started

---

### Step 2 — Inspect Raw Embedding Output

**Goal:** Confirm whether the output values are sensible float32 or corrupted.

**Method:** For one sample, print from both FP32 and INT8 models:
- `dtype`, `shape`
- `min`, `max`, `mean`, `std`, `norm`
- Cosine similarity between FP32 and INT8 embeddings for the same input

**What to look for:**
- If INT8 output `min/max ≈ ±127` → raw int8 values leaking through (QDQ format bug)
- If values look reasonable but cosine_sim ≈ 0 → calibration/scaling issue
- If dtype is not float32 → output type mismatch

**Status:** [ ] Not started

---

### Step 2b — Weight & Activation Error Analysis (Layer-by-Layer)

**Goal:** Pinpoint exactly which layers introduce the most quantization error.

**Method:** Use ONNXRuntime quantization debug APIs:

```python
from onnxruntime.quantization.qdq_loss_debug import (
    create_weight_matching,
    compute_weight_error,
    create_activation_matching,
    # compute_activation_error  (or manually compute MSE/cosine per node)
)
```

**Sub-steps:**

#### 2b-i: Weight Error Analysis
```python
matched_weights = create_weight_matching(fp32_model_path, int8_model_path)
weight_errors = compute_weight_error(matched_weights)
# → DataFrame with SNR per layer — sort to find worst layers
```

#### 2b-ii: Activation Error Analysis
```python
matched_activations = create_activation_matching(
    fp32_model_path, int8_model_path, calibration_data_reader
)
# → Compare activations at each node: compute MSE, cosine_sim
# → Sort by error to find where quantization noise explodes
```

**What to look for:**
- Layers with SNR < 10 dB → weights don't survive int8 rounding
- Layers where activation MSE suddenly spikes → error amplification point
- Common culprits in ViT/CLIP:
  - Attention QKV projections (wide weight ranges)
  - Attention softmax output (very sharp distributions)
  - Layer norm (amplifies small errors)
  - Final text/image projection to embedding space

**Deliverable:** Ranked list of problematic layers → feed into Step 4.

**Status:** [ ] Not started

---

### Step 3 — Switch QDQ → QOperator Format

**Goal:** Test if the output corruption is caused by QDQ format (designed for hardware compilers, not ORT inference).

**Method:** In `quantize_local.py`, change:
```python
quant_format=QuantFormat.QDQ       # ← current (for hardware export)
quant_format=QuantFormat.QOperator  # ← try this (native ORT inference)
```

**Why this might fix it:** QDQ inserts paired Quantize/Dequantize nodes. If the final output node stays quantized (no trailing DequantizeLinear), ORT returns raw int8 values. QOperator fuses quantization into operators and always returns float32.

**Status:** [ ] Not started

---

### Step 4 — Exclude Worst Layers from Quantization

**Goal:** Surgically exclude the layers identified in Step 2b.

**Method:** Use `nodes_to_exclude` parameter in `quantize_static()`:
```python
quantize_static(
    ...,
    nodes_to_exclude=["layer_8_attn_qkv", "final_projection", ...],
)
```

**Strategy:** Start by excluding top-5 worst layers → re-quantize → test Recall@10. Add/remove layers until you find the minimal exclusion set that preserves accuracy.

**Status:** [ ] Not started

---

### Step 5 — Try Percentile Calibration

**Goal:** MinMax is sensitive to outliers in attention layers. Percentile clips extreme values.

**Method:**
```python
calibrate_method=CalibrationMethod.Percentile
# Can also try: CalibrationMethod.Entropy
```

**Status:** [ ] Not started

---

### Step 6 — Try QUInt8 Activations

**Goal:** ORT has broader kernel support for unsigned int8 activations.

**Method:**
```python
activation_type=QuantType.QUInt8   # keep weight_type=QInt8
```

**Status:** [ ] Not started

---

### Step 7 — Progressive Op Scope Narrowing

**Goal:** If all else fails, find the minimal set of quantizable op types.

**Method:** Try quantizing only one op type at a time:
```python
# Try each separately:
op_types_to_quantize=["Conv"]          # → test Recall@10
op_types_to_quantize=["MatMul"]        # → test Recall@10
op_types_to_quantize=["Conv", "MatMul"] # → test Recall@10
```

**Status:** [ ] Not started

---

### Step 8 — Fix FP32 ONNX Baseline Preprocessing Gap

**Goal:** FP32 ONNX = 0.8728 vs PyTorch = 0.8805. Small gap caused by preprocessing difference.

**Root cause:** `inference_onnx_local.py` uses `PIL.resize(224,224)` (box filter) but CLIP expects `Resize(224, bicubic) + CenterCrop(224)`.

**Fix:** Match the preprocessing in `inference_local.py`:
```python
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, InterpolationMode
preprocess = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    ToTensor(),  # /255
])
```

**Status:** [ ] Not started

---

## Execution Order

```
Step 1 + 2    →  Run together in debug_quantization.py (~5 min)
                  Confirms WHAT is broken
        ↓
Step 2b       →  Weight & activation error analysis (~10 min)
                  Identifies WHERE the error is introduced
        ↓
Step 3        →  QDQ → QOperator switch, re-quantize, test (~5 min)
                  Most likely single fix
        ↓
Step 4        →  Exclude worst layers from 2b findings (~10 min)
                  Surgical fix based on data
        ↓
Step 5 + 6    →  Calibration & dtype experiments (~5 min each)
                  Only if Steps 3-4 insufficient
        ↓
Step 7        →  Last resort — narrow op scope
        ↓
Step 8        →  Fix FP32 baseline (can do anytime, independent)
```

## Scripts to Create/Modify

| Script | Purpose |
|--------|---------|
| `debug_quantization.py` (NEW) | Steps 1, 2, 2b — all diagnostics in one run |
| `quantize_local.py` (MODIFY) | Steps 3-7 — add config flags for format, calibration, exclusions |
| `inference_onnx_local.py` (MODIFY) | Step 1 — add cross-combination mode |

## Calibration Dataset Strategy

| Phase | Dataset | Size | Why |
|-------|---------|------|-----|
| **Debugging (Steps 1–7)** | Sample dataset | 57 images, 222 texts | Sufficient — debugging structural bugs, not tuning accuracy |
| **Calibration tuning (Step 9)** | COCO val subset | 500–1000 images | Percentile/Entropy calibration benefits from diversity |
| **Fine-tuning / QAT** | COCO + Flickr30k | 5000+ images | Needed for actual training loops |

**Decision:** Use 57-sample dataset for all debugging steps. Revisit with larger dataset only
after INT8 Recall@10 > 0.80 (i.e., structural bugs are fixed, we're optimizing the last few %).

---

## Results Log

Record results here as we execute each step:

| Step | Date | Result | Notes |
|------|------|--------|-------|
| Initial | 2026-03-19 | FP32=0.8728, INT8=0.0527 | Catastrophic failure |
| | | | |
