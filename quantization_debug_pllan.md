Good, I have everything I need. Let me analyze what's happening first, then lay out the debugging plan.

---

## Root Cause Analysis (Before Debugging)

**0.0527 ≈ 10/222 ≈ 0.045 — that's essentially random chance.** The embeddings aren't slightly wrong, they're completely garbage. This rules out calibration issues and points squarely at a **format/output type problem**.

The most likely culprit chain:

```
QDQ format → ORT returns the last node's output which may still be
in quantized INT8 scale → evaluate_track1 interprets those raw int8
values as float32 → cosine similarity is computed on nonsense → ~random Recall@10
```

---

## Debugging Plan (ordered by confidence / effort)

### Step 1 — Isolate which encoder is broken
Run **cross combinations** to pinpoint the culprit:

| Image | Text | Expected result |
|-------|------|----------------|
| FP32 | INT8 | Near 0.87 if text is fine, near random if text is broken |
| INT8 | FP32 | Near 0.87 if image is fine, near random if image is broken |

This immediately tells you whether one or both encoders are bad. Implement as a flag in `inference_onnx_local.py`.

---

### Step 2 — Inspect raw embedding output (the smoking gun)
Before computing Recall@10, print stats for a **single sample** from both FP32 and INT8:
```
FP32 embedding:  dtype=float32  shape=(1,512)  min=-2.14  max=3.87  norm=18.4
INT8 embedding:  dtype=float32  shape=(1,512)  min=-127.0 max=127.0  norm=???
cosine_sim(fp32_embed, int8_embed) = ???
```
- If `min/max ≈ ±127` → the output is raw int8 values being misread as float32 → **QDQ format bug**
- If shape/dtype is wrong → something deeper in the graph
- If values look reasonable but cosine sim is ~0 → calibration/scaling issue

---

### Step 3 — Switch QDQ → QOperator (highest-confidence fix)
`QuantFormat.QDQ` is designed for **export to hardware compilers** (QAI Hub, TensorRT). It inserts paired Q/DQ nodes and the model's output may remain in int8. **For local ORT inference, `QuantFormat.QOperator` is the right format** — it fuses quantization into ops and always outputs float32.

```python
# Change this:
quant_format=QuantFormat.QDQ

# To this:
quant_format=QuantFormat.QOperator
```

---

### Step 4 — Switch activation type from QInt8 → QUInt8
ORT's CPU kernel support for **signed int8 activations** is limited — many ops fall back to float32 silently (or error). Unsigned int8 (`QUInt8`) has broader kernel coverage for activations:

```python
activation_type=QuantType.QUInt8  # weights can stay QInt8
```

---

### Step 5 — Fix calibration: MinMax → Percentile
MinMax captures the absolute worst-case outliers in the activation range. Transformer **attention softmax** produces very sharp distributions — one outlier can cause the entire int8 range to collapse for other values:

```python
calibrate_method=CalibrationMethod.Percentile,
# optionally: extra_options={"percentile": 99.99}
```

---

### Step 6 — Exclude the output embedding node from quantization
Even if intermediate ops are quantized correctly, the **final output node being quantized** corrupts what the caller receives. Force the output to stay float32:

```python
# In quantize_static(), add:
nodes_to_exclude=[<output_node_name>]
# or use: extra_options={"ActivationSymmetric": True}
```

This requires first inspecting the model graph to get the last node name.

---

### Step 7 — Progressively widen the exclusion list
If still broken, find the specific op(s) causing the crash by quantizing fewer and fewer op types:

```
All ops → Conv only → MatMul only → Gemm only → Conv+MatMul → etc.
```

Find the smallest set that gives acceptable Recall@10, then decide if that's good enough.

---

### Step 8 — Validate the FP32 ONNX gap (secondary issue)
Note: FP32 ONNX = 0.8728 vs PyTorch = 0.8805. That small gap is because `inference_onnx_local.py` does `resize(224,224)` (PIL box resize) instead of `Resize(224) + CenterCrop(224)` (bicubic, which CLIP expects). Fix the preprocessing to match `inference_local.py` so you have a true FP32 ONNX baseline before measuring INT8 degradation.

---

## Execution Order

```
Step 2 → print embedding stats first (5 min, no retraining needed)
  ↓ confirms QDQ format bug?
Step 3 → switch to QOperator, re-quantize, test (~5 min)
  ↓ still bad?
Step 4 → switch activations to QUInt8 (~5 min)
  ↓ still bad?
Step 5 → switch to Percentile calibration (~5 min)
  ↓ still bad?
Step 6 → exclude output node (~10 min, need to inspect graph)
  ↓ still bad?
Step 7 → narrow op scope progressively
Step 8 → fix FP32 baseline (parallel, anytime)
```

Ready to start implementing? I'd suggest beginning with **Step 2** (embedding inspection) and **Step 3** (QOperator) in parallel — they'll likely solve it immediately.