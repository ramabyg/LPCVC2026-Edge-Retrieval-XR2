# Phase 1: INT8 Quantization Fix — Results & Learnings

## Summary

Fixed catastrophic INT8 image encoder failure (Recall@10 0.07 → 0.83) by excluding
Softmax and LayerNormalization from INT8 quantization. ViT-B/16 INT8 is now viable
for production, with only a 4.7% Recall@10 drop from FP32.

---

## Measured Baselines (Updated)

| Variant | Recall@10 | Notes |
|---------|-----------|-------|
| PyTorch FP32 (local) | **0.8805** | Ground truth — `inference_local.py` |
| FP32 ONNX (local ORT) | **0.8728** | Small gap from PIL box resize vs bicubic+centercrop |
| INT8 ONNX — attempt 1 (QDQ + all ops) | **0.0527** | Catastrophic — raw int8 output leak |
| INT8 ONNX — attempt 2 (QOperator, all ops) | **0.1003** | Still catastrophic — Softmax/LayerNorm poisoned |
| **INT8 ONNX — fixed (QOperator, Conv/MatMul/Gemm only)** | **0.8256** | ✅ Production-ready |

---

## Root Cause Progression

### Bug 1: QDQ format → raw int8 output leak (fixed in prior session)

`QuantFormat.QDQ` is designed for hardware compilers (QAI Hub, TensorRT). Under local
ORT inference, the last output node may lack a trailing `DequantizeLinear`, causing ORT
to return raw ±127-range int8 values interpreted as float32.

**Fix:** Switch to `QuantFormat.QOperator` (default). QOperator fuses Q/DQ into ops
and always outputs float32.

### Bug 2: Softmax + LayerNorm INT8 quantization (fixed in this session)

Even with QOperator, INT8 quantization of `Softmax` and `LayerNormalization` in ViT
causes catastrophic accuracy loss:

- **Softmax:** Produces very sharp probability distributions. INT8 collapses all
  non-peak attention weights to zero, breaking attention heads.
- **LayerNormalization:** INT8 divisor corrupts the normalization, propagating error
  through all 12 transformer layers.

The result is not just magnitude loss — the INT8 embeddings point in **wrong directions**
(cosine_sim dropped from 0.2565 to 0.2007 for the same image-text pair).

**Fix:** Restrict image encoder to `op_types_to_quantize=["Conv", "MatMul", "Gemm"]`.
Softmax and LayerNorm are left in FP32.

---

## Final Quantization Settings (Validated)

| Setting | Value |
|---------|-------|
| Format | `QOperator` |
| Weight type | `QInt8` |
| Activation type | `QUInt8` |
| Calibration method | `Percentile` |
| Per-channel | `True` |
| **Image encoder scope** | **`Conv, MatMul, Gemm` — excludes Softmax, LayerNorm** |
| Text encoder scope | `MatMul, Gemm` (unchanged — was already conservative) |
| Calibration data | 56 images, 211 text prompts |

---

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

## Code Changes (quantize_local.py)

```python
# In quantize_image_encoder():
quantize_static(
    ...
    per_channel=True,
    op_types_to_quantize=["Conv", "MatMul", "Gemm"],  # KEY FIX: excludes Softmax, LayerNorm
)
```

New CLI flag added:
```bash
python quantize_local.py --optimize-graph   # opt-in: transformer graph fusion before quantization
```

---

## Options for Further Improvement

| Option | Expected gain | How |
|--------|-------------|-----|
| **Larger calibration set** | Medium | Use COCO/Flickr30k (~1K images) for better activation coverage |
| **Entropy calibration** | Low-Medium | `--calibration entropy` — minimizes KL divergence |
| **Transformer graph fusion** | Low-Medium | `--optimize-graph` flag (already implemented) |
| **QAI Hub quantization** | Unknown | QAI Hub's hardware-aware quantizer may do better for Hexagon NPU |
| **Layer-by-layer debug** | High | `qdq_loss_debug` to find worst layers → exclude via `nodes_to_exclude` |

---

## Key Learnings

1. **QDQ vs QOperator:** QDQ is for hardware export; QOperator is for local ORT inference.
2. **Softmax + LayerNorm must stay FP32 in ViT** — INT8 breaks attention geometry.
3. **Text encoder was already conservative** (`MatMul/Gemm` only) — that's why it worked.
4. **Direction matters more than magnitude** — cosine similarity detects subtle embedding corruption that norm checks miss.
5. **Mixed-precision isolates culprits** — `--mode int8_img_fp32_txt` vs `--mode fp32_img_int8_txt` pinpoints which encoder is broken without full debug tooling.

---

## Next: Phase 2 — ViT-L/14

- Export ViT-L/14 ONNX (`export_onnx.py --model ViT-L/14`)
- Same INT8 fix applies (`num_heads=16`, `hidden_size=1024` for graph optimization)
- On-device target: INT8 ViT-L/14 ≤ 35ms combined on XR2 Gen 2
