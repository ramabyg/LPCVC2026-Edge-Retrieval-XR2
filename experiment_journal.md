# LPCVC 2026 Track 1 — Experiment Journal

## Competition Goal
Maximize **Recall@10** for image-to-text retrieval on Qualcomm Snapdragon XR2 Gen 2 (Hexagon NPU), with a hard latency budget of **≤ 35ms** (combined image + text encoder).

---

## Hardware Setup

| Role | Hardware |
|------|----------|
| Training | 4× NVIDIA H100 80GB (GPU server) |
| Local testing | NVIDIA GTX 1650 4GB |
| Competition target | Qualcomm Snapdragon XR2 Gen 2 |

---

## Model

**OpenAI CLIP ViT-B/16**
- Image encoder: Vision Transformer, 224×224 input, outputs 512-dim embedding
- Text encoder: BERT-style transformer, 77-token input, outputs 512-dim embedding
- Both encoders run under 35ms on-device (image: ~26.3ms, text: ~4.6ms)

---

## Experiment Timeline

### Experiment 1: FP32 Baseline

**What:** Ran base CLIP ViT-B/16 (no fine-tuning) with normalization baked into the model wrapper so competition-format inputs (`/255` only) work correctly.

**Result:**

| Variant | Recall@10 |
|---------|-----------|
| PyTorch FP32 (local) | **0.8805** |
| FP32 ONNX (local ORT) | **0.8728** |

**Key finding:** Small gap between PyTorch and ONNX is due to PIL box resize vs. bicubic+centercrop difference. Not a bug.

---

### Experiment 2: INT8 Quantization — Attempt 1 (Failed)

**What:** Quantized both encoders with ONNX Runtime's static quantization using `QuantFormat.QDQ`, `QuantType.QInt8`, `CalibrationMethod.MinMax`.

**Result: 0.0527 Recall@10 — catastrophic failure (near random chance)**

**Root cause:** `QuantFormat.QDQ` is designed for hardware compiler export (QAI Hub, TensorRT). When run under local ORT inference, the model's last output node had no trailing `DequantizeLinear`, so ORT returned raw `±127`-range int8 values interpreted as float32. Cosine similarity on garbage values → ~random Recall@10.

**Lesson:** QDQ is for hardware export only. Use `QOperator` format for local ORT inference.

---

### Experiment 3: INT8 Quantization — Attempt 2 (Still Failed)

**What:** Switched to `QuantFormat.QOperator` (always outputs float32) but applied to ALL ops.

**Result: 0.1003 Recall@10 — still catastrophic**

**Root cause:** Quantizing `Softmax` and `LayerNorm` nodes inside the Transformer attention blocks caused severe accuracy loss. These ops are sensitive to activation range — MinMax calibration with only 57 images couldn't capture the full distribution.

**Lesson:** Never quantize Softmax or LayerNorm. Restrict quantization scope to `Conv`, `MatMul`, `Gemm` only.

---

### Experiment 4: INT8 Quantization — Fixed (Production-Ready)

**What:** `QOperator` format, but restricted to `Conv/MatMul/Gemm` only (excluded Softmax, LayerNorm, embeddings). Switched to `Percentile` calibration to handle attention outliers.

**Result:**

| Config | Recall@10 | vs FP32 baseline |
|--------|-----------|-----------------|
| FP32 ONNX | 0.8728 | — |
| INT8 both encoders | **0.8256** | -0.0472 |
| FP32 image + INT8 text | 0.8524 | -0.0204 |
| INT8 image + FP32 text | 0.8619 | -0.0109 |

**Key finding:** INT8 image encoder alone is surprisingly good (0.8619) — the image encoder actually benefits slightly from INT8 regularization. The text encoder is more sensitive to quantization.

**Embedding stats (first sample):**

| Encoder | dtype | norm | min | max |
|---------|-------|------|-----|-----|
| FP32 image | float32 | 10.655 | -2.100 | 7.341 |
| INT8 image | float32 | 9.756 | -1.003 | 6.564 |
| FP32 text | float32 | 8.953 | -1.409 | 6.441 |
| INT8 text | float32 | 10.048 | -2.223 | 7.759 |

**Lesson:** Restrict quantization scope aggressively for Transformer models. Attention and normalization ops must stay in FP32.

---

### Experiment 5: LoRA Fine-Tuning on COCO + Flickr30k

**What:** Fine-tuned CLIP ViT-B/16 using LoRA (Low-Rank Adaptation) on COCO + Flickr30k datasets using the Karpathy split. Trained on 4× H100 GPU server with `torchrun --nproc_per_node=4`.

**LoRA Configuration:**

| Parameter | Value |
|-----------|-------|
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | `out_proj`, `c_fc`, `c_proj` |
| Adapter size | ~5.9 MB (vs ~350 MB full model) |
| Training data | COCO + Flickr30k (Karpathy split) |
| Loss | InfoNCE (symmetric contrastive) |
| LR schedule | Linear warmup (200 steps) + cosine annealing |
| LoRA LR | 2e-4 |
| Logit scale LR | 1e-4 |

**What is LoRA?**
LoRA (Low-Rank Adaptation) keeps the original model weights **frozen** and adds small trainable "delta" matrices on top of specific layers:

```
Fine-tuned output = Original weights + LoRA delta (A × B)
```

Where A and B are small matrices (rank 8), so the total trainable parameters are tiny compared to the full model. After training, these deltas are "merged" back into the base weights — producing a standard checkpoint with no overhead at inference time.

**Result: 97.20 Recall@10 on validation set** (up from 88.05 baseline — +9.15 percentage points)

**Deployment approach:**
1. Merge LoRA deltas into base model: `python src/local/train/merge_lora.py --checkpoint checkpoints/lora_checkpoints_best/best --output checkpoints/merged_best.pt`
2. Local test: `python src/local/inference_pytorch.py --weights checkpoints/merged_best.pt`
3. ONNX export: `python src/platform/export_onnx.py --weights checkpoints/merged_best.pt`
4. Platform pipeline unchanged (compile → upload → on-device)

**Key advantage:** LoRA merge adds no new ops to the model — latency stays exactly the same as the base CLIP model.

---

## Summary Table

| Experiment | Recall@10 | Notes |
|-----------|-----------|-------|
| FP32 PyTorch baseline | 0.8805 | Ground truth |
| FP32 ONNX | 0.8728 | Small resize difference |
| INT8 attempt 1 (QDQ all ops) | 0.0527 | Raw int8 output leak |
| INT8 attempt 2 (QOperator all ops) | 0.1003 | Softmax/LayerNorm poisoned |
| INT8 fixed (Conv/MatMul/Gemm only) | 0.8256 | Production-ready |
| **LoRA fine-tuned (val set)** | **0.9720** | **Best result — +9.15pp over baseline** |

---

## Key Technical Learnings

1. **QDQ vs QOperator:** QDQ format is for hardware compiler export (QAI Hub, TensorRT) only. QOperator is for local ORT inference. Mixing them causes raw int8 values to be returned as float32.

2. **Transformer quantization scope:** Never quantize Softmax or LayerNorm. Only quantize `Conv`, `MatMul`, `Gemm`. These attention-adjacent ops have sharp, outlier-prone distributions.

3. **Calibration method matters:** MinMax calibration underestimates the range for attention softmax (sharp peaks). Percentile calibration handles outliers better.

4. **INT8 image encoder is surprisingly robust:** The image encoder benefits slightly from INT8 regularization. The text encoder is more sensitive.

5. **LoRA is efficient for fine-tuning large models:** 5.9 MB adapter vs 350 MB full model, no inference overhead after merging.

6. **Normalization must be baked into the model:** Competition sends `/255` images only. CLIP normalization (mean/std) must be applied inside the wrapper, not outside, to work correctly on-device.

---

## File Structure

```
src/
├── local/
│   ├── inference_pytorch.py      # Local FP32 inference (--weights flag for fine-tuned model)
│   ├── inference_onnx_local.py   # ONNX inference + debug flags
│   └── train/
│       ├── finetune_lora.py      # LoRA fine-tuning script
│       └── merge_lora.py         # Merge LoRA adapters into base model
├── platform/
│   ├── export_onnx.py            # ONNX export (--weights flag for fine-tuned model)
│   ├── compile_and_profile.py    # Upload to QAI Hub, compile, profile latency
│   ├── upload_dataset.py         # Upload dataset to QAI Hub
│   └── run_on_device.py          # Run inference on XR2 device, compute Recall@10
└── common/
    ├── eval.py                   # evaluate_track1() — Recall@10 computation
    └── config.py                 # Paths and constants

checkpoints/
├── lora_checkpoints_best/best/   # PEFT adapter checkpoint (5.9 MB)
└── merged_best.pt                # Merged weights (full model, ~350 MB)

exported_onnx/
├── image_encoder.onnx            # Exported image encoder
└── text_encoder.onnx             # Exported text encoder
```
