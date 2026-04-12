# CLIP Optimization Plan v4 — LPCVC 2026 Track 1

**Last updated:** 2026-04-08
**Target:** Maximize Recall@10 for image-to-text retrieval on Qualcomm XR2 Gen 2
**Constraint:** ≤ 35ms combined latency (image + text encoder), hard threshold

---

## Current State (as of 2026-04-08)

### Competition Results (actual test set — 300 images)
| Run | Competition Recall@10 | Notes |
|-----|-----------------------|-------|
| ViT-B/16 FP32 baseline (no fine-tune) | 0.4923 | First submission |
| ViT-B/16 FP32 + LoRA v1 (COCO only, 3 epochs) | **0.5459** | +0.054 improvement |

### Sample Dataset Results (57 images / 222 texts)
| Variant | Local Recall@10 | On-Device Recall@10 | Latency | Status |
|---------|-----------------|---------------------|---------|--------|
| ViT-B/16 FP32 (base) | 0.8728 (ONNX) / 0.8805 (PyTorch) | 0.7299 | 31.4ms ✓ | Baseline |
| ViT-B/16 FP32 (fine-tuned v1) | 0.8909 | **0.8909** | 31.4ms ✓ | **Current best** |
| ViT-B/16 INT8 QDQ+quint8 | 0.8256 | 0.6804 | 39.1ms ✗ | ❌ Slower than FP32 |
| ViT-L/14 FP32 | 0.8857–0.9003 | 0.7429 | 130+ms ✗ | ❌ 4× over budget |

### Dead Ends (from v3 — do not revisit)
- **ViT-L/14**: 130ms+ at any precision. 4× over budget. Abandoned.
- **W8A16**: XR2 Gen 2 NPU does not support INT16 activations.
- **QOperator format**: Rejected by QAI Hub entirely — only QDQ accepted.
- **Platform-side quantization** (quantize_and_compile.py): Destroys Softmax/LayerNorm → 0.027 R@10.

### Open Problem: INT8 QDQ is SLOWER than FP32
- FP32: 31.4ms → INT8 QDQ: 39.1ms (+8ms overhead)
- Root cause (observed): DequantizeLinear nodes + FP32→FP16 conversion ops inserted by QNN compiler
- Memory also increases: 421 MiB (INT8) vs 227 MiB (FP32) — dual buffer overhead
- **This is the #1 technical blocker** — fixing quantization latency unlocks bigger/better models

---

## Three-Track Plan (all run in parallel)

---

### Track A — Fine-Tuning v2

**Goal:** Improve competition Recall@10 from 0.5459 → 0.65+

**Why v1 underperformed:**
- Trained on COCO only (~113K image-text pairs) — limited caption diversity
- Patience=3 caused early stopping at epoch 3 — model likely not converged
- LoRA rank r=8 may underfit for distribution adaptation to competition test set

**Changes from v1:**

| Param | v1 | v2 | Rationale |
|-------|----|----|-----------|
| Datasets | COCO only | COCO + Flickr30k (`--datasets both`) | +29K images, more diverse captions |
| Patience | 3 | 5 | Avoid premature stopping |
| LoRA rank | r=8 | r=16 | 2× adapter capacity (~11.8MB vs ~5.9MB) |
| LoRA alpha | 16 | 32 | Keep alpha/r=2 ratio |
| Epochs | 10 | 15 | Allow full convergence |
| Warmup steps | 200 | 500 | Scale with larger dataset |
| Save dir | lora_checkpoints/ | lora_checkpoints_v2/ | Preserve v1 results |

**Command (GPU server, 4× H100):**
```bash
torchrun --nproc_per_node=4 src/local/train/finetune_lora.py \
    --datasets both \
    --epochs 15 \
    --batch-size 128 \
    --lr 2e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --patience 5 \
    --warmup-steps 500 \
    --eval-every 1 \
    --amp \
    --val-max-images 1000 \
    --save-dir lora_checkpoints_v2
```

**After training — merge and validate:**
```bash
python src/local/train/merge_lora.py \
    --checkpoint lora_checkpoints_v2/best \
    --output checkpoints/merged_v2_best.pt

python src/local/inference_pytorch.py --weights checkpoints/merged_v2_best.pt

python src/platform/export_onnx.py --weights checkpoints/merged_v2_best.pt
python src/platform/run_on_device.py
```

**Decision gates:**
- Val R@10 < 85% at epoch 1 → verify Flickr30k data path in `src/common/config.py`
- Early stop fires before epoch 7 → increase `--patience 7`, reduce lr to `1e-4`
- Local R@10 after merge < 0.8909 → overfitting; try `--lora-alpha 16` (reduce)

**Code changes:** None — all configurable via CLI args.

**Critical files:**
- `src/local/train/finetune_lora.py` — run with new args
- `src/local/train/merge_lora.py` — merge best checkpoint
- `src/common/config.py` — verify `FLICKR30K_JSON` and `FLICKR30K_IMG_DIR` paths on GPU server

---

### Track B — Quantization & Latency Troubleshooting (PRIORITY)

**Goal:** Make quantized/reduced-precision inference genuinely faster than FP32 on XR2 Gen 2.
This is the #1 blocker — if solved, bigger models (ViT-L/14, MobileCLIP2-S3) become feasible.

**Three investigation paths (run all in parallel — each is a different compile job):**

#### B1: QAI Hub Compile-Time Quantization

Instead of quantizing locally to QDQ ONNX and then compiling, let QAI Hub handle both:

```python
# In compile_and_profile.py, add compile option:
options = "--target_runtime qnn_dlc --truncate_64bit_io --quantize_full_type int8"
```

**Hypothesis:** QAI Hub's compiler can fuse Q/DQ nodes during compilation, avoiding the
DequantizeLinear overhead we see with pre-quantized QDQ ONNX. The compiler has full graph
visibility and can optimize the quantization scheme for the target hardware.

**What to measure:** Latency, memory, and Recall@10 vs our local QDQ approach.

#### B2: FP16 Native Precision

Force the HTP (Hexagon Tensor Processor) to run all ops in FP16:

```python
# In compile_and_profile.py, add compile option:
options = "--target_runtime qnn_dlc --truncate_64bit_io --qnn_options default_graph_htp_precision=FLOAT16"
```

**Hypothesis:** FP32 on Hexagon 69 may already be running mixed FP32/FP16 internally. Explicitly
requesting FP16 removes any FP32 ops, halves memory bandwidth, and avoids precision conversion.
Expected: ~15ms (2× speedup) with minimal accuracy loss (<0.3pp).

**No model changes needed** — same FP32 ONNX, different compile flag.

#### B3: Profile Analysis — Identify Overhead Sources

Download and compare profile JSONs for FP32 vs INT8 compile jobs:

```python
import qai_hub
# FP32 profile
fp32_job = qai_hub.get_job("jpr4ynz7g")  # ViT-B/16 FP32 image profile
# INT8 profile
int8_job = qai_hub.get_job("jpx12eklg")  # ViT-B/16 INT8 image profile
```

**Analysis targets:**
1. Count DequantizeLinear nodes in INT8 graph vs total ops
2. Identify per-layer latency breakdown — which layers add the most overhead?
3. Check for FP32→FP16 conversion nodes inserted by compiler
4. Check memory transfer events (DDR spills) — INT8 uses 421 MiB vs 227 MiB FP32
5. Compare NPU utilization % between FP32 and INT8 runs

**Code changes needed:**

**`src/platform/compile_and_profile.py`:**
- Add `--precision` flag: choices = ["fp32", "fp16", "int8-compile", "int8-local"]
- "fp32": current default (no change)
- "fp16": adds `--qnn_options default_graph_htp_precision=FLOAT16` to compile options
- "int8-compile": adds `--quantize_full_type int8` to compile options (uses FP32 ONNX input)
- "int8-local": uses locally quantized QDQ ONNX (current behavior with `--int8` flag)

**`src/platform/run_on_device.py`:**
- Same `--precision` flag and path logic

**`src/platform/export_onnx.py`:**
- Add `--dtype fp16` flag for explicit FP16 export (optional — B2 may not need this if
  compile flag alone handles FP16 casting)
- In `ImageEncoderWrapper.forward()`: add `images = images.to(self.mean.dtype)` for dtype safety

**Expected outcomes matrix:**

| Approach | Expected latency | Expected R@10 | Risk |
|----------|-----------------|---------------|------|
| B1: Compile-time INT8 | ~16-20ms | ~0.82-0.87 | Medium — depends on QAI Hub's quantizer quality |
| B2: FP16 native | ~15-18ms | ~0.87 (near FP32) | Low — FP16 is well-supported on Hexagon 69 |
| B3: Profile analysis | N/A (diagnostic) | N/A | None — read-only |

**Decision gates:**
- B1 latency < 20ms AND R@10 > 0.80 → adopt compile-time INT8 as default
- B2 latency < 20ms AND R@10 within 0.3pp of FP32 → adopt FP16 as default
- B1 OR B2 gives ~16ms → ViT-L/14 class models become feasible (~65ms FP32 → ~32ms FP16/INT8)
- Both B1 and B2 fail → stay with FP32 31.4ms, focus entirely on Track A + Track C

---

### Track C — Model Architecture Exploration

**Goal:** Find a base model with higher Recall@10 ceiling than OpenAI CLIP ViT-B/16.

#### Published Benchmarks (COCO Image-to-Text, zero-shot)

| Model | COCO I→T R@10 | Architecture | Params (img+txt) | Est. XR2 FP32 latency | Integration |
|-------|---------------|-------------|-------------------|----------------------|-------------|
| OpenAI CLIP ViT-B/16 | ~75.6 | ViT-B/16 | 86M+63M | 31.4ms (measured) | Current |
| EVA-CLIP ViT-B/16 | ~78-80 | Identical to above | 86M+63M | 31ms (identical) | Drop-in (OpenCLIP) |
| DataComp CLIP ViT-B/16 | ~78 | Identical | 86M+63M | 31ms (identical) | Drop-in (OpenCLIP) |
| SigLIP ViT-B/16 | **84.2** | ViT-B/16 | ~86M+63M | ~31ms | Medium (HF transformers) |
| SigLIP-2 ViT-B/16 | **85.5** | ViT-B/16 | ~86M+63M | ~31ms | Medium (HF transformers) |
| MobileCLIP2-B | ~79% IN | Hybrid MobileNet+ViT | 86M+63M | ~14ms (iPhone est.) | High (Apple API) |
| MobileCLIP2-S3 | ~81% IN | Hybrid | 125M+124M | ~15ms (iPhone est.) | High (Apple API) |

*Note: MobileCLIP numbers are ImageNet zero-shot accuracy, not COCO retrieval R@10.*

#### Evaluation Order

**Step 1: EVA-CLIP ViT-B/16 (30 min, drop-in)**

Lowest risk — identical ViT-B/16 architecture, different pretraining. Uses OpenCLIP library.

```bash
pip install open_clip_torch
```

Changes to `src/local/inference_pytorch.py`:
```python
# Add EVA-CLIP loading branch:
if args.model == "EVA02-B-16":
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'EVA02-B-16', pretrained='merged2b_s8b_b131k'
    )
    tokenizer = open_clip.get_tokenizer('EVA02-B-16')
```

CLIP normalization constants are identical — `ImageEncoderWrapper` buffers unchanged.
If local R@10 > 0.8805: proceed to ONNX export + on-device test.

**Step 2: SigLIP-2 ViT-B/16 (2-3 hrs, wrapper changes needed)**

Highest potential gain (+9.9pp retrieval R@10 over CLIP). Uses sigmoid loss → better for
image-to-text retrieval (exactly our competition metric).

```bash
pip install transformers
```

Integration requires:
- New `SigLIPImageEncoderWrapper` and `SigLIPTextEncoderWrapper` in `export_onnx.py`
- SigLIP uses `SiglipVisionModel` and `SiglipTextModel` (HuggingFace API, not OpenAI CLIP API)
- Different tokenizer (SentencePiece-based, not BPE)
- Verify normalization constants — may differ from CLIP
- Embedding dimension: 768 (not 512) for SigLIP-B/16 → `evaluate_track1()` still works (cosine sim)

**Step 3: MobileCLIP2 (only if Track B unlocks latency headroom)**

Purpose-built for edge. Much faster but lower absolute retrieval quality.
Only evaluate if we have >15ms of latency headroom from Track B.

**Decision gates:**
- EVA-CLIP local R@10 < 0.88 → check normalization; if correct, skip EVA-CLIP
- EVA-CLIP on-device R@10 > 0.8909 → adopt as new base model for Track A fine-tuning
- SigLIP-2 local R@10 > 0.90 → high priority; invest in full wrapper integration
- SigLIP-2 ONNX export fails → check for unsupported ops; try `dynamo=False`

**Code changes needed:**

**`src/local/inference_pytorch.py`:**
- Add `--model` choices: "ViT-B/16", "ViT-L/14", "EVA02-B-16", "SigLIP-2-B-16"
- Add OpenCLIP and HuggingFace loading branches

**`src/platform/export_onnx.py`:**
- Add model loading branches for EVA-CLIP (via open_clip) and SigLIP (via transformers)
- Add `SigLIPImageEncoderWrapper` / `SigLIPTextEncoderWrapper` if normalization differs
- Verify embedding output shape matches (512 for CLIP/EVA, 768 for SigLIP)

---

## Sequencing

```
Day 1:
  GPU server:  Launch Track A training (runs 8-24 hours in background)
  Local:       Track B — submit 3 compile jobs with different flags (B1, B2 use existing FP32 ONNX)
  Local:       Track C Step 1 — EVA-CLIP local test (~30 min)

Day 1-2:
  QAI Hub:     Track B results arrive (compile jobs take ~30 min each)
               → Compare: B1 (compile-time INT8) vs B2 (FP16) vs current FP32
  Local:       Track C Step 2 — SigLIP-2 integration if EVA-CLIP shows promise

Day 2-3:
  Combine:     Best Track B precision + Track C model → on-device test
  GPU server:  Track A training completes → merge → validate

Day 3-5:
  Integration: Best model (Track C) + best weights (Track A) + best precision (Track B)
  Submit:      Final competition entry
```

---

## Target Outcomes

| Combination | Expected competition R@10 | Latency |
|-------------|--------------------------|---------|
| Track A only (fine-tune v2, FP32) | ~0.58–0.65 | 31.4ms |
| Track B only (FP16 or INT8 fix) | ~0.55 (same model) | ~15-20ms |
| Track C only (SigLIP-2, FP32) | ~0.60–0.70 | ~31ms |
| **A + B** (fine-tune + FP16) | ~0.65–0.72 | ~15-20ms |
| **A + C** (fine-tune SigLIP-2) | ~0.70–0.78 | ~31ms |
| **A + B + C** (fine-tune SigLIP-2 + FP16) | **~0.75–0.82** | ~15-20ms |

**Primary target:** A + B (lowest risk, highest confidence)
**Stretch target:** A + B + C (if SigLIP-2 integration works and Track B unlocks latency)

---

## Critical Files Summary

| File | Track | Change |
|------|-------|--------|
| `src/platform/compile_and_profile.py` | B | Add `--precision` flag (fp32/fp16/int8-compile/int8-local) |
| `src/platform/run_on_device.py` | B | Same `--precision` flag |
| `src/platform/export_onnx.py` | B, C | Add `--dtype fp16`, model loading branches, `images.to(self.mean.dtype)` |
| `src/local/inference_pytorch.py` | C | Add EVA-CLIP + SigLIP-2 loading branches |
| `src/local/train/finetune_lora.py` | A | No code changes — different CLI args only |
| `src/common/config.py` | A | Verify Flickr30k paths on GPU server |

## Reusable Functions (no changes needed)
- `evaluate_track1()` in `src/common/eval.py`
- `ImageEncoderWrapper` / `TextEncoderWrapper` in `src/platform/export_onnx.py` (for CLIP/EVA-CLIP)
- `src/local/train/merge_lora.py` — run as-is

---

## Changes from v3

| Area | v3 | v4 |
|------|----|----|
| Fine-tuning | Planned but not run | v1 results in, v2 planned with Flickr30k + higher rank |
| INT8 strategy | Identified QDQ overhead | 3 parallel investigation paths (compile-time, FP16, profiling) |
| Model exploration | "ViT-L/14 abandoned" only | Concrete candidate list with published benchmarks |
| FP16 | "Try FP16 native compile" | Specific QAI Hub flag identified (`default_graph_htp_precision=FLOAT16`) |
| Priority | Track A first | Track B (quantization fix) is #1 — unlocks everything else |
