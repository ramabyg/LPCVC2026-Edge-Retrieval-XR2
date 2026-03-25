# CLIP Optimization Plan v3 — LPCVC 2026 Track 1

**Last updated:** 2026-03-25
**Target:** Maximize Recall@10 for image-to-text retrieval on Qualcomm XR2 Gen 2
**Constraint:** ≤ 35ms combined latency (image + text encoder), hard threshold
**Current baseline:** 0.8805 Recall@10 (ViT-B/16 FP32, ~31ms on-device)

---

## Current Status Summary (Updated 2026-03-25)

| Variant | Local Recall@10 | On-Device Recall@10 | On-Device Latency | Status |
|---------|-----------------|---------------------|-------------------|--------|
| ViT-B/16 FP32 | 0.8728 | **0.7299** | 31.4ms ✓ | ⚠️ Passes latency, gap unexplained |
| ViT-B/16 INT8 qdq+qint8 | 0.8256 | 0.0433 | 39.3ms ✗ | ❌ Slower AND catastrophic accuracy |
| ViT-B/16 INT8 qdq+quint8 | 0.8256 | 0.6804 | 39.1ms ✗ | ❌ Slower AND worse accuracy |
| ViT-L/14 FP32 | 0.8857 | 0.7429 | 130.4ms ✗ | ❌ 3.7× over budget |
| ViT-L/14 INT8 qdq+qint8 | 0.8567 | 0.0625 | 144.9ms ✗ | ❌ Worse in every dimension |
| ViT-L/14 INT8 qdq+quint8 | 0.8567 | 0.7262 | 146.9ms ✗ | ❌ 4.2× over budget |

**Conclusions from sweep:**
- **ViT-L/14 is definitively abandoned** — INT8 makes it *slower* (144ms vs 121ms), not faster
- **INT8 qdq is slower than FP32** on this device: QDQ format adds ~5ms overhead (DequantizeLinear nodes, FP32→FP16 casts at input) and increases memory 2× (421 MiB vs 227 MiB)
- **Only viable current option:** ViT-B/16 FP32 at 31.4ms — passes latency but Recall@10 needs improvement
- **QOperator format** rejected by QAI Hub entirely

**Key open issues:**
1. FP32 on-device gap: 0.8728 local → 0.7299 device (−15%) — root cause unknown
2. INT8 always worse on-device — QDQ format adds overhead AND compiler re-quantizes Softmax/LayerNorm
3. Track C must pivot: ViT-L/14 is out, focus on FP16 native + architecture analysis

---

## Three-Track Plan

### Track A — Diagnose On-Device Performance Degradation (PRIORITY 1)
*Start immediately. Unblocks interpretation of all downstream results.*

#### A1: Inspect FP32 compile logs (30 min)
Open QAI Hub compile jobs for the FP32 norm-baked models:
- Image: `jgkymrqvp` — Text: `j5q2o9re5`

Look for:
- Any op fallback to CPU (would show as `RUNTIME_CPU` in op assignment)
- FP32→FP16 precision downcasting on attention ops
- Any unsupported op warnings

#### A2: Compare local ONNX vs on-device embeddings (1–2 hrs)
**File to create:** `diagnose_device_gap.py`

Steps:
1. Run `inference_onnx_local.py --mode fp32` → save all 57 image embeddings + 222 text embeddings as `.npy`
2. Download on-device output tensors from a QAI Hub inference job:
   ```python
   job = qai_hub.get_job("job_id")
   outputs = job.get_output_dataset()
   ```
3. Compute cosine similarity between local and on-device embeddings per sample
4. Report: mean cosine sim, min, max — if < 0.99 → embeddings diverge at inference level

#### A3: Verify upload_dataset.py preprocessing (30 min)
Confirm images sent to device are `/255` only — no additional CLIP mean/std.
Since normalization is baked into `ImageEncoderWrapper`, double-applying would destroy performance.

**File to check:** `upload_dataset.py`

#### A4: Verify ONNX export quality (30 min)
Run `inference_onnx_local.py --mode fp32`. Expected ONNX FP32 recall ≈ 0.8728.
If lower → `dynamo=True` tracing issue in `export_onnx.py`.

#### A5: Diagnose INT8/uint8 on-device collapse
**Hypothesis:** QAI Hub compiler applies an FP32→FP16→INT8 cascade conversion that compounds
precision loss, AND re-quantizes ops (Softmax, LayerNorm) that were excluded locally.

Investigation steps:
1. Check INT8 compile logs for image encoder `jpx7kj88g` and text encoder `jgdrx9kkp`:
   - Look for intermediate FP16 casting nodes, attention op re-quantization
2. Download Hub-quantized model and inspect vs local quantized ONNX:
   ```python
   model = qai_hub.get_job("j563y6on5").get_target_model()  # quantize job output
   ```
   Compare in Netron: count Q/DQ nodes, verify which ops were quantized
3. If FP32→FP16→INT8 cascade confirmed → try compile flags:
   - `--use_fp16_intermediate` to force FP16 only (no INT8 for sensitive ops)
   - `--force_use_onnx_defined_types` to prevent QNN overriding our quantization scheme
4. If Softmax/LayerNorm re-quantization confirmed → plan QAT (Quantization-Aware Training) in v4

---

### Track B — Fine-Tuning ViT-B/16 (LONG-RUNNING)
*Start dataset download immediately (runs in background). Training starts once data is ready.*

**Goal:** Improve Recall@10 above 0.8805 via contrastive fine-tuning on COCO + Flickr30k.

#### B1: Setup
Add to `requirements.txt`:
```
peft>=0.10.0
accelerate>=0.27.0
```

#### B2: Download datasets

**MS-COCO train2017** (~18 GB):
```bash
# Images
curl -L http://images.cocodataset.org/zips/train2017.zip -o train2017.zip
curl -L http://images.cocodataset.org/zips/val2017.zip -o val2017.zip

# Karpathy split annotations (COCO + Flickr30k in one zip, ~45 MB)
curl -L https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip -o caption_datasets.zip
```

**Flickr30k** (~4 GB):
- Images require form: https://shannon.cs.illinois.edu/DenotationGraph/data/index.html
- Annotations included in `caption_datasets.zip` above

Expected layout:
```
C:\rama\projects\data\
  coco\
    train2017\              # 118K images
    val2017\                # 5K images
    dataset_coco.json       # Karpathy split (113K train, 5K val, 5K test)
  flickr30k\
    images\                 # 31K images
    dataset_flickr30k.json  # Karpathy split (29K train)
```

#### B3: GPU Strategy

| GPU | VRAM | Recommended Strategy | Batch size | Est. time/epoch |
|-----|------|---------------------|------------|-----------------|
| RTX 3090 / 4090 | 24 GB | Full fine-tune (best gains) | 256 | ~30 min |
| RTX 3070 / 3080 | 8–10 GB | Full fine-tune or LoRA | 64–128 | ~1–2 hrs |
| GTX 1650 (fallback) | 4 GB | LoRA only | 4 + grad accum ×8 | ~10 hrs |

**Recommended order:**
1. Run LoRA first (1–3 epochs) to validate pipeline + get early signal
2. If LoRA shows improvement → switch to full fine-tuning for final model

#### B4: Implement finetune_lora.py
**File to create:** `finetune_lora.py`

**LoRA config:**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["out_proj", "c_fc", "c_proj"],
    lora_dropout=0.05, bias="none"
)
# Covers: attention output + MLP first/second layers in all 12 blocks of both encoders
# NOTE: in_proj_weight (Q/K/V fused) is a raw Parameter, not nn.Linear → PEFT can't target it
```

**Training setup:**
```python
model, _ = clip_lib.load("ViT-B/16", device=device)
model = model.float()           # CRITICAL: CLIP loads fp16 on CUDA → must cast to fp32
peft_model = get_peft_model(model, lora_config)

# CRITICAL: logit_scale is frozen by PEFT — explicitly unfreeze it
for name, p in peft_model.named_parameters():
    if "logit_scale" in name:
        p.requires_grad_(True)
```

**Hyperparameters (RTX 3080 target):**
```
batch_size=128 (no grad accumulation needed)
lr_lora=2e-4, lr_logit_scale=1e-4
scheduler=CosineAnnealingLR, warmup_steps=200
epochs=10 (validate at epoch 1, continue if improving)
AMP (fp16) with Tensor cores enabled
logit_scale clamp after each step: [0.0, 4.6052]
DataLoader: drop_last=True, num_workers=4
```

**Loss — symmetric InfoNCE:**
```python
def infonce_loss(img_feat, txt_feat, logit_scale):
    img_feat = F.normalize(img_feat, dim=-1)
    txt_feat = F.normalize(txt_feat, dim=-1)
    logits   = logit_scale * img_feat @ txt_feat.t()   # (B, B)
    labels   = torch.arange(B, device=img_feat.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
```

**Training preprocessing** (applied in dataset, NOT in model):
```python
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
train_transform = RandomResizedCrop(224, scale=(0.7,1.0), BICUBIC) +
                  RandomHorizontalFlip + ColorJitter(0.1,0.1,0.1,0.05) +
                  ToTensor + Normalize(CLIP_MEAN, CLIP_STD)
```

**Dataset class:** `CaptionDataset(json_path, image_root, split, transform)` — Karpathy JSON format.
Randomly sample one of 5 captions per image per `__getitem__`. Combine COCO + Flickr30k.

#### B5: Evaluation datasets — two-tier approach

| Dataset | Size | Purpose | When to use |
|---------|------|---------|-------------|
| Competition sample | 57 images, 222 texts | Quick sanity checks (Track A) | After any model/pipeline change |
| COCO Karpathy 1K val | 1,000 images, 5,000 texts | Fine-tuning evaluation (Track B) | After each training epoch |
| Competition test | 300 images, unknown texts | Final submission | Submitted via QAI Hub |

**Why COCO 1K val:**
- 57 images → Recall@10 has high variance; small changes are noise
- COCO Karpathy val split is guaranteed non-overlapping with COCO train (safe eval set)
- 1K images gives stable Recall@10 estimates; 5K texts gives realistic retrieval difficulty
- Closer proxy to the competition's unknown 300-image test than our 57-image sample
- Standard in CLIP literature — comparable to published results

**COCO 1K val subset:** Use images 0–999 from the Karpathy val split (`dataset_coco.json`, `split="val"`).
The full Karpathy val has 5,000 images — take the first 1,000 for speed (standard 1K test protocol).

**File to create:** `evaluate_coco1k.py` — standalone evaluation script:
```bash
python evaluate_coco1k.py                                          # evaluate current PyTorch model
python evaluate_coco1k.py --onnx                                   # evaluate ONNX FP32
python evaluate_coco1k.py --checkpoint lora_checkpoints/epoch_3   # evaluate LoRA checkpoint
```

Reports: Recall@1, Recall@5, Recall@10 (image-to-text direction, matching competition metric).

During training, call this every epoch instead of (or in addition to) the 57-image check.
Preprocessing: same as `inference_local.py` (Resize+CenterCrop+ToTensor+Normalize).

#### B6: Save and merge LoRA weights
```python
merged = peft_model.merge_and_unload()   # returns plain nn.Module, no PEFT wrapper
torch.save(merged.state_dict(), "clip_vitb16_lora_finetuned.pt")
```
**File to create:** `merge_lora.py` — standalone merge + save script.

#### B7: Integration with existing pipeline
Modify `inference_local.py` and `export_onnx.py` model loading (2 lines):
```python
model, _ = clip_lib.load("ViT-B/16", device="cpu")
model = model.float()
model.load_state_dict(torch.load("clip_vitb16_lora_finetuned.pt", map_location="cpu"))
model = model.to(device).eval()
```
`ImageEncoderWrapper` and `TextEncoderWrapper` in `export_onnx.py` work unchanged.

**Acceptance criteria:** Local Recall@10 (PyTorch) > 0.8805 after merge.

---

### Track C — Latency Optimization & XR2 Architecture Investigation (PIVOTED)
*ViT-L/14 INT8 already tried and failed (145ms). Track C now focuses on FP16 native and XR2 analysis.*

#### C1: ✅ DONE — ViT-L/14 INT8 on-device (results: 144–147ms, abandoned)
ViT-L/14 at any precision exceeds 35ms budget by 4×. Definitively not viable. No further work needed.

#### C2: Try FP16 native compile for ViT-B/16 (HIGH VALUE, quick)
**Insight:** INT8 QDQ is *slower* than FP32 (39ms vs 31ms) due to DequantizeLinear overhead.
FP16 native compute on Hexagon 69 avoids these conversion nodes and may be genuinely faster.

```bash
# Export FP16 ONNX (cast model weights + activations to fp16 before export)
# Then compile normally — QNN DLC will use FP16 ops natively
python export_onnx.py --model ViT-B/16 --dtype fp16
python compile_and_profile.py --model ViT-B/16  # compile FP16 ONNX
```

Expected result: ~13–16ms total (2× speedup over FP32 with near-zero accuracy loss).
If confirmed: latency headroom opens up; FP16 becomes the new baseline.

**File to modify:** `export_onnx.py` — add `--dtype` flag (fp32/fp16).

#### C3: XR2 Gen 2 Architecture Investigation

**Goal:** Ensure we're using the hardware optimally — NPU parallelism, VTCM memory,
FP16 native compute — to reduce latency for ViT-B/16.

**Key questions to answer:**

1. **NPU parallelism:** Does QAI Hub run image + text encoders sequentially or in parallel?
   - If sequential: submit both as a single combined graph to allow compiler to pipeline
   - Check QAI Hub profile JSON for overlap/gaps between encoder executions

2. **VTCM (Vector TCM) capacity:** Hexagon 69 has ~8MB VTCM.
   - ViT-B/16 peak activation: ~196 patches × 768 × 4 bytes ≈ 600KB — fits in VTCM ✓
   - DDR spill = major latency spike. Check profile job for memory transfer events.

3. **FP16 native compute:** Hexagon 69 supports FP16 natively — likely faster than FP32 AND avoids INT8 QDQ overhead.
   - Already confirmed: INT8 QDQ is *slower* (39ms) than FP32 (31ms) due to conversion nodes
   - FP16 avoids all of this while halving compute requirements
   - ViT-B/16 FP16 estimate: ~13–16ms → creates 19ms of headroom for future work
   - Covered in C2 above

4. **Operator fusion:** Check if QNN compiler fuses attention patterns (QKV matmul + softmax).
   - Un-fused attention = multiple memory round-trips = higher latency
   - Check compile logs for fusion warnings; try `--enable_htp_fp16_precision` if available
   - Profile job JSON shows per-layer timing — identify if attention ops have gaps between them

**File to create:** `analyze_profile.py`
- Parses QAI Hub profile job JSON output
- Groups by op type, identifies top-5 latency bottlenecks
- Checks for memory transfer events (DDR spills)
- Reports: total latency, NPU util%, memory bandwidth

**Existing profile data to analyze:**
- ViT-B/16 FP32: profile jobs from March bench march results (in `working_with_claude/commands_notes.md`)
- ViT-L/14 FP32: profile jobs from bench march results

---

## Sequencing

```
Week 1 (now):
  Track A: Day 1–2 — inspect logs, run diagnose_device_gap.py, verify preprocessing
  Track B: Day 1+  — start dataset download (18GB, background)
  Track C: Day 2   — add --dtype fp16 to export_onnx.py, submit ViT-B/16 FP16 compile job
  Track C: Day 3   — write analyze_profile.py, analyze existing ViT-B/16 FP32 profile data

Week 2:
  Track A: Apply fixes found; re-export + recompile if preprocessing bug confirmed
  Track B: Training begins (3-epoch validation run on LoRA first)
  Track C: FP16 results arrive → decision on new latency baseline

Week 3–4:
  Track B: Full fine-tuning run (if LoRA epoch 1 shows improvement)
  Track A/C: QAT planning if INT8 on-device still broken after Track A fixes
```

---

## Files Summary

| File | Track | Action |
|------|-------|--------|
| `evaluate_coco1k.py` | B | CREATE — COCO 1K val evaluation (R@1/5/10) |
| `diagnose_device_gap.py` | A | CREATE — compare local vs on-device embeddings, INT8 dtype inspection |
| `upload_dataset.py` | A | VERIFY — preprocessing must be /255 only |
| `finetune_lora.py` | B | CREATE — LoRA fine-tuning training script |
| `merge_lora.py` | B | CREATE — merge LoRA weights into base model |
| `requirements.txt` | B | ADD: `peft>=0.10.0`, `accelerate>=0.27.0` |
| `analyze_profile.py` | C | CREATE — parse QAI Hub profile JSON, find bottlenecks |
| `export_onnx.py` | C | ADD `--dtype fp16` flag for FP16 export |
| `inference_local.py` | B verify | MODIFY: load merged fine-tuned weights |

## Reusable Existing Functions

| Function | File | Used in |
|----------|------|---------|
| `evaluate_track1()` | `inference.py:29` | Track B eval loop, Track A verification |
| `ImageEncoderWrapper` | `export_onnx.py:55` | Track B — works unchanged after merge |
| `TextEncoderWrapper` | `export_onnx.py:67` | Track B — works unchanged after merge |
| `inference_onnx_local.py` | — | Track A: local ONNX baseline for gap analysis |

---

## Expected Outcomes

| Track | Success Criterion |
|-------|------------------|
| A — FP32 gap | cosine sim(local, device) > 0.99, or root cause identified + fix applied |
| A — INT8 | Compile log analysis confirms cause; fix or workaround identified |
| B — Fine-tuning | Local Recall@10 > 0.8805 after merge; ONNX FP32 > 0.8728 |
| C — FP16 compile | ViT-B/16 FP16 latency measured on-device; target < 16ms |
| C — Architecture | Top-3 latency bottlenecks identified; NPU parallelism and VTCM analysis done |
