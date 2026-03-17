# CLIP Optimization Plan v2 — LPCVC 2026 Track 1
## Image-to-Text Retrieval on Qualcomm XR2 Gen 2

**Last updated:** March 12, 2026

---

## Competition Constraints

| Constraint | Value | Notes |
|-----------|-------|-------|
| **Latency budget** | ≤ 35ms combined (image + text encoder) | Hard threshold — submissions above this are rejected |
| **Ranking metric** | Recall@10 (higher is better) | Only scored if latency passes |
| **Input format** | `float32 (1, 3, 224, 224)` | Images pre-resized to 224×224, divided by 255 only |
| **Normalization** | NOT applied by competition | Must be baked into the model if needed |
| **Input resolution** | Fixed at 224×224 | Cannot change — competition controls preprocessing |
| **Target device** | Snapdragon XR2 Gen 2 | Hexagon NPU optimized for INT8 |
| **Submission window** | March 1 – April 30, 2026 | ~7 weeks remaining |

---

## Current Baseline (as of March 12, 2026)

| Metric | Value |
|--------|-------|
| Model | ViT-B/16 (OpenAI CLIP, 86M params) |
| Image encoder latency (FP32) | 26.3ms |
| Text encoder latency (FP32) | 4.6ms |
| **Combined latency (FP32)** | **~31ms** (under 35ms ✓) |
| Latency headroom | ~4ms |
| Local Recall@10 | TBD (with correct preprocessing) |
| On-device Recall@10 | TBD (normalization not baked in yet) |

**Key insight:** We are already under 35ms with the base ViT-B/16 in FP32. This means our
primary goal is **maximizing Recall@10**, not reducing model size. We have budget to even
explore larger models if INT8 quantization is used.

---

## Strategy Overview

Since accuracy is the ranking criterion and we already meet the latency threshold, our
strategy inverts the typical "shrink the model" approach:

1. **Fix what's broken** — bake normalization into the model (free accuracy gain)
2. **Explore upward** — can a larger model (ViT-L/14) fit under 35ms with INT8?
3. **Fine-tune for retrieval** — use COCO + Flickr30k to boost retrieval quality
4. **Quantize intelligently** — use INT8 to create headroom, not just for speed
5. **Knowledge distillation** — if ViT-L/14 doesn't fit, distill its knowledge into ViT-B/16

---

## Phase 0: Foundations & Baseline (Days 1–5)

### 0.1 Bake CLIP Normalization into the Model

**Priority: CRITICAL | Effort: 2–3 hours | No retraining needed**

The competition feeds images normalized only by `/255`. CLIP expects additional ImageNet
normalization: `(x - mean) / std` with `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

Without this, every image on-device has incorrect pixel statistics, severely hurting Recall@10.

**What to do:**
- Modify `ImageEncoderWrapper` in `export_onnx.py` to add a normalization layer as the
  very first operation, before the image hits `model.encode_image()`
- This is a purely mathematical transform — no retraining or fine-tuning required
- The pretrained weights remain exactly the same; we're just ensuring the model receives
  correctly normalized inputs

**Implementation sketch:**
```python
class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model
        # ImageNet normalization constants (applied to /255 input)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def forward(self, image):
        image = (image - self.mean) / self.std  # Bake normalization
        return self.model.encode_image(image)
```

**Verification:**
- Run `inference_local.py` with this wrapper — Recall@10 should now match or be very
  close to the local baseline that uses CLIP's `preprocess` pipeline
- Export ONNX, compile, run on-device — this becomes our true FP32 baseline

**Use Claude Code in VS Code** to implement this change directly.

### 0.2 Record Baselines

After baking normalization, record all baselines in a tracking table:

| Configuration | Recall@10 (local) | Recall@10 (device) | Image Enc (ms) | Text Enc (ms) | Total (ms) |
|--------------|-------------------|-------------------|-----------------|----------------|------------|
| ViT-B/16 FP32 (no norm) | ? | ? | 26.3 | 4.6 | ~31 |
| ViT-B/16 FP32 (with norm) | ? | ? | ? | 4.6 | ? |
| ViT-B/16 INT8 (with norm) | ? | ? | ? | ? | ? |
| ViT-L/14 FP32 | ? | N/A | ? | ? | ? |
| ViT-L/14 INT8 | ? | ? | ? | ? | ? |

### 0.3 Study the Architecture (Ongoing, Week 1–2)

Build deep understanding of the system you're optimizing. This runs in parallel with
implementation work.

**CLIP Architecture (Days 1–3):**
- Read the CLIP paper: "Learning Transferable Visual Models From Natural Language
  Supervision" (Radford et al., 2021)
  - Focus on: contrastive learning objective (InfoNCE loss), dual-encoder architecture,
    how cosine similarity retrieval works, temperature parameter
- Read your codebase thoroughly:
  - `clip_model/clip/model.py` — understand `encode_image()` and `encode_text()`,
    the `VisionTransformer` class, `ResidualAttentionBlock`
  - `clip_model/clip/clip.py` — understand `clip.load()`, `clip.tokenize()`,
    the `_transform()` preprocessing pipeline
  - Trace the full forward pass: image → patch embedding → positional embedding →
    transformer blocks → projection → L2-normalize → embedding

**Vision Transformer Internals (Days 3–5):**
- Read the ViT paper: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
  - Focus on: how 224×224 image → 14×14 grid of 16×16 patches → 196 tokens + 1 CLS token,
    patch embedding as a convolution, learned positional embeddings,
    multi-head self-attention, layer norm placement
- Understand the dimensions for ViT-B/16:
  - 12 transformer layers, 12 attention heads, 768 hidden dim
  - Patch embedding: Conv2d(3, 768, kernel=16, stride=16) → 196 tokens
  - Each token is 768-dim, final CLS token is projected to 512-dim embedding
- Compare with ViT-L/14:
  - 24 transformer layers, 16 attention heads, 1024 hidden dim
  - Patch embedding: Conv2d(3, 1024, kernel=14, stride=14) → 256 tokens
  - Larger in every dimension: deeper, wider, more tokens

**Platform Architecture (Days 4–7):**
- Study the Qualcomm XR2 Gen 2 / Snapdragon 8 Gen 2 architecture:
  - Hexagon NPU: designed for INT8 operations, ~4× AI perf vs Gen 1
  - Kryo CPU (2 performance + 4 efficiency cores) and Adreno GPU
  - Key: FP32 models underutilize the NPU — INT8 is the native precision
- Read QAI Hub documentation:
  - Quantization examples: https://app.aihub.qualcomm.com/docs/hub/quantize_examples.html
  - Compile options and profiling
  - Understanding the compile → profile → inference pipeline
  - How to read profiling output: which layers run on NPU vs CPU fallback
- Understand the ONNX → QNN DLC pipeline:
  - ONNX export (opset 18, float32)
  - Optional quantization (post-training via QAI Hub)
  - Compile to QNN DLC (the binary format the NPU executes)
  - `--truncate_64bit_io` flag (int64 → int32 on device)

**Recommended Reading Order:**
1. CLIP paper (skim Sections 1–3, study Section 2 carefully)
2. Your `model.py` source code (hands-on, trace the forward pass)
3. ViT paper (focus on Section 3: Model)
4. QAI Hub quantization docs (practical, follow along with your model)

---

## Phase 1: Explore Model Options — No Training Needed (Days 5–10)

### 1.1 Profile ViT-L/14 on Platform

**Priority: HIGH | Effort: 1–2 days | No GPU needed**

ViT-L/14 has ~307M params (vs 86M for ViT-B/16) and produces 768-dim embeddings
(vs 512). It is significantly better at retrieval tasks — OpenCLIP's ViT-H/14 achieves
78% zero-shot ImageNet accuracy and 73.4% Recall@5 on COCO, far above ViT-B/16.

**The question: does ViT-L/14 fit under 35ms with INT8 quantization?**

Latency does NOT scale linearly with parameter count. It depends on:
- Memory bandwidth (loading weights)
- Compute intensity (matrix multiplications in attention + FFN)
- Token count: ViT-L/14 has 256 patches vs ViT-B/16's 196 (14×14 vs 16×16 grid),
  and attention is quadratic in token count
- How well operations map to the NPU's INT8 execution units

ViT-L/14 gets hit both by wider layers AND more tokens, so expect ~3–5× slower than
ViT-B/16 in FP32. But INT8 could bring it within range.

**Steps:**
1. Load ViT-L/14 via OpenCLIP: `open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')`
2. Bake normalization into the wrapper (same approach as ViT-B/16)
3. Export to ONNX
4. Submit compile job (FP32) → check latency
5. Submit quantize job (INT8) → compile → check latency
6. If under 35ms: this becomes our primary model path
7. If over 35ms: ViT-L/14 becomes our distillation teacher instead

**INT8 vs FP32 — this is a compile-time change, not a model change:**
- You export ONNX as FP32 (same as always)
- Submit a quantize job to QAI Hub with calibration data (~100–500 images)
- QAI Hub returns a quantized ONNX with INT8 weights/activations
- Compile that to DLC and profile
- No retraining, no GPU needed — it's all done on QAI Hub's cloud

### 1.2 INT8 Quantize ViT-B/16

**Priority: HIGH | Effort: 1 day | No GPU needed**

Even if we stay with ViT-B/16, INT8 quantization is valuable:
- Creates latency headroom (likely ~10–15ms image encoder instead of 26.3ms)
- This headroom can be "spent" on a larger model or kept as safety margin
- If Recall@10 drops too much with INT8, try W8A16 (weights INT8, activations INT16)

**Calibration data preparation:**
- Use ~200 images from your sample dataset
- Apply the baked-in normalization (just /255, since normalization is inside the model)
- These are fed through the model to determine quantization scale/zero-point per layer

### 1.3 Decision Point: Which Model to Fine-Tune?

After profiling, you'll have clear data to decide:

```
ViT-L/14 INT8 fits under 35ms?
├── YES → Fine-tune ViT-L/14 (Phase 2 uses ViT-L/14)
│         This is the best-case scenario — bigger model = better retrieval
│
└── NO → Fine-tune ViT-B/16 (Phase 2 uses ViT-B/16)
          Also use ViT-L/14 as distillation teacher (Phase 3)
          INT8 ViT-B/16 gives ~15ms headroom for potential model enhancements
```

---

## Phase 2: Fine-Tuning for Retrieval (Days 10–25)

### 2.1 Dataset Preparation

**Priority: HIGH | Effort: 2–3 days for download + preprocessing**

You need external data to fine-tune because:
- The 57 sample images are too few to train on (massive overfitting risk)
- The real evaluation dataset is different and likely larger
- You're optimizing for general image-to-text retrieval quality

**Primary datasets (use both):**

| Dataset | Size | Pairs | Quality | Download | Use For |
|---------|------|-------|---------|----------|---------|
| **MS-COCO Captions** | ~20 GB images | ~330K images, 5 captions each = ~1.65M pairs | High — human-written, detailed captions | `images.cocodataset.org` | Primary fine-tuning |
| **Flickr30k** | ~4 GB images | ~31K images, 5 captions each = ~158K pairs | High — human-written | `bryanplummer.com/Flickr30kEntities` | Fine-tuning + evaluation |

**Secondary dataset (if compute allows):**

| Dataset | Size | Pairs | Quality | Download | Use For |
|---------|------|-------|---------|----------|---------|
| **CC3M** | ~30 GB images | ~3M web-scraped image-text pairs | Medium — web alt-text, noisier | Use `img2dataset` tool (~1 hour download) | Additional diversity during fine-tuning |

**Download plan:**
- COCO: download train2014.zip + val2014.zip + Karpathy split annotations
- Flickr30k: request access, download images + Karpathy split
- CC3M (optional): use `img2dataset` — `pip install img2dataset`

**Data splits for training:**
- Train on: COCO train split (~113K images) + Flickr30k train split (~29K images)
- Validate on: COCO val Karpathy split (5K images) — monitor Recall@1, R@5, R@10
- Test with: your 57 sample images (sanity check, not the real metric)

### 2.2 Fine-Tuning Approach Selection

Choose based on available GPU:

| Approach | GPU Requirement | VRAM Needed | Training Time | Expected Gain |
|----------|----------------|-------------|---------------|---------------|
| **LoRA** (recommended start) | GTX 1650 (4GB) OK | ~3–4 GB | 4–8 hours | +2–5% Recall@10 |
| **Full fine-tune ViT-B/16** | GTX 1650 tight, 8GB+ better | ~6–8 GB | 1–2 days | +3–8% Recall@10 |
| **Full fine-tune ViT-L/14** | RTX 3080+ (10GB+) needed | ~12–16 GB | 2–5 days | Larger gains |
| **Knowledge distillation** | RTX 3080+ recommended | ~12–16 GB | 3–7 days | Best of both worlds |

**Recommendation:** Start with LoRA on your GTX 1650 while hunting for a better machine.
Switch to full fine-tuning once better hardware is available.

### 2.3 LoRA Fine-Tuning (Start Here — Works on GTX 1650)

LoRA inserts small low-rank matrices into attention layers. Only these small matrices
are trained (< 1% of total parameters), keeping VRAM low.

**Key training details:**
- Loss: Contrastive loss (InfoNCE) — same objective CLIP was originally trained with
- Learning rate: 1e-5 to 5e-5 (higher than full fine-tuning since only adapters train)
- Batch size: 4–8 on GTX 1650 with gradient accumulation to effective batch 32–64
- Epochs: 5–15 on COCO+Flickr30k
- LoRA rank: 8–16 (start with 8)
- Apply LoRA to: Q, K, V projection matrices in all attention layers

**After LoRA training:**
- Merge LoRA weights back into the base model (the model architecture doesn't change)
- Export to ONNX as usual — the merged model is the same size as the original
- No extra latency cost at inference

### 2.4 Full Fine-Tuning (When Better GPU Available)

If you get a machine with 8GB+ VRAM (RTX 3060/3070/3080):

- Fine-tune both image and text encoders end-to-end
- Use contrastive loss (InfoNCE)
- Learning rate: 1e-6 to 5e-6 (low to avoid catastrophic forgetting)
- Batch size: 32–128 (larger is better for contrastive learning)
- Epochs: 5–20 on COCO+Flickr30k
- Use cosine learning rate schedule with warmup
- Consider WiSE-FT: ensemble fine-tuned weights with original weights
  (α × fine-tuned + (1-α) × original) to preserve robustness

### 2.5 Data Augmentation

Since we're fine-tuning for retrieval (not classification), augmentation strategy matters:
- **Image augmentations:** Random resized crop, horizontal flip, color jitter (mild)
- **Text augmentations:** Use multiple captions per image (COCO has 5 per image)
- **Hard negative mining:** Within each batch, the hardest negatives are the most
  informative — ensure batch size is large enough to contain useful negatives

### 2.6 Evaluation During Training

Monitor these metrics on COCO Karpathy val split:
- Image-to-Text Recall@1, R@5, R@10
- Text-to-Image Recall@1, R@5, R@10 (bonus — may not be tested but good to track)
- Also periodically test on your 57 sample images for sanity

---

## Phase 3: Advanced Optimization (Days 20–40)

### 3.1 Knowledge Distillation (If ViT-L/14 Doesn't Fit)

Use ViT-L/14 (or even ViT-H/14) as teacher to improve ViT-B/16:

**Approach:**
- Teacher: ViT-L/14 pretrained (frozen, run on GPU to generate target embeddings)
- Student: ViT-B/16 (the model you'll deploy)
- Loss: MSE between teacher and student embeddings + contrastive loss
- Train on: COCO + Flickr30k + optionally CC3M subset
- The student learns to produce embeddings closer to the teacher's richer representations

**This is the best path if ViT-L/14 is too slow for on-device** — you get most of the
accuracy benefit of the larger model at the inference cost of the smaller one.

**GPU needed:** RTX 3080+ recommended (need to run both teacher and student simultaneously)

### 3.2 Quantization-Aware Training (QAT)

If post-training INT8 quantization (Phase 1) hurts Recall@10 significantly:
- Insert fake quantization nodes into the model during training
- Fine-tune for a few epochs — the model learns to be robust to INT8 rounding
- Typically recovers most of the accuracy lost from post-training quantization
- Combine with the fine-tuning from Phase 2 (do QAT as the last few epochs)

### 3.3 Compile Options Tuning

Experiment with QAI Hub compile options for potential latency improvements:
- `--force_channel_last_input` — may improve NPU memory access patterns
- `--quantize_io` — quantize input/output tensors to reduce CPU↔NPU data transfer
- Check profiling output: ensure all layers run on NPU (no CPU fallback)

---

## Phase 4: Final Integration (Days 35–45)

### 4.1 Text Encoder Optimization

The text encoder is only 4.6ms — already fast. But consider:
- **Precompute text embeddings:** if competition rules allow storing precomputed text
  embeddings, you eliminate text encoder latency entirely. Check the rules carefully.
- If text encoder must run on-device: INT8 quantize it (minimal effort, may save 1–2ms)

### 4.2 Final Quantization & Submission Pipeline

After all model changes are finalized:
1. Export to ONNX with opset 18 (baked normalization included)
2. Quantize with proper calibration data (your actual dataset images, /255 only)
3. Compile to QNN DLC for XR2 Gen 2
4. Profile — verify total latency < 35ms
5. Run inference — verify Recall@10
6. Submit via QAI Hub + submission form

### 4.3 Submission Checklist

- [ ] Normalization baked into image encoder
- [ ] Model exported to ONNX with `.onnx` + `.onnx.data` files
- [ ] Compiled to QNN DLC
- [ ] Profiled: total latency < 35ms
- [ ] Inference tested: Recall@10 recorded
- [ ] Shared model with `lowpowervision@gmail.com` on QAI Hub
- [ ] Submission form completed

---

## GPU & Hardware Plan

| Task | Hardware | Status |
|------|----------|--------|
| Normalization fix + ONNX export | Local machine (CPU OK) | Available now |
| INT8 quantization | QAI Hub cloud (no local GPU needed) | Available now |
| LoRA fine-tuning on COCO+Flickr30k | GTX 1650 (4GB VRAM) | Available now |
| Full fine-tuning ViT-B/16 | 8GB+ GPU recommended | Hunt for better machine |
| Full fine-tuning ViT-L/14 | RTX 3080+ (10GB+ VRAM) | Hunt for better machine |
| Knowledge distillation | RTX 3080+ (10GB+ VRAM) | Hunt for better machine |
| QAT training | Same as fine-tuning | Depends on Phase 2 hardware |

**Alternatives if better hardware isn't available:**
- Google Colab Pro: T4 (16GB) or A100 (40GB) — good for full fine-tuning
- Kaggle: P100 (16GB) — free tier, 30 hours/week
- Lambda Cloud / Vast.ai: rent a GPU for a few days

---

## Timeline (7 Weeks Until Deadline: April 30, 2026)

### Week 1 (March 12–18): Foundations
- [x] Record FP32 baseline latencies
- [ ] Bake normalization into ImageEncoderWrapper
- [ ] Record corrected Recall@10 baselines (local + on-device)
- [ ] Study: CLIP paper, `model.py` source code, ViT paper
- [ ] Start downloading COCO + Flickr30k datasets

### Week 2 (March 19–25): Model Exploration
- [ ] Profile ViT-L/14 (FP32 and INT8) on QAI Hub
- [ ] INT8 quantize ViT-B/16, profile and test Recall@10
- [ ] Study: QAI Hub docs, quantization concepts, ONNX pipeline
- [ ] Decision point: which model to fine-tune?
- [ ] Finish dataset download and preprocessing

### Week 3 (March 26–April 1): Fine-Tuning Round 1
- [ ] LoRA fine-tuning on GTX 1650 (COCO + Flickr30k)
- [ ] Evaluate fine-tuned model: local Recall@10
- [ ] Export, quantize, compile, test on-device
- [ ] Study: LoRA paper, CLIP-KD paper, LPCVC 2025 winning solutions
- [ ] If better GPU found: start full fine-tuning

### Week 4 (April 2–8): Fine-Tuning Round 2
- [ ] Full fine-tuning if better GPU available
- [ ] Or: refine LoRA approach (hyperparameter sweep)
- [ ] Try knowledge distillation from ViT-L/14 if applicable
- [ ] Test multiple quantization strategies (W8A8, W8A16, mixed)

### Week 5 (April 9–15): Optimization
- [ ] QAT if quantization hurt accuracy
- [ ] Compile options tuning
- [ ] Text encoder optimization (precompute or INT8)
- [ ] Multiple submission attempts — track scores on leaderboard

### Week 6 (April 16–22): Polish
- [ ] Best model selected
- [ ] Final quantization pass with optimal calibration
- [ ] End-to-end testing
- [ ] Submit best solution

### Week 7 (April 23–30): Final Push
- [ ] Any last improvements
- [ ] Final submission before deadline (April 30, 11:59 PM ET)
- [ ] Document approach for presentation

---

## Decision Tree (Updated)

```
START
  │
  ├─ Bake normalization into ImageEncoderWrapper
  ├─ Record corrected baselines (local + on-device)
  │
  ├─ Profile ViT-L/14 (FP32 → INT8) on QAI Hub
  │   ├─ Under 35ms with INT8? → Use ViT-L/14 as primary model
  │   └─ Over 35ms? → Stay with ViT-B/16, use ViT-L/14 as teacher
  │
  ├─ INT8 quantize chosen model
  │   ├─ Recall@10 acceptable? → Proceed to fine-tuning
  │   └─ Recall@10 dropped too much? → Try W8A16 or mixed precision
  │
  ├─ Fine-tune on COCO + Flickr30k
  │   ├─ LoRA first (works on GTX 1650)
  │   └─ Full fine-tune when better GPU available
  │
  ├─ Re-quantize fine-tuned model and test on-device
  │
  ├─ If ViT-B/16 chosen: try knowledge distillation from ViT-L/14
  │
  ├─ QAT if post-training quantization hurts accuracy
  │
  └─ Final: optimize text encoder + compile options + submit
```

---

## Key Resources

**Papers:**
- CLIP: https://arxiv.org/abs/2103.00020
- ViT: https://arxiv.org/abs/2010.11929
- LoRA: https://arxiv.org/abs/2106.09685
- CLIP-KD: search for "CLIP-KD" on arXiv (CVPR 2024)
- MobileCLIP: https://arxiv.org/abs/2311.17049

**Tools & Docs:**
- QAI Hub: https://app.aihub.qualcomm.com/
- OpenCLIP: https://github.com/mlfoundations/open_clip
- img2dataset (for CC3M download): https://pypi.org/project/img2dataset/

**Datasets:**
- COCO Captions: https://cocodataset.org/#download
- Flickr30k: https://bryanplummer.com/Flickr30kEntities/
- CC3M: https://ai.google.com/research/ConceptualCaptions/
- CC12M: https://github.com/google-research-datasets/conceptual-12m

**LPCVC:**
- Competition site: https://lpcv.ai/
- 2025 winning solutions: https://github.com/lpcvai/ (look for 25LPCVC repos)
- LPCVC 2025 overview: https://lpcv.ai/competitions/c2025/
