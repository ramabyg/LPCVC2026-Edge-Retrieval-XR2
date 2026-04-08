# Command Notes & Results

## Initial Recall Scores (FP32 Baseline)

### Before Local CLIP Model
- Recall@10: **0.7260**

### After Exporting Local Model & Compiling
- Recall@10: **0.7260** (identical performance)

### Local Inference (No Device)
- Recall@10: **0.8805**

---

## 2026-03-13 — Issue Investigation

⚠️ **Observed Discrepancy:**
- On-device Recall@10: 0.7299
- Local Inference Recall@10: 0.8805
- **Root cause:** CLIP normalization moved but still investigating differences.

---

## INT8 Quantization Results — ViT-B/16

### Job IDs

| Component | Quantize | Compile | Profile |
|-----------|----------|---------|---------|
| **Image Encoder** | jpev34mv5 | jgz7kv3xp | jpx12eklg |
| **Text Encoder** | j5q2o9qo5 | j5w9nmemp | j5mzyvn9p |

### Update `inference.py`
```python
image_compiled_id = "jgz7kv3xp"
text_compiled_id  = "j5w9nmemp"
```

### Performance Metrics

| Metric | Image Encoder | Text Encoder |
|--------|---------------|--------------|
| **PSNR** | 28.2 dB | 27.2 dB |
| **Inference Time** | 16.6 ms | 5.1 ms |
| **Peak Memory** | 0–497 MB | 0–160 MB |

### Final Recall@10
- **0.0327** (significant loss after INT8 quantization)

### Dataset and Compile IDs after correctsion:
```python
First image shape: (1, 3, 224, 224)
First 3 filenames: ['1127792001_9b9b950f20_o.jpg', '1157182238_992e41a670_o.jpg', '12736230865_e67caaeef2_o.jpg']
Uploading dataset: 10.1MB [00:05, 2.06MB/s]
Dataset(id='d7x5nnzz9', name='h5-dataset', expiration_time='2026-04-16 05:17:49')
(1, 77)
int32
Uploading dataset: 634kB [00:00, 650kB/s]
Dataset(id='d2qe11q82', name='h5-dataset', expiration_time='2026-04-16 05:17:56')

Image compilation job ID: j5mwn7ewp
Text compilation job ID: j5q76n04g
```
- **0.8805059523809524** (After arranging the dataset correctly)

### ViT-L/14 tests:

- local run bench mark recal score: 0.90
- On platform with fp32 recal score: 0.74285
Analysis:
ViT-L/14 On-Device vs Local Recall Gap — Diagnosis & Fix

 Context

 ViT-L/14 shows 0.90 Recall@10 locally (PyTorch) but only 0.74 on-device (QAI Hub XR2 Gen 2).
 The gap is ~0.16 — far larger than the typical ONNX/PyTorch drift seen for ViT-B/16 (~0.008).
 Goal: identify which pipeline layer introduces the gap and fix it.

 ---
 Root Cause Hypotheses (ranked)

 H1: ONNX export quality (dynamo=True for ViT-L/14) — most likely

 - dynamo=True traces the model graph; larger models can have tracing issues
 - ViT-L/14 has 24 transformer layers, 1024 hidden dim vs 12 layers / 768 for ViT-B/16
 - CLAUDE.md explicitly notes: "dynamo=True exports may use non-standard patterns"
 - Diagnosis: if local ONNX FP32 recall ≈ 0.74 → this is the culprit

 H2: QNN compilation / FP32 precision on XR2 Gen 2

 - Some FP32 ops may run in lower precision or fall back to CPU on QNN for large models
 - QNN compile logs may show op-level warnings/fallbacks for ViT-L/14
 - Diagnosis: if local ONNX FP32 recall ≈ 0.90 → QNN is the culprit

 H3: Image preprocessing mismatch (lower likelihood)

 - upload_dataset.py uses Resize(224, BICUBIC) + CenterCrop(224) (torchvision)
 - inference_onnx_local.py uses PIL resize((224, 224)) — bilinear, no center crop
 - This preprocessing mismatch exists for both models, so it's not ViT-L/14-specific
 - Wouldn't explain a ViT-L/14-specific regression

 ---
 Diagnostic Steps (must run before fixing)

 Step 1 — ONNX FP32 local baseline for ViT-L/14

 python inference_onnx_local.py --model ViT-L/14 --mode fp32 --inspect-embeddings
 Interpret result:
 - Recall ≈ 0.74 → gap exists at ONNX level → fix ONNX export (→ H1)
 - Recall ≈ 0.90 → ONNX is fine → fix QNN compilation (→ H2)

 Step 2 — Check QAI Hub compile job logs

 - Visit QAI Hub console for the ViT-L/14 compile jobs
 - Look for: fallback ops, precision downcasting warnings, unsupported op messages

 ---
 Fix Path A: ONNX Export Issue (if Step 1 gives ~0.74)

 Option A1: Switch to dynamo=False for ViT-L/14
 - File: export_onnx.py
 - Change dynamo=True → dynamo=False for the ViT-L/14 export path
 - dynamo=False uses TorchScript tracing — more mature for transformers
 - Risk: may need torch.no_grad() context and explicit strict=True/False tuning

 Option A2: Verify TextEncoderWrapper argmax tracing
 - eos_index = token_ids.argmax(dim=-1) with int64 input may trace unexpectedly
 - Verify the EOS token selection produces correct indices in the ONNX graph
 - Use onnx.helper or Netron to inspect the exported graph node for this op

 ---
 Fix Path B: QNN Compilation Issue (if Step 1 gives ~0.90)

 Option B1: Add compile options to preserve FP32 precision
 - File: compile_and_profile.py
 - Add --force_channel_last_output false or precision flags if QAI Hub supports them
 - Check QAI Hub docs for FP32-preservation flags for QNN DLC

 Option B2: Check QAI Hub logs for specific failing ops
 - If certain ops fall back to CPU with wrong layout, they degrade accuracy
 - May need to restructure ONNX graph to avoid non-accelerated ops

 ---
## Assuming issue with FP32->FP16 automatic conversion, Plan is to check how it behaves with quantization enabled. With local and on platform

   Step 1 — Quantize ViT-L/14 with QDQ (for QAI Hub)                                                                                                                                                                                                                                                        python quantize_local.py --model ViT-L/14 --format qdq --activation qint8                                                                                                                                                                                                                              
  This produces exported_onnx/image_encoder_vitl14_int8.onnx and text_encoder_vitl14_int8.onnx.                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                           Step 2 — Verify INT8 quality locally                                                                                                                                                                                                                                                                     python inference_onnx_local.py --model ViT-L/14 --mode int8 --inspect-embeddings                                                                                                                                                                                                                       
  Target: ≥ 0.85 recall (ViT-L/14 INT8 should beat ViT-B/16 INT8 at 0.8256).                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                         
  Step 3 — Compile INT8 to QNN DLC                                                                                                                                                                                                                                                                       
  python compile_and_profile.py --model ViT-L/14 --int8                                                                                                                                                                                                                                                    Update the compile IDs in inference.py, then run on-device.   

  ### Bench Marks with ViT-L/14 for local inference for 56 sample data set
  | Name| Recall!10 Score | Notes |
  |-----|-------|-------|
  | **Plane** | 0.9003 | Just loaded model as is|
  | **ONNX Exported** |  0.8857 | Exported and run on FP32 mode|

  | Config | Recall@10 |
  |--------|-----------|
  | FP32 | 0.8857 |
  | INT8 | 0.8567 |
  | FP32_img + INT8_txt | 0.8656 |
  | INT8_img + FP32_txt | 0.8893 |
  | Delta (INT8 vs FP32) | -0.0290 |

### Bench Marks with ViT-L/14 on platform for sample data set
| Config | Recall@10 | Notes |
|--------|-----------|-------|
| quantize_and_compile.py | 0.026785714285714284 | Quantization Done on Platform. destroying Softmax/LayerNorm|
| run_on_device.py  | 0.7262 | local quantization, qdq, compile and running. Inference time: 13.2ms and 434MB peak memory for Text, 133.0ms and 1369MB peak memory for Image |

### New file with combining compile, profile, inference
#### ViT-B/16 FP32                                                                                                                                                                                                                               python run_on_device.py
                                                                                                                                                                                                                                              
#### ViT-L/14 FP32
  python run_on_device.py --model ViT-L/14
#### ViT-B/16 INT8 (need to run quantize_local.py --format qdq first)
  python run_on_device.py --int8
#### ViT-L/14 INT8
  python run_on_device.py --model ViT-L/14 --int8
#### With updated datasets
  python run_on_device.py --image-dataset-id dXXX --text-dataset-id dXXX

---

## 2026-03-24 — Local Quantization Benchmark (all combos)

Run: `python quantize_local.py` → `python inference_onnx_local.py --sweep`

| Model    | Format     | Activation | Recall@10 | vs FP32  |
|----------|------------|------------|-----------|----------|
| ViT-B/16 | FP32       | —          | 0.8728    | —        |
| ViT-B/16 | qdq        | qint8      | 0.8256    | -0.0472  |
| ViT-B/16 | qdq        | quint8     | 0.8256    | -0.0472  |
| ViT-B/16 | qoperator  | qint8      | 0.8250    | -0.0478  |
| ViT-B/16 | qoperator  | quint8     | 0.8256    | -0.0472  |
| ViT-L/14 | FP32       | —          | 0.8857    | —        |
| ViT-L/14 | qdq        | qint8      | 0.8567    | -0.0290  |
| ViT-L/14 | qdq        | quint8     | 0.8567    | -0.0290  |
| ViT-L/14 | qoperator  | qint8      | 0.8649    | -0.0208  |
| ViT-L/14 | qoperator  | quint8     | 0.8567    | -0.0290  |

**Key observations:**
- ViT-B/16: all INT8 combos identical (~0.8256) — format/activation has no local impact
- ViT-L/14: qoperator+qint8 is best (0.8649), ~0.02 better than qdq variants
- ViT-L/14 INT8 (0.8567–0.8649) beats ViT-B/16 INT8 (0.8256) by ~0.03–0.04
- For QAI Hub (on-device): use qdq format; qoperator is local-only

---

## 2026-03-25 — On-Device Sweep Results (XR2 Gen 2)

Run: `python sweep_on_device.py`

| Model    | Format    | Activation | Img(ms) | Txt(ms) | Total(ms) | ImgMem(MiB) | TxtMem(MiB) | Recall@10 | vs FP32  |
|----------|-----------|------------|---------|---------|-----------|-------------|-------------|-----------|----------|
| ViT-B/16 | fp32      | —          | 26.8    | 4.6     | 31.4      | 227         | 132         | 0.7299    | —        |
| ViT-L/14 | fp32      | —          | 121.6   | 8.9     | 130.4     | 886         | 153         | 0.7429    | —        |
| ViT-B/16 | qdq       | qint8      | 32.5    | 6.8     | 39.3      | 421         | 210         | 0.0433    | -0.6867  |
| ViT-B/16 | qdq       | quint8     | 32.2    | 6.8     | 39.1      | 355         | 205         | 0.6804    | -0.0496  |
| ViT-B/16 | qoperator | qint8      | FAIL    | FAIL    | —         | n/a         | n/a         | FAIL      | —        |
| ViT-B/16 | qoperator | quint8     | FAIL    | FAIL    | —         | n/a         | n/a         | FAIL      | —        |
| ViT-L/14 | qdq       | qint8      | 131.7   | 13.2    | 144.9     | 394         | 221         | 0.0625    | -0.6804  |
| ViT-L/14 | qdq       | quint8     | 134.0   | 13.0    | 146.9     | 461         | 219         | 0.7262    | -0.0167  |
| ViT-L/14 | qoperator | qint8      | FAIL    | FAIL    | —         | n/a         | n/a         | FAIL      | —        |
| ViT-L/14 | qoperator | quint8     | FAIL    | FAIL    | —         | n/a         | n/a         | FAIL      | —        |

### Analysis (from profile log review)

**Why INT8 is SLOWER than FP32 (32.5ms vs 26.8ms):**
The QDQ compiler inserts explicit `DequantizeLinear` and dtype conversion nodes throughout the graph:
- `QNN_DATATYPE_FLOAT_32 → FLOAT_16` conversions at input
- `DequantizeLinear` nodes scattered after quantized ops
- These add ~4-6ms overhead — all layers still run on NPU, zero CPU fallback

**Why qint8 Recall collapsed (0.04):**
Softmax and LayerNorm are being re-quantized by the QAI Hub compiler on-device despite being
excluded in local ONNX. Signed int8 (±127) destroys attention score distributions.
quint8 (0–255) preserves more dynamic range → 0.68 recall.

**Why memory INCREASED with INT8 (421 MiB vs 227 MiB FP32):**
QDQ format keeps both quantized and dequantized tensor buffers simultaneously in SRAM.
Compiler does not fuse/eliminate intermediate buffers.

**Why QOperator FAILS on-device:**
QAI Hub compiler does not support QOperator format — only QDQ is accepted.

**ViT-L/14 verdict:** 121ms+ at any precision, 4.5× over 35ms budget. Abandoned for on-device.

# Rerun as per Optimization Plan V3
```
Model:        ViT-B/16  FP32
Recall@10:    0.8805
Image compile job: jgzx74nk5
Text  compile job: jp27mvqr5
Image profile job: jpr4ynz7g
Text  profile job: jpy4d79lp
Img Inference Job ID: jp8374log
Txt Inference Job ID: jpy4d7elp
```
```
==================================================
Recall@10 Results — ViT-B/16 ONNX, local , FP32
==================================================
  FP32                       0.8728
```

# Fine Tuning and Merging Plan
Plan: Load Fine-Tuned LoRA Weights into CLIP Pipeline

 Context

 Fine-tuned LoRA weights are saved in PEFT format at checkpoints/lora_checkpoints_best/best/ (adapter only, ~5.9 MB). The training achieved    
 97.20 Recall@10 on the validation set.

 The current inference_pytorch.py and export_onnx.py both load the base CLIP model only — neither loads the fine-tuned weights. To use the     
 improved model end-to-end (local test → ONNX export → on-device), we need to:
 1. Merge LoRA weights into the base model (one-time step, already scripted)
 2. Add a --weights flag to inference_pytorch.py and export_onnx.py to load the merged checkpoint

 Merging (not just loading adapter) is required because:
 - ONNX export cannot trace through the PEFT wrapper cleanly
 - Merged weights are a plain CLIP state dict — drop-in for load_state_dict()

 ---
 Step 1: Merge LoRA Weights

 Run the existing merge_lora.py script to bake LoRA adapter weights into base CLIP:

 cd C:\rama\projects\LPCVC2026-Edge-Retrieval-XR2\.claude\worktrees\condescending-mcnulty
 python src/local/train/merge_lora.py \
   --checkpoint checkpoints/lora_checkpoints_best/best \
   --output checkpoints/merged_best.pt

 Output: checkpoints/merged_best.pt — a plain PyTorch state dict with LoRA weights baked in.

 ---
 Step 2: Modify inference_pytorch.py

 File: src/local/inference_pytorch.py

 Add a --weights CLI argument. When provided, load the merged state dict on top of the base model.

 Change (in argparse section):
 parser.add_argument("--weights", type=str, default=None,
                     help="Path to merged fine-tuned weights (.pt file)")

 Change (in model loading section, after clip_lib.load()):
 model, _ = clip_lib.load(MODEL, device=device)
 model = model.float()

 if args.weights:
     state = torch.load(args.weights, map_location=device)
     model.load_state_dict(state, strict=False)
     print(f"Loaded fine-tuned weights from: {args.weights}")

 model.eval()

 Usage:
 python src/local/inference_pytorch.py --weights checkpoints/merged_best.pt

 ---
 Step 3: Modify export_onnx.py

 File: src/platform/export_onnx.py

 Same pattern — add --weights flag to load merged checkpoint before wrapping and exporting.

 Change (in argparse section):
 parser.add_argument("--weights", type=str, default=None,
                     help="Path to merged fine-tuned weights (.pt file)")

 Change (in model loading section, after clip_lib.load()):
 clip_model, _ = clip_lib.load(args.model, device=device)
 clip_model = clip_model.to(torch.float32)

 if args.weights:
     state = torch.load(args.weights, map_location=device)
     clip_model.load_state_dict(state, strict=False)
     print(f"Loaded fine-tuned weights from: {args.weights}")

 clip_model.eval()

 Usage:
 python src/platform/export_onnx.py --weights checkpoints/merged_best.pt

 ---
 Full Workflow After Implementation

 Phase 1: Local validation (57 samples)

 # Merge LoRA into base model
 python src/local/train/merge_lora.py \
   --checkpoint checkpoints/lora_checkpoints_best/best \
   --output checkpoints/merged_best.pt

 # Test on 57-sample dataset
 python src/local/inference_pytorch.py --weights checkpoints/merged_best.pt
 Expected: Recall@10 near 97.20 (may be lower since sample set ≠ training val split, but should exceed 0.8805 baseline).

 Phase 2: ONNX export with fine-tuned weights

 python src/platform/export_onnx.py --weights checkpoints/merged_best.pt
 Output: exported_onnx/image_encoder.onnx + text_encoder.onnx with fine-tuned weights.

 Phase 3: Compile and profile on-device

 python src/platform/compile_and_profile.py   # verify still under 35ms
 python src/platform/upload_dataset.py         # get new dataset IDs
 python src/platform/run_on_device.py          # get on-device Recall@10

 ---
 Critical Files

 ┌─────────────────────────────────────────┬────────────────────────────────────────────┐
 │                  File                   │                   Change                   │
 ├─────────────────────────────────────────┼────────────────────────────────────────────┤
 │ src/local/inference_pytorch.py          │ Add --weights arg + load_state_dict() call │
 ├─────────────────────────────────────────┼────────────────────────────────────────────┤
 │ src/platform/export_onnx.py             │ Add --weights arg + load_state_dict() call │
 ├─────────────────────────────────────────┼────────────────────────────────────────────┤
 │ src/local/train/merge_lora.py           │ Run as-is (no changes needed)              │
 ├─────────────────────────────────────────┼────────────────────────────────────────────┤
 │ checkpoints/lora_checkpoints_best/best/ │ Source of LoRA adapter weights             │
 └─────────────────────────────────────────┴────────────────────────────────────────────┘

 Verification

 1. merge_lora.py prints cosine similarity between base and merged embeddings — should be close to 1.0 but not identical
 2. inference_pytorch.py prints Recall@10 — should be noticeably above 0.8805
 3. ONNX export completes without shape errors
 4. compile_and_profile.py shows latency still ≤ 35ms (LoRA adds no new ops — weights just merged)
 

 # 04/07 After Fine Tuning, Sample dataset recall score on platform
 ### Accuracy got improved to 0.5458839 from 0.49 after fine tuning
 ```
 ==================================================
Model:        ViT-B/16  FP32
Recall@10:    0.8909
Image compile job: jp27vvdq5
Text  compile job: j563dd0y5
Image profile job: jp34wwrng
Text  profile job: jpv199nrp
Edge Alchemist	4/7/2026 15:10:19	****vvdq5	****dd0y5	0.5458839792	31318	26750	4568
==================================================
```
