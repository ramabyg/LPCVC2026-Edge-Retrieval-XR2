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

|===================================================| 
| Recall@10 Results|
| ==================================================|

  FP32                       0.8857

  INT8                       0.8567

  FP32_img+INT8_txt          0.8656

  INT8_img+FP32_txt          0.8893

  Delta (INT8 vs FP32)       -0.0290

  
