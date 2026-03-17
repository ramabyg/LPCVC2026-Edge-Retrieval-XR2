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