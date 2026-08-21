# LPCVC 2026 Track 1 — Image-to-Text Retrieval on Snapdragon

Fork of the [LPCVC 2026 Track 1 sample solution](https://github.com/lpcvai/26LPCVC_Track1_Sample_Solution),
extended into a full on-device optimization study: export OpenAI CLIP to ONNX, compile
and profile it on Qualcomm AI Hub, and push Recall@10 as high as possible inside a hard
latency budget.

**Task.** For each image, check whether its ground-truth caption appears in the top-10
most similar texts by cosine similarity (Recall@10).

| Constraint | Value |
|---|---|
| Latency budget | **≤ 35 ms** combined (image + text encoder) — hard threshold |
| Ranking metric | Recall@10, scored only if latency passes |
| Input | `float32 (1, 3, 224, 224)`, pre-resized, divided by 255 only |
| Normalization | **Not** applied by the competition — must be baked into the model |
| Target | Snapdragon, Hexagon NPU (INT8-oriented) |

> **Device note.** The competition targets XR2 Gen 2, but that device was deprecated on
> AI Hub in July 2026. All measurements in this repo are on **Samsung Galaxy S22 (Family)**
> — Snapdragon 8 Gen 1, Hexagon v69 — which LPCVC now recommends. Override with
> `--device` or `QAI_HUB_DEVICE`.

---

## Status — what actually ships

**FP32 is the shipping configuration.** It meets the budget with 28% headroom, and every
INT8 build measured so far is *slower* than FP32 on this hardware.

| Build | Recall@10 | Image | Text | Total | Verdict |
|---|---|---|---|---|---|
| PyTorch FP32 (local reference) | 0.8805 | — | — | — | reference |
| ONNX FP32 (local ORT) | 0.8728 | — | — | — | resize-path gap |
| **FP32 on device** | **0.8805** | 20.16 ms | 4.99 ms | **25.14 ms** | ✅ **PASS** |
| INT8 AIMET `addsoftmax`/`bestscope` | 0.8756 | 44.23 ms | 7.37 ms | 51.60 ms | ❌ over budget |
| INT8 AIMET `addsoftmax`/`defscope` | 0.8637 | 44.31 ms | 10.63 ms | 54.95 ms | ❌ over budget |
| INT8 AIMET `defscope`/`bestscope` | 0.8704 | 55.65 ms | 7.48 ms | 63.12 ms | ❌ over budget |
| INT8 AIMET `defscope`/`defscope` | 0.8639 | 55.86 ms | 10.67 ms | 66.53 ms | ❌ over budget |

Raw logs: `profile_logs/on_device_fp32_20260812_095911.txt`,
`profile_logs/aimet_pair_sweep_20260812_131320.txt`.

The 57-image sample set makes one image worth ~0.018 Recall@10, so differences below
~0.02 are not meaningful.

### Quantization: accuracy was recoverable, latency was not

Accuracy history — INT8 went from near-random to within 0.005 of FP32 by restricting
*which ops* get quantized:

| Attempt | Recall@10 |
|---|---|
| ORT QDQ, all ops | 0.0527 (near-random) |
| ORT QOperator, all ops | 0.1003 |
| ORT QOperator, Conv/MatMul/Gemm only | 0.8256 |
| AIMET scope-restricted, local | 0.8610 |
| AIMET scope-restricted, on device | **0.8756** |

But the same scope restriction that saves accuracy destroys latency. Three findings,
each reproducible with the diagnostics below:

1. **The scope-restricted builds are not really INT8.** In a QDQ graph the activation
   quantizer belongs to the *producing* op. Restricting the scope to
   `Conv/ConvTranspose/MatMul/Gemm` strips the output quantizers from LayerNorm, Add,
   Mul and Softmax — which are exactly the tensors feeding the matmuls. Result: **50 of
   74 GEMM-class ops in the image encoder run on float activations** with dequantized
   weights. That is FP32 compute *plus* ~240 Q/DQ pairs of overhead, hence 56 ms against
   FP32's 20 ms.
2. **A mixed float/int8 graph cannot be composed from a DLC on the HTP.** The compile job
   succeeds and then every profile and inference job fails with
   `QnnModel_composeGraphsFromDlc: MODEL_GRAPH_ERROR`. Fix: compile these builds to a QNN
   **context binary** (`--aimet-runtime`, now the default for `int8-aimet`). Per-channel
   MatMul weights were investigated and ruled out.
3. **Putting LayerNorm back in scope collapses accuracy** (0.0601) because of ViT massive
   activations: LN outputs span a range of 51–110 while every other tensor sits near 11,
   so per-tensor INT8 crushes them. This is a dynamic-range problem, not a scope problem —
   no permutation of the op allow-list fixes it. W8A16 is the open lead.

Detail: `plans_notes/aimet_quantization_review_2026-08-04.md`.

---

## Repository layout

```
src/
  common/      config.py (all paths + device + budget), eval.py (evaluate_track1), calibration.py
  platform/    QAI Hub side — export_onnx, compile_and_profile, upload_dataset,
               run_on_device (compile→profile→infer→Recall@10), sweep_on_device
  local/       local iteration — inference_pytorch, inference_onnx,
               quantize.py (ONNX Runtime PTQ), quantize_aimet.py (AIMET PTQ),
               fix_qdq_for_htp.py, analysis/, train/ (LoRA fine-tune + merge)
  debug/       stage_runner, diagnose_device_gap, evaluate_coco1k
diagnostics/   graph-precision tooling (git-ignored output)
plans_notes/   design docs and experiment reviews
profile_logs/  on-device result tables
exported_onnx/ ONNX artifacts (.onnx + .onnx.data must stay together)
```

## Setup

```bash
conda env create -f environment.yml     # creates the `lpcvc` env (AIMET, clip, qai_hub)
conda activate lpcvc
qai-hub configure --api_token <YOUR_TOKEN>
```

`src/common/config.py` still defaults `SAMPLE_DATA_DIR` to a Windows path, so on Linux
point it at the dataset explicitly:

```bash
export LPCVC_DATA_DIR=/path/to/lpcvc_track1_sample_data   # images/, img_list.csv, txt_list.csv
```

Other overrides: `QAI_HUB_DEVICE`, `QAI_HUB_IMAGE_DATASET`, `QAI_HUB_TEXT_DATASET`,
`LPCVC_ONNX_DIR`, `LPCVC_CALIB_SOURCE`, `LPCVC_LATENCY_BUDGET_MS`.

> Quantization builds peak at 3–5 GB RSS. Pass `--encoder image` or `--encoder text`
> separately rather than quantizing both in one process.

## Usage

### Fast local loop (no AI Hub)

```bash
python src/local/inference_pytorch.py                  # FP32 reference Recall@10
python src/local/inference_onnx.py --mode all          # FP32 vs INT8 ONNX, all combinations
python src/local/inference_onnx.py --inspect-embeddings   # dtype/range/norm per encoder
```

### Full device pipeline

```bash
# 1. export encoders to ONNX (CLIP normalization baked into the image wrapper)
python src/platform/export_onnx.py --encoder image
python src/platform/export_onnx.py --encoder text

# 2. upload the evaluation datasets — prints the dataset IDs
python src/platform/upload_dataset.py

# 3. compile + profile + infer + score, in one job graph
python src/platform/run_on_device.py --precision fp32
```

Update `DEFAULT_IMAGE_DATASET_ID` / `DEFAULT_TEXT_DATASET_ID` in `src/common/config.py`
after each upload. Results are written to `profile_logs/`.

`run_on_device.py --precision` accepts:

| Mode | What it does |
|---|---|
| `fp32` | default; the shipping configuration |
| `fp16` | native FP16 on the Hexagon HTP |
| `int8-compile` | AI Hub quantizes at compile time |
| `int8-hub` / `w8a16` | AI Hub quantize job (W8A8 / W8A16), then compile |
| `int8-local` | locally quantized QDQ ONNX |
| `int8-aimet` | AIMET scope-restricted QDQ, tag-selected per encoder |

```bash
# validate artifacts, datasets and device without submitting any job
python src/platform/run_on_device.py --precision int8-aimet --sweep-pairs --dry-run

# all four AIMET scope pairs in one comparison table (24 jobs)
python src/platform/run_on_device.py --precision int8-aimet --sweep-pairs
```

### Quantization

```bash
# AIMET PTQ, one encoder at a time; --variant names the output artifact
python src/local/quantize_aimet.py --encoder image --variant defscope
python src/local/quantize_aimet.py --encoder image --variant addsoftmax \
    --quant-op-types Conv,ConvTranspose,MatMul,Gemm,Softmax

# ONNX Runtime PTQ (QOperator default; use --format qdq for hardware export)
python src/local/quantize.py
```

Scope tags used in `PAIR_MATRIX`: `defscope` = Conv/ConvTranspose/MatMul/Gemm;
`addsoftmax` = defscope + Softmax (image); `bestscope` = defscope + LayerNormalization,
Softmax (text). LayerNormalization must stay float in the **image** encoder.

### Understanding a quantized graph

```bash
# per-op verdict: does this op actually run in int8, or in float?
python diagnostics/trace_precision.py exported_onnx/image_encoder_int8_aimet_defscope.onnx

# cut one transformer block out of the 346 MB model so Netron opens instantly
python diagnostics/slice_one_block.py exported_onnx/image_encoder_int8_aimet_defscope.onnx
netron diagnostics/image_encoder_int8_aimet_defscope_block2.onnx
```

Reading rule: every edge in a QDQ ONNX is float32 **except** the edge between a
`QuantizeLinear` and its `DequantizeLinear`. An op runs in int8 only if its activation
input comes from a `DequantizeLinear`.

### LoRA fine-tuning

```bash
torchrun --nproc_per_node=4 src/local/train/finetune_lora.py \
    --datasets both --epochs 15 --lora-r 16 --batch-size 128
python src/local/train/merge_lora.py    # fold the adapter back into the encoder
```

Last run (COCO + Flickr30k, 1000-image val pool): Recall@10 96.6 → 98.2, Recall@1
67.1 → 70.9; 2.95M trainable params (1.93% of the model), 11.8 MB adapter. Note this is
a different dataset and pool from the 57-image LPCVC sample score above — the two
numbers are not comparable, and this checkpoint has not been exported or run on device.

---

## Gotchas

- **`.onnx.data` files hold the weights.** Keep them beside their `.onnx`; the model
  references them by relative path.
- **Normalization is baked into `ImageEncoderWrapper`**, so `upload_dataset.py` must send
  `/255` images only. Adding CLIP mean/std there would double-normalize.
- **Re-export and re-compile after any model change** — a previously compiled DLC does not
  pick it up.
- **`int8-aimet` must use a context binary**, not a DLC (finding 2 above). Forcing
  `--aimet-runtime qnn_dlc` reproduces the `MODEL_GRAPH_ERROR`.
- **QDQ INT8 ONNX files are not smaller than FP32** (330 MB vs 330 MB) — they store float
  weights plus quantize nodes. Size reduction only appears in the compiled artifact.

## Submissions

See the [AI Hub guide](https://github.com/lpcvai/25LPCVC_AIHub_Guide) for how the
competition runs submitted models.
