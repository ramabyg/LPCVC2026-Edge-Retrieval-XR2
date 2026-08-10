# AIMET Quantization Plan — Track B, AIMET path

**Branch:** `quantization_wt_aimet`
**Created:** 2026-08-04
**Scope:** ViT-B/16 only (ViT-L/14 is a declared dead end — 130ms+, 4x over budget, see `CLIP_Optimization_Plan_v4.md`)

## Why AIMET

Track B (`CLIP_Optimization_Plan_v4.md`) is the #1 blocker: local ORT static INT8
(`src/local/quantize.py`) gets 0.8256 local Recall@10 (vs 0.8728 FP32), but on-device the same
QDQ ONNX is **slower than FP32** (39.1ms INT8 vs 31.4ms FP32) and Recall@10 drops to 0.6804.

`plans_notes/quantization_on_XR2_platform_debug.md` already dug into why: `qai_hub.submit_quantize_job`
has **no per-op or per-layer control** — only `--range_scheme {auto,mse_minimizer,min_max}` and
beta `--lite_mp` (auto-picks sensitive layers, can't name them). The manual fix that took local
Recall@10 from 0.10 → 0.82 was excluding Softmax/LayerNorm/Gather from quantization — exactly the
kind of per-op control QAI Hub's own API doesn't expose.

AIMET is Qualcomm's own toolkit, built for QNN as the deployment target, and gives that control:
`QuantizationSimModel` (per-op-type/per-layer inclusion), `QuantAnalyzer` (per-layer sensitivity
+ MSE loss, to find which layers actually need excluding instead of guessing), `AdaRound`
(weight-rounding refinement instead of blanket exclusion), and CLE/BN-fold (mostly CNN-oriented,
uncertain value on a ViT, but cheap to try). The hope: an AIMET-produced QDQ graph is closer to
what the QNN compiler expects, avoiding the DequantizeLinear/FP16-conversion overhead identified
as the likely cause of the on-device INT8 slowdown.

`aimet-onnx` 2.34.0 and `aimet-torch` 2.34.0 are **already installed** in the `lpcvc` conda env
(`/mnt/rama_ml/conda_envs/lpcvc`, Python 3.10) — confirmed importable:
`aimet_onnx.quantsim.QuantizationSimModel`, `aimet_onnx.quant_analyzer.QuantAnalyzer`,
`aimet_onnx.batch_norm_fold.fold_all_batch_norms_to_weight`,
`aimet_onnx.cross_layer_equalization.equalize_model`, `aimet_onnx.adaround.adaround_weight.Adaround`.
No install step needed. Per AIMET's docs (`techniques/ptq.html`), the `aimet_onnx` path is
recommended over `aimet_torch` specifically because PyTorch→ONNX op splitting can cause quantizer
mismatches between simulation and on-device deployment — and we already export ONNX.

Machine: GTX 1650 4GB laptop (confirmed via `nvidia-smi`) — fine for single-batch PTQ/analysis,
not for training.

## Environment (every command)

```bash
export LPCVC_DATA_DIR=/mnt/rama_ml/data/lpcvc_track1_sample_data   # config.py default is a stale Windows path
/mnt/rama_ml/conda_envs/lpcvc/bin/python <script>                   # base python3 has no torch/aimet
```

## Steps

### 1. Branch — done

`quantization_wt_aimet`, off `main`.

### 2. FP32 ONNX baseline (reuse existing scripts — no new code) ✅ DONE

Re-exported ViT-B/16 and confirmed local FP32 number on this machine:

```
FP32 Recall@10: 0.8728 (56 images, 211 prompts)
```

✅ **Step 2 gate passed.** This matches the expected baseline (CLAUDE.md: 0.8728 ONNX local).
This is the reference number every AIMET variant below is compared against.

Note: Actual dataset is 56 images / 211 prompts (not 57/222 as some docs reference —
sample dataset may have had a few removed or the docs are stale; doesn't affect calibration logic).

### 3. AIMET PTQ pipeline — new script `src/local/quantize_aimet.py`

Same shape as `src/local/quantize.py` (reuse `ONNX_DIR`/`IMAGE_DIR`/`IMG_LIST`/`TXT_LIST` from
`src/common/config.py`, same competition-style `/255` image loading + `clip_lib.tokenize()` text
loading for calibration). Per encoder (image, text):

1. `onnx.load()` the FP32 ONNX, wrap in `aimet_onnx.quantsim.QuantizationSimModel` (confirm exact
   constructor signature against the installed 2.34.0 API at implementation time — signatures
   shift across AIMET releases).
2. `fold_all_batch_norms_to_weight` — CLIP ViT has no BatchNorm (LayerNorm only), expected no-op;
   call anyway for completeness.
3. `equalize_model` (CLE) — designed for conv+BN chains, uncertain value on a transformer. Run,
   measure, keep only if neutral-or-better vs skipping it (`--skip-cle` flag for A/B).
4. `compute_encodings(forward_pass_callback, ...)` over the calibration set (57 images / 222
   text prompts — same data `quantize.py` already uses).
5. `sim.export(...)` → QDQ ONNX, e.g. `exported_onnx/image_encoder_aimet_int8.onnx`.

Defaults per AIMET's QNN-oriented guidance: int8 weight + int8 activation, per-channel weights,
symmetric weights / asymmetric activations.

### 4. Benchmark AIMET INT8 output

Run through existing `src/local/inference_onnx.py` (`--sweep` auto-discovers `*_int8_*.onnx`, or
point at the new files directly). Compare three numbers side by side:

| Variant | Local Recall@10 | On-device Recall@10 | On-device latency |
|---|---|---|---|
| FP32 (step 2) | ~0.8728 | 0.7299 | 31.4ms |
| ORT static INT8 (existing, `quantize.py`) | 0.8256 | 0.6804 | 39.1ms ❌ slower than FP32 |
| AIMET INT8 (this plan) | TBD | TBD (step 7) | TBD (step 7) |

### 5. QuantAnalyzer sensitivity pass — new script `src/local/quant_analyzer_aimet.py`

`aimet_onnx.quant_analyzer.QuantAnalyzer(model, dummy_input, forward_pass_callback, eval_callback)`:
- `forward_pass_callback`: all 57 images / 222 texts (small enough to use the full calibration set).
- `eval_callback`: wraps `evaluate_track1` from `src/common/eval.py` → Recall@10 as the accuracy metric.
- Runs per-layer sensitivity, per-layer enable/disable sweep, min-max encoding export, per-layer
  MSE loss. Skip PDF histogram generation initially (heavier, GPU-memory sensitive on 4GB) —
  enable later if the rest runs fine.
- Output to `quant_analysis/image_encoder/` and `quant_analysis/text_encoder/` (new, add to
  `.gitignore` alongside `exported_onnx/`/`diagnostics/` if those are already ignored).

Use this to check which ops are actually sensitive (expect Softmax/attention/post-LN activations
and the text encoder's EOS `Gather` — `export_onnx.py:96`, `x[arange(x.shape[0]), eos_index]` —
already flagged in `quantization_on_XR2_platform_debug.md` as historically fragile) rather than
guessing/excluding by op-type as `quantize.py` currently does.

**This step is also the actual "does this make sense on this laptop" test** — if full per-layer
analysis with histograms OOMs or is too slow on 4GB, fall back to CPU (fine at batch size 1) or a
reduced sample count, and note the finding here.

### 6. Optional stretch: AdaRound

Only if step 5 flags specific weight-sensitive layers and step 4's plain QuantSim result is
close-but-not-quite competitive with FP32. `aimet_onnx.adaround.adaround_weight.Adaround` on the
FP32 model before building QuantSim, then re-export and re-benchmark. Skip entirely if QuantSim
alone is already close to FP32 or is already clearly bad (won't be saved by AdaRound).

### 7. On-device validation (existing infra, no new code)

If an AIMET variant beats 0.8256 local / looks structurally cleaner, push its QDQ ONNX through
the existing `src/platform/compile_and_profile.py` / `run_on_device.py` to get real XR2 Gen 2
latency + Recall@10 — this is what actually determines whether AIMET fixes the on-device
INT8-slower-than-FP32 problem. No code changes needed, just point the existing pipeline at the
new file.

## Decision gates

- Step 2 doesn't reproduce ~0.8728 → stop, fix env/paths before going further (something's wrong
  with this machine's setup, not with AIMET).
- Step 4 AIMET INT8 local Recall@10 ≤ 0.8256 (existing ORT result) and no on-device latency
  win in step 7 → AIMET path doesn't beat what we already have; fall back to Track B's B1/B2
  (QAI Hub compile-time INT8 / FP16-native) instead.
- Step 4 AIMET INT8 ≥ 0.85 local AND step 7 latency < 35ms → adopt as new INT8 default, update
  `CLAUDE.md` measured baselines table.
- Step 5 QuantAnalyzer infeasible on 4GB GPU even after reductions → fall back to CPU-only
  analysis (slower but workable at batch size 1); note this limits future GPU-side analysis to
  the GPU server, not this laptop.

## Files to create

- `src/local/quantize_aimet.py`
- `src/local/quant_analyzer_aimet.py`

No changes needed to `src/common/eval.py`, `src/common/config.py`, `src/local/inference_onnx.py`,
or `src/platform/export_onnx.py` — reused as-is.
