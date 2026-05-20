Your premise is wrong (in a useful way)

  You CAN compare INT8 vs FP32. A quantized op's output is just scale * (q − zero_point) — it lives in float space the moment you read it. The right metric isn't bit-equality, it's per-tensor signal quality:

  - Cosine similarity (your final embedding metric — most relevant)
  - SQNR = 20 * log10(||fp32|| / ||fp32 − int8||) — a 30+ dB layer is fine, <20 dB is suspect, <10 dB is broken
  - Max abs error / mean abs error

  So the debug question becomes: at which boundary in the QCom pipeline does cosine_sim(fp32_embedding, intX_embedding) collapse?

  The pipeline gives you 3 inspectable artifacts

  quantize_and_compile() in run_on_device.py:159 produces three downloadable models. Run all three locally with onnxruntime and compute Recall@10 on each — that isolates the failure to one stage:

  ┌───────┬──────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────┐
  │ Stage │                                         Artifact                                         │                       What it tells you if Recall@10 is bad                        │
  ├───────┼──────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ 0     │ Raw exported ONNX (exported_onnx/image_encoder.onnx)                                     │ Export bug (norm baking, dynamo opset)                                             │
  ├───────┼──────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ 1     │ optimize_job.get_target_model() — Hub's optimized fp32 ONNX                              │ Hub's LN/Attention/GELU fusion broke a subgraph                                    │
  ├───────┼──────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ 2     │ quant_job.get_target_model() — the quantized ONNX, runnable locally on CPU, outputs fp32 │ Quantization itself is the problem — calibration quality, op coverage, granularity │
  ├───────┼──────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ 3     │ Compiled DLC, run via QAI Hub inference job                                              │ Stage 2 was fine but the QNN runtime/compiler diverged                             │
  └───────┴──────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

  Most likely culprit by your own data: Stage 2. Your local QOperator attempt with all ops quantized got 0.10; excluding LN/Softmax/Gather got 0.82 (CLAUDE.md). The QAI Hub quantizer almost certainly quantizes everything, reproducing the 0.10-class failure. Layer scope is the prime suspect,
  not the QCom flow.

  Suggested debug order (smallest first)

  1. Verify Stage 0 reproduces: load each artifact in onnxruntime CPU, compute Recall@10. ~30 min, no Hub credits.
  2. If Stage 1 ≈ FP32 but Stage 2 collapses → per-layer SQNR sweep: add intermediate tensors as outputs in both fp32 and quantized ONNXs, run a few calibration samples through both, log SQNR per layer. The worst 2-3 layers will pop out — usually softmax outputs, post-LN activations, or the
  text encoder's EOS gather (x[arange, eos_index] in export_onnx.py:96 — that's a Gather on quantized acts, which historically blows up).
  3. If Stage 2 is fine but Stage 3 collapses → check if QAI Hub's --quantize_io default flipped, or whether the compiler is re-fusing ops post-quantization. Different problem class entirely.

  The actual question worth answering first

  Before any of the above: does qai_hub.submit_quantize_job accept an op-exclude list or per-op precision override? That's the one knob that turned 0.10 → 0.82 locally. If it does, your fix is one config change. If it doesn't, you'll need to inject QDQ skips into the ONNX before sending it
  to the quantize job.

  Want me to dig into the QAI Hub SDK to check that, or build out the local artifact-inspection scaffolding (Stage 0–2 runner) first? I'd lean toward the SDK check — it's 10 minutes and could short-circuit the whole investigation.

---

## QAI Hub SDK investigation — findings (2026-04-30)

### The full options surface for `submit_quantize_job`

The API is `qai_hub.submit_quantize_job(model, calibration_data, weights_dtype, activations_dtype, name, options, project)`. All knobs that affect quantization quality flow through the single `options: str` argument.

The official docs (`workbench.aihub.qualcomm.com/docs/hub/api.html`) and the QAI Hub Models reference repo (`qai_hub_models/models/common.py:985`) confirm there are **only two** flags:

| Flag | Values | Default | What it does |
|---|---|---|---|
| `--range_scheme` | `auto` \| `mse_minimizer` \| `min_max` | `auto` (currently `mse_minimizer`) | Calibration algorithm. `mse_minimizer` builds histograms and clips outliers; `min_max` uses absolute min/max |
| `--lite_mp` | `percentage=N;override_qtype=int16\|fp16` | not applied | Beta. Auto-detects the most quantization-sensitive N% of layers and promotes them to a higher precision |

### What's NOT in the API

- **No per-layer / per-op-name exclusion list.** Cannot say "leave LayerNorm in fp16."
- **No op-type filter.** Cannot say "skip Softmax, Gather."
- **No per-channel vs per-tensor toggle.**
- **No symmetric/asymmetric toggle.**

The closest substitute is `--lite_mp`: it does exactly the kind of "skip the sensitive layers" we did manually in the local QOperator path (where excluding LN/Softmax/Gather lifted us 0.10 → 0.82), but it picks the layers automatically rather than letting us name them.

### What QAI Hub Models do for similar problem classes

Looking at how Qualcomm themselves quantize transformers in `qai_hub_models`:

- `Precision.w8a16` → `--range_scheme min_max`
- `Precision.w8a16` mixed-precision → `--range_scheme min_max --lite_mp percentage={N};override_qtype=int16`
- The `qai_hub_models/models/openai_clip/perf.yaml` only lists a `float` precision — Qualcomm's own CLIP wrapper does NOT ship a quantized variant. Telling.

### Concrete recipes to try (lowest risk first)

These are single-config changes to `quantize_and_compile()` in `run_on_device.py:185`:

1. **Range scheme switch (W8A8):** `options="--range_scheme min_max"` — closer to MinMax which sometimes holds embedding norms better than histogram clipping.
2. **W8A16 baseline:** keep `weights_dtype=INT8, activations_dtype=INT16` + `options="--range_scheme min_max"`. Our `--precision w8a16` mode already does this — confirm it actually beats W8A8.
3. **W8A16 + Lite-MP fp16:** `options="--range_scheme min_max --lite_mp percentage=10;override_qtype=fp16"` — lets the most sensitive 10% of layers stay in fp16. This is the official path for "transformer quantization keeps blowing up."
4. **Sweep Lite-MP percentage:** 5, 10, 20, 30. There is a knee on these curves — at some point you've recovered enough signal that going further is just paying latency.
5. **Per-encoder asymmetry:** the prior local data shows the text encoder is the weaker link (`INT8_img + FP32_txt = 0.86` vs `FP32_img + INT8_txt = 0.85`, with a known fragile EOS Gather in `export_onnx.py:96`). Consider quantizing image at W8A8 and text at W8A16 (or text at W8A16 + lite_mp) so the latency budget goes where it gives back the most Recall@10.

### If the API knobs aren't enough — pre-quantization QDQ injection

Since QAI Hub will not let us nominate "do not quantize this op," the only way to enforce exclusions is to **shape the ONNX before submitting it**. Two viable approaches:

- **Pre-bake QDQ nodes locally with QOperator/QDQ for the ops we DO want quantized**, leaving the rest in fp32. Submit that hybrid ONNX to the compile job (skipping `submit_quantize_job` entirely). The QNN compiler will respect the QDQ scopes already present.
- **Use `onnxruntime.quantization` with `nodes_to_exclude=[...]` or `op_types_to_quantize=[...]`** to produce the hybrid ONNX, then send it through the standard compile path with `--target_runtime qnn_dlc` and no `--quantize_full_type`. This is exactly the path that already produced our 0.8256 Recall@10 in `inference_onnx_local.py` — we just haven't pushed that exact ONNX through QAI Hub yet.

The second option is the cheapest experiment by a wide margin: we already have the artifact, we already know it scores 0.8256 on local ORT, and the question collapses to "does the QNN compiler reproduce that on-device?"

### Recommended sequence of experiments

1. **Baseline measurement:** push the existing `image_encoder_int8.onnx` (the 0.8256 local QDQ artifact, hand-tuned to exclude LN/Softmax/Gather) through `--precision int8-local`. **This isolates: does the QNN compiler honor the QDQ scopes we set locally?** If yes, we have a working int8 path with ~0.83 Recall@10 today.
2. **Lite-MP sweep:** rerun `--precision w8a16` with `--lite_mp percentage=10;override_qtype=fp16` appended to the quantize options. This is the official Qualcomm answer to "transformer quantization is fragile."
3. **Per-tensor SQNR audit:** if both above plateau below baseline, instrument the optimized fp32 ONNX (Stage 1 artifact) and the quantized ONNX (Stage 2 artifact), expose intermediate tensors as outputs, and compute SQNR per layer on a few hundred calibration samples. The two or three worst layers are the budget-killers — knowing their names tells us what `--lite_mp` should be biased toward and whether a hybrid ONNX is even necessary.

---

## Summary: what we learned (LinkedIn-ready synopsis)

**Problem.** Quantizing CLIP ViT-B/16 with QAI Hub's stock `submit_quantize_job` (W8A8, default range scheme) collapses Recall@10 from 0.87 → ~0.05. The Qualcomm-recommended pipeline (compile-to-ONNX → quantize → compile-to-DLC) did not change this materially.

**Common myth busted.** "You can't compare INT8 outputs with FP32." You can — quantized op outputs live in float space (`scale * (q − zero_point)`). The right metrics are **per-tensor cosine similarity**, **SQNR**, and **max abs error** between fp32 and dequantized int8 activations. Treat quantization debugging as a numerical signal-loss audit, not a binary "did it break" check.

**Why the stock pipeline fails on transformer encoders.** Vision/text transformers concentrate signal in a handful of fragile sub-graphs — Softmax, LayerNorm, and the text encoder's EOS-token Gather. MinMax/MSE-minimizer calibration cannot safely fit int8 ranges to these. In the local ORT path, quantizing *all* ops scored 0.10; restricting quantization to Conv/MatMul/Gemm only scored **0.83**. The lift is entirely about which ops you leave alone.

**The QAI Hub API constraint.** `submit_quantize_job` exposes only `--range_scheme` and `--lite_mp`. There is **no per-layer / per-op exclusion mechanism**. The official Qualcomm answer to fragile-layer problems is `--lite_mp percentage=N;override_qtype=int16|fp16` — auto-promote the most sensitive layers. Notably, Qualcomm's own `qai_hub_models/openai_clip` only ships a `float` precision variant; they do not publish a quantized CLIP.

**Workaround.** When the API is too coarse, shape the ONNX before submission: produce a hybrid QDQ ONNX locally (QOperator format, restricted to Conv/MatMul/Gemm, percentile calibration) and send that through the compile path directly, bypassing `submit_quantize_job`.

**Debug ladder for any vendor-toolchain quantization regression.** Compile → quantize → compile-to-target produces three downloadable artifacts. Run each one locally, compute the target metric on each, and the failure stage points at the responsible component (export bug vs. compiler fusion vs. quantization itself vs. runtime). It is the cheapest possible bisection.

### Sources
- [`qai_hub.submit_quantize_job` reference](https://workbench.aihub.qualcomm.com/docs/hub/generated/qai_hub.submit_quantize_job.html)
- [QAI Hub Quantize Options](https://workbench.aihub.qualcomm.com/docs/hub/api.html)
- [QAI Hub Quantization examples](https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html)
- [`qai-hub-models` `Precision.aihub_quantize_options` (lines 960-988)](https://github.com/quic/ai-hub-models/blob/main/qai_hub_models/models/common.py)

---

## Reframing for learning mode (post-competition)

The competition is over. The remaining goal is **understanding the full edge-ML pipeline well enough to apply it to any future model on any future device** — CLIP / ViT today, MobileViT / a small LLM / a custom retrieval model tomorrow, on Hexagon today, NPU/CoreML/TFLite/EdgeTPU tomorrow. Every claim in this doc should be testable, every conclusion should be reproducible from the artifacts produced by `run_on_device.py`.

### Why we still need to fix CLIP-B/16 quantization first

A previous note suggested skipping ViT-B/16 quantization since FP32 already meets the latency budget. That was competition-thinking and is **retracted** for the learning track. Reasons:

- The pathologies that broke int8 on ViT-B/16 (Softmax, LayerNorm, EOS Gather) **scale up** on ViT-L/14, not down.
- ViT-B/16 is the right learning vehicle: small enough to iterate quickly, structurally identical to most production transformers, and we already have a known-bad and known-OK baseline (0.10 vs 0.83) to triangulate against.
- Insights here transfer directly to ViT-L/14, MobileViT, a quantized BERT, or any future int8 deployment problem.

### CLIP architecture reference (for grounding the debug)

Confirmed by direct introspection of `clip_model/clip/model.py`:

| Variant | Visual blocks | Text blocks | Visual width | Text width | Visual heads | Text heads |
|---|---|---|---|---|---|---|
| ViT-B/16 | 12 | 12 | 768 | 512 | 12 | 8 |
| ViT-L/14 | **24** | 12 | 1024 | 768 | 16 | 12 |

Each transformer block contains 2× LayerNorm + 1× Softmax = **3 fragile sub-graphs per block**. Partial-quantized ViT-B/16 has 36 fp32 islands in the visual encoder; ViT-L/14 has 72. This is also why partial-int8 was **slower**, not faster, on Hexagon (sweep_20260324_211019.log: img 32.5ms int8 vs 26.8ms fp32) — every island forces a quant→fp→quant conversion that the QNN runtime cannot fuse out, and the conversion overhead exceeds the int8 MatMul savings.

### Two questions, two instruments — separate cleanly

Most public quantization tutorials conflate these. Keep them apart:

| Question | Right metric | What it tells you |
|---|---|---|
| Where is the **signal** lost? | per-tensor SQNR, cosine-sim, max abs error between fp32 and dequantized-int8 activations | which ops degrade accuracy under int8 |
| Where is the **time** spent? | per-layer execution profile from QNN runtime, grouped by execution unit (HMX int8 / HVX fp16 / CPU) | which ops actually run on the fast path |

The fragile-for-accuracy and fragile-for-latency layers are correlated but **not identical** on Hexagon. That is the lesson worth internalizing.

### The four-tool learning ladder

Each tool builds on the prior. Each is small, each teaches a transferable skill.

| # | Tool | Lives in | Teaches |
|---|---|---|---|
| 1 | **Stage runner** — download Stage 0/1/2 ONNX artifacts from `quantize_and_compile()`, run each locally with onnxruntime, score Recall@10 | `src/debug/stage_runner.py` | how the QCom 3-step pipeline actually works; which stage the failure lives in (export bug vs Hub fusion vs quantization itself) |
| 2 | **Per-layer SQNR audit** — programmatically expose every transformer block's intermediate tensors as ONNX outputs in fp32 and quantized variants, push N calibration samples through both, log SQNR per tensor | `src/debug/sqnr_audit.py` (TBD) | ONNX graph surgery; signal-loss measurement; attention-block anatomy. Output is a ranked list of "the worst layers" |
| 3 | **Per-layer latency reader** — re-submit local-QDQ ONNX through QAI Hub, parse the profile log, group execution time by op type and execution unit | `src/debug/profile_reader.py` (TBD) | what "really running int8" means on Hexagon; how to read the QNN runtime's view of your graph |
| 4 | **Lite-MP exploration** — sweep `--lite_mp percentage={5,10,20,30}` on a W8A16 base, plot Recall@10 vs latency | extension to `run_on_device.py` (TBD) | how Qualcomm's sensitivity-detection works; whether it picks the same layers your SQNR audit flagged |

After tools 2 + 3 you'll know exactly which layers eat accuracy and which eat latency, and whether they're the same layers. After tool 4 you'll know whether QAI Hub's auto-detection agrees with your manual analysis. That's a complete mental model of transformer quantization on Hexagon, portable to any future edge ML project.

### Tool 1 design (this iteration)

`src/debug/stage_runner.py`. Single command, three stages, one summary table.

**Inputs.** A `--precision` flag (`int8-hub` or `w8a16`), the existing exported FP32 ONNX, and the calibration data already wired up in `src/common/calibration.py`. Optional `--*-job-id` flags to reuse jobs from a prior run (no Hub credits spent).

**What it does.**
1. Stage 0 — load the raw `exported_onnx/image_encoder.onnx` + `text_encoder.onnx`, run each locally, compute Recall@10. Establishes the ground truth.
2. Stage 1 — submit two `submit_compile_job(..., options="--target_runtime onnx")` jobs (image + text) with calibration data. Download the resulting optimized ONNX bundles. Run each locally. Compute Recall@10. Tells us whether Hub's optimizer pass (LayerNorm/Attention/GELU fusion) breaks anything in fp32.
3. Stage 2 — submit two `submit_quantize_job(..., weights_dtype, activations_dtype, options=...)` jobs against the Stage 1 models with the same calibration data. Download the quantized ONNX bundles. Run each locally on CPU. Compute Recall@10. Tells us whether quantization itself is the failure point.
4. Print a summary table: stage × Recall@10 × delta-from-Stage-0 × first-sample embedding stats.

**Outputs to disk:**
- `diagnostics/stage_artifacts/{precision}/{stage}/{encoder}.onnx` (+ `.data` if external)
- `diagnostics/stage_artifacts/{precision}/job_ids.json` — for `--reuse-jobs` on subsequent runs

**Expected signal:**
- If Stage 0 ≈ 0.87 and Stage 1 ≈ 0.87 and Stage 2 ≈ 0.05 → quantization itself is the failure (likely outcome — confirms fragile-ops hypothesis, sets up Tool 2).
- If Stage 1 < Stage 0 → Hub's optimizer broke the fp32 graph (different problem class, very different fix).
- If all three local stages are ≈ 0.87 but on-device is 0.05 → QNN runtime divergence (rare, but a clean signal).

This is the foundation everything else builds on. Once we know which stage owns the failure, Tool 2 (SQNR audit) is targeted at that stage's ONNX rather than blindly auditing every variant.
