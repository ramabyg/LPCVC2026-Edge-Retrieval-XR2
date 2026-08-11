# AIMET Quantization — Code Review & Root-Cause Analysis

**Date:** 2026-08-04
**Branch:** `quantization_wt_aimet`
**Files reviewed:** `src/local/quantize_aimet.py`, `src/local/inference_onnx.py`
**Status:** Phase 0 executed 2026-08-10 — **results are the LAST section of this
file, "PHASE 0 — RESULTS"**. Headline: `defscope` = **0.8610**, the best INT8
result to date. Open work is in "Next steps"; the next action is item 6
(leave-one-out bisection).

> **Reading order note.** Sections above are in the order they were written
> (2026-08-04 review → 08-05 mask work → 08-10 Phase 0). New results are
> **appended at the end**, not interleaved. Three sections mention "Phase 0":
> the commands (recipe), the pre-run rationale (superseded), and the results
> (last section) — the results are the one you want.

---

## TL;DR

1. **BN folding and CLE are exact no-ops on CLIP ViT.** Measured. The four output
   variants (`cle`/`nocle`/`clenobnf`/`noclenobnf`) are four filenames for one
   identical model.
2. **The text-encoder collapse is an export-level defect, not an AIMET one.**
   `text_encoder.onnx` contains 12 baked `-inf` mask initializers.
3. **The current default (`Conv/MatMul/Gemm` only) leaves ~200 compute ops in
   float32** — which is plausibly why on-device "INT8" is *slower* than FP32
   (39.1ms vs 31.4ms). It optimizes local Recall@10, the metric we already pass,
   and works against on-device latency, the metric we're stuck on.
4. **The mask clamp works but does not fix quantization (tested 2026-08-05).**
   `--quantize-all` on a clamped export scores **0.0000**, same as unclamped.
   Both encoders are destroyed by full activation quantization — including the
   image encoder, which has no mask at all. See the negative-result section.
   Keep the clamp anyway (it's free and removes non-finite values from the graph),
   but the search for the real cause is open. **Mask clamp value: `M = -25`.**
   Logits measured at `[-11.27, +15.48]`, spread D = 26.75. Any `|M| ≥ 20`
   reproduces the FP32 baseline **exactly** (0.8728); `-25` adds margin at 11%
   coarser int8 resolution. The `-1e4` value floated mid-review was never run and
   does not work.
   **Caveat: that the clamp actually rescues `--quantize-all` is still untested.**
   See "What is still NOT demonstrated."
5. The "order-dependent quantizer disabling" concern was **overstated**: it is
   latent and cannot trigger on these two graphs. Low priority.
6. **PHASE 0 RAN 2026-08-10. Two results.**
   **(a) The scope-restricted AIMET build works: `defscope` = 0.8610** — best
   INT8 result to date, +0.0354 over ORT static INT8 at equal scope (0.8256),
   −0.0118 vs FP32. This configuration had simply never been run.
   **(b) The target-hardware config hypothesis is dead — the config is *inert*.**
   `htp_v69` produces **byte-identical artifacts** to the generic default and an
   identical sim (371 quantizers, 297 enabled, same op-type breakdown). Every
   v69/default difference is a no-op on this graph: AIMET 2.34 already does
   per-channel by default, Gather outputs are already unquantized, and no
   supergroup matches (no Relu in CLIP ViT). Phase 0 could not have
   distinguished the two configs. See "PHASE 0 — RESULTS".
   **Correction:** an earlier draft claimed v69 has "18 op_type overrides and 14
   supergroups" — those belong to v73/v75/v79/v81; v69 has 15 and 3.
7. **Target device changed.** XR2 Gen 2 is deprecated from July 2026; LPCVC now
   recommends **Samsung Galaxy S22 (Family)** → Snapdragon 8 Gen 1 / sm8450 /
   **hexagon v69**. All prior on-device numbers in this document were measured on
   XR2 Gen 2 and are no longer the target.

---

## Finding 1 — BN folding and CLE do nothing (confirmed by measurement)

Ran `fold_all_batch_norms_to_weight` + `equalize_model` on both FP32 encoders,
then evaluated **without any quantization**:

```
[image] folded pairs: ([], [])     [text] folded pairs: ([], [])
Quant - INFO - High Bias folding is not supported for models without BatchNorm Layers

FP32 + BNfold + CLE (NO quantization) Recall@10 = 0.8728   (FP32 baseline 0.8728)
```

CLE equalizes conv/linear pairs joined by a positive-homogeneous activation. CLIP
ViT has LayerNorm + QuickGELU and no BatchNorm, so CLE finds no pairs to act on.

**Good news:** CLE is not *harming* the ViT — that was the leading suspicion going
into the review, and it is ruled out.

**Implication:** `--skip-cle` / `--skip-bnf` should be dropped from the variant tag
in the output filename. Keeping them costs 4x quantization runtime for zero
information and makes the sweep table look like it is A/B-testing something it
isn't. Keep the flags if desired; stop encoding them in the filename.

---

## Finding 2 — The `-inf` causal mask is baked into the exported ONNX

```
NON-FINITE initializer: unsqueeze_2  (1,1,77,77) min -inf max 0.0  n_inf 2926
NON-FINITE initializer: unsqueeze_5  (1,1,77,77) min -inf max 0.0  n_inf 2926
...  x12 total (unsqueeze_2, _5, _8, ... _35 — one per transformer layer)
```

Source: `clip_model/clip/model.py:335` — `mask.fill_(float("-inf"))`.

The comment block at `src/local/quantize_aimet.py:63-68` diagnosed this correctly.

---

## What `--quantize-all` actually means

AIMET's default behaviour is to attach a quantizer to **every** tensor — each op's
output activation and each weight. That is what the NPU wants: if every tensor is
int8, the Hexagon never has to convert back to float between ops.

`configure_quantizers()` switches most of those off. Measured effect:

| | image_encoder | text_encoder |
|---|---|---|
| nodes total | 481 | 490 |
| quantized (Conv/MatMul/Gemm) | 74 | 73 |
| data-movement (Reshape/Transpose/Squeeze/… — no real cost) | 209 | 208 |
| **compute ops left in float32** | **198** | **209** |

Ops left in float, image encoder:
`Add 61, Mul 48, Gather 37, LayerNormalization 26, Softmax 12, Sigmoid 12, Sub 1, Div 1`

Ops left in float, text encoder:
`Add 73, Mul 48, Gather 37, LayerNormalization 25, Softmax 12, Sigmoid 12, ArgMax 1, GatherND 1`

(`Mul`+`Sigmoid` = QuickGELU; `Add` = residual adds + mask add.)

Every one of those ~200 ops is a point where the graph must leave int8, compute in
float, and re-enter int8. That is the shape of a graph that measures 39.1ms in
"INT8" versus 31.4ms in FP32: conversion overhead on ~200 boundaries, int8 speedup
on only 74 matmuls.

**So `--quantize-all` is not merely a broken A/B control — it is the only
configuration that could plausibly produce an on-device speedup.**

### Why `--quantize-all` currently collapses the text encoder

1. Attention logits from `Q@Kᵀ/√d` sit in **`[-11.27, +15.48]`, spread D = 26.75**
   (measured 2026-08-05 — see "Measured logit range" below; an earlier draft of
   this document *assumed* ±30, which was never verified).
2. Graph computes `logits + mask`, where mask is `0` or `-inf`. Confirmed wiring:
   `MatMul → Add(mask) → Softmax`, i.e. the mask is **additive**, not an assignment.
3. A quantizer sits on that `Add` output. Calibration observes `min = -inf`,
   `max ≈ 15.5`.
4. Asymmetric int8 has 256 levels → `step = (max − min)/255 = inf`.
5. Every real logit rounds into a single bucket → softmax over identical logits →
   uniform attention → every token attends equally to every token.
6. All 211 prompts produce the identical embedding.

This is exactly the `mean pairwise cosine 1.0000` signature that
`check_embedding_health()` in `inference_onnx.py` was written to catch.

### Why `-1e4` does NOT fix it

> **Provenance — `-1e4` was never run.** It was never exported, never quantized,
> never benchmarked. It appeared only in conversation: it was suggested in prose at
> the end of the first review pass, and retracted one turn later when the arithmetic
> below was actually worked through. It is recorded here solely so the dead end is
> not re-proposed. **The only AIMET run that has ever executed is `cleallops`**
> (`--quantize-all`, unclamped `-inf` mask) — the two files dated Aug 4 05:47/05:49.
> No clamp of any value has been exported or quantized.

Same formula with the measured `L_max = 15.48`:
`step = (15.48 − (−10⁴))/255 ≈ 39`. Real logits spanning 26.75 collapse into
under one step. Just as dead as `-inf`.

### What the mask value actually needs to be

> **Two corrections to earlier drafts of this document.** Both were reasoning
> errors, caught by measurement:
>
> 1. An earlier draft claimed *"set `M = L_min` and masking costs zero quantization
>    resolution."* **Wrong** — that holds only if masked positions are *assigned*
>    `M`. The graph is `Add`, so masked positions become `logit + M`, and the span
>    is **always** `D + |M|`. There is no free lunch.
> 2. The first measurement script estimated spurious attention mass as
>    `77·exp(M)`, which assumes masked positions sit near the row max. Under causal
>    masking, **row 0 has exactly one unmasked entry**, and a masked logit there can
>    exceed it by up to `D`. The honest worst-case bound is `76·exp(D + M)`, roughly
>    13 orders of magnitude tighter, moving the requirement from ~`−25` to ~`−38`.
>
> Because bound-based reasoning had now been wrong twice, the mask value was settled
> by **direct measurement** (sweep below) rather than by a third bound.

The two constraints, with `S = L_max − (L_min + M) = D + |M|` the span the
quantizer must cover:

- **Masking must work.** Masked entries contribute spurious softmax mass. Worst
  case (row 0) is bounded by `76·exp(D + M)`. Sets a **lower** bound on `|M|`.
- **Logits must stay resolvable.** `step = S/255`; a logit error ε perturbs softmax
  probability ratios by `e^ε`. Sets an **upper** bound on `|M|`.

`-inf` (S = ∞) and `-1e4` (S = 10⁴) both blow the upper bound. That is the entire
reason they fail — and note this is a property of the *graph*, not of AIMET.

---

## Measured logit range (2026-08-05)

**Method.** Located the 12 non-finite mask initializers, found their consuming
`Add` nodes, and exposed both the `Add`'s non-mask input (pre-mask logits) and its
output (post-mask, what a quantizer would see) as extra graph outputs. Ran all 211
prompts through ORT with **`ORT_DISABLE_ALL`** graph optimization so the nodes were
not fused away. Read-only — the instrumented copy went to scratchpad,
`exported_onnx/` was never modified.

Script: `scratchpad/measure_logit_range.py` (see "Reproducing" below).

```
layer      L_min      L_max   spread D      p0.1     p99.9
------------------------------------------------------------
    0     -9.304     15.479     24.783    -6.353    14.762
    1    -11.268     14.273     25.541    -8.770     9.993
    2     -9.611     10.710     20.321    -7.356     7.938
    3     -9.974      5.327     15.301    -8.194     2.585
    4     -8.894      4.751     13.645    -8.146     2.577
    5     -9.354      4.667     14.021    -8.181     2.418
    6     -9.834      5.454     15.288    -8.561     2.304
    7    -10.621      5.034     15.655    -7.777     2.874
    8     -9.962      5.988     15.950    -9.062     2.522
    9    -10.975      6.306     17.281    -8.663     2.424
   10    -11.040      6.086     17.126    -8.308     4.657
   11     -8.093      4.711     12.804    -5.346     4.041
------------------------------------------------------------
GLOBAL pre-mask logits:  L_min=-11.268  L_max=15.479  spread D=26.746
post-mask (finite entries only): [-11.268, 15.026]
```

Notes:
- Layers 0-2 carry a much wider range (`L_max` up to 15.5) than layers 3-11
  (`L_max` ≈ 5). If a single mask constant proves too coarse, **per-layer mask
  values are available** — each layer has its own initializer, so they can differ.
- The p0.1/p99.9 columns show the distribution is not outlier-dominated; min/max
  and percentile calibration should give similar answers here.

---

## Mask-value sweep — measured FP32 Recall@10 (2026-08-05)

Patched the 12 mask initializers to a finite `M` and evaluated end-to-end.
Image embeddings computed once and reused (image encoder untouched).
Script: `src/local/analysis/sweep_mask_value.py`.

```
         M   Recall@10   vs FP32   worst-case bound 76·exp(D+M)
------------------------------------------------------------------
       -10      0.8796   +0.0068        1.42e+09
       -15      0.8683   -0.0045        9.59e+06
       -20      0.8728   -0.0000        6.46e+04
       -25      0.8728   -0.0000        4.36e+02
       -30      0.8728   -0.0000        2.94e+00
       -40      0.8728   -0.0000        1.33e-04
       -50      0.8728   -0.0000        6.05e-09
       -60      0.8728   -0.0000        2.75e-13
      -100      0.8728   -0.0000        1.17e-30
```

**Result: `|M| ≥ 20` reproduces the FP32 baseline exactly (0.8728).**
`M = -10` and `-15` visibly perturb the model — the mask genuinely leaks there.
(`-10` scoring *above* baseline at +0.0068 is not an improvement to chase; on 56
images that is well inside noise.)

### The bound was loose by ~4 orders of magnitude

At `M = -20` the worst-case bound predicts spurious attention mass of `6.5e4` —
catastrophic — yet measured Recall@10 is bit-identical to baseline. The bound is
not *wrong*; it describes an adversarial arrangement (a masked position holding the
global max logit while its row's single unmasked entry holds the global min) that
does not occur in this trained model.

This is the **third** time in this investigation that reasoning from bounds gave a
misleading answer (`-1e4`, `M = L_min`, `76·exp(D+M)`). All three were cheap to
settle by measurement. Treat the arithmetic here as intuition for *why* the effect
exists, and the sweep as the authority on *where the threshold is*.

### Recommendation: `M = -25`

| M | FP32 Recall@10 | span S = D+\|M\| | int8 step S/255 |
|---|---|---|---|
| -20 | 0.8728 (exact) | 46.7 | 0.183 |
| **-25** | **0.8728 (exact)** | **51.7** | **0.203** |
| -30 | 0.8728 (exact) | 56.7 | 0.223 |
| -40 | 0.8728 (exact) | 66.7 | 0.262 |

`-20` gives the finest quantization step, but sits exactly on the measured
threshold, and the sweep was run on the same 211 prompts used for calibration.
`-25` costs 11% coarser resolution for a safety margin against distribution shift
between calibration and eval. **Take `-25`; fall back to `-20` only if resolution
proves to be the binding constraint.**

If a single global constant turns out too coarse, per-layer values are available —
layers 3-11 have `L_max ≈ 5` vs layer 0's `15.5`, so they could use a tighter mask.

---

## The reframe

The current default excludes by **op type**, sweeping ~200 ops out of int8 to solve
a problem that lives in **12 tensors** (the per-layer masked-logit tensors).
The exclusion set should be counted in tensors, not op types.

Graded plan:

1. Measure the real pre-mask logit range; clamp the mask accordingly at export.
2. Run `--quantize-all` on the clamped export. The health check reports
   immediately whether it survived.
3. If it doesn't survive, exclude only the 12 softmax-input tensors (and possibly
   the 12 Softmax outputs) — ~24 exclusions instead of ~200.
4. Op-type restriction stays as the known-good fallback (0.8256 local).

**Caveat that must stay visible:** "dequant/requant boundaries cause the on-device
slowdown" is still a **hypothesis**. `plans_notes/quantization_on_XR2_platform_debug.md`
did not confirm it. Local Recall@10 cannot test it — only compiling both variants
and profiling on XR2 can. The value of step 2 is that it produces a graph worth
spending a profiling run on.

---

## Code issues — `src/local/quantize_aimet.py`

### Latent, low priority: shared quantizer objects in `configure_quantizers`

Earlier in the review this was described as "order-dependent." **That was wrong**,
and the correction matters:

```python
if op.type in allowed_op_types:
    kept += sum(1 for q in quantizers if q is not None and q.enabled)
    continue                      # counts only — never *enables* anything
for q in quantizers:
    if q is not None and q.enabled:
        q.enabled = False         # only this branch mutates
```

Only the disallowed branch mutates state. So a quantizer reachable from both an
allowed and a disallowed op ends up disabled **regardless of visit order** —
nothing ever turns it back on. Visit order affects only the printed
`kept`/`disabled` counts (the same quantizer can be tallied in both).

The real concern was that quantizers are shared objects keyed by *tensor name* in
`sim.qc_quantize_op_dict`, and `get_op_quantizers()` returns references to them —
so one op can switch off a quantizer another op wanted. Checked whether that can
happen here:

```
image_encoder.onnx  no-producer tensors consumed by BOTH allowed and disallowed ops: 0
text_encoder.onnx   no-producer tensors consumed by BOTH allowed and disallowed ops: 0
```

Zero, in both graphs, across all three sharing paths (`get_op_quantizers` only
classes a tensor as an *input* quantizer when it has no producer; outputs belong to
exactly one op by construction; the initializer/param path is covered by the same
check).

**Cannot trigger on these models.** Latent robustness issue + a misleading log
line. Fix if convenient (collect the keep-set first, then disable everything not in
it), but it has affected no result to date.

### Docstring overclaims quantization scheme

Lines 28-29 claim "per-channel weights, symmetric weights / asymmetric
activations." Nothing in the code sets any of that. The AIMET 2.34
`QuantizationSimModel.__init__` signature is:

```
(self, model, *, param_type='int8', activation_type='int8',
 quant_scheme=QuantScheme.min_max, config_file=None, dummy_input=None,
 user_onnx_libs=None, providers=None, path=None)
```

There is no per-channel argument — per-channel comes from `config_file` (unset here,
so AIMET's default config applies) or an explicit
`QcQuantizeOp.enable_per_channel_quantization()` call. **Either verify what the
default config actually does, or fix the docstring.** As it stands we don't know
which scheme produced our numbers.

### `sanitize_encodings()` is a band-aid, not a fix

- With op-type restriction, `-inf` never reaches an enabled quantizer → it never fires.
- With `--quantize-all`, it fires on the mask `Add` — but the damage is at the
  *Softmax input* quantizer, whose range is finite-but-catastrophic.
- `MAX_SANE_ENCODING = 1e6` will not catch a `-1e4` clamp either.

Keep it as an assertion / tripwire. Do not rely on it as a remedy.

### Minor

- `encoder_type` parameter unused in both `configure_quantizers` and `sanitize_encodings`.
- `import torch` at line 170 is unused.
- `ImageCalibrationReader` / `TextCalibrationReader` build their full list in
  `__init__` and are immediately `list()`-ed — the `__iter__` protocol is vestigial.
- Their preprocessing duplicates `load_images` / `load_text_tokens` in
  `inference_onnx.py` verbatim. If one drifts, calibration silently stops matching
  eval. Import from one shared location.
- `--variant` is not validated against the discovery regex in `inference_onnx.py`
  (`^image_encoder(...)_int8_([a-z]+)_([a-z0-9]+)\.onnx$`), so an underscore or
  uppercase char in the tag makes the output file invisible to the sweep with no error.
- The `>2GB` external-data branch (line 262) never triggers for ViT-B/16
  (346MB / 255MB) — only relevant to ViT-L/14, already declared a dead end.

---

## Code review — `src/local/inference_onnx.py`

In good shape. No bugs found.

- `check_embedding_health()` is well-targeted — mean pairwise cosine is the right
  signal for the collapse mode, and running it unconditionally rather than behind
  `--inspect-embeddings` is the correct call.
- `--int8-tag` is threaded consistently through both the sweep and targeted paths.
- Only nit: the return value of `check_embedding_health()` is discarded, so a
  collapsed model still runs a full eval and prints `Recall@10: 0.0000` beneath the
  warning. Arguably fine — seeing the zero is confirmation.

---

## Verified AIMET 2.34 API facts (checked against the installed env)

Env: `/mnt/rama_ml/conda_envs/lpcvc/bin/python`, `aimet-onnx` 2.34.0

- `QuantizationSimModel.__init__` signature as quoted above — keyword-only after `model`.
- `QuantScheme.post_training_tf_enhanced` exists (value 2). Full set: `min_max`,
  `post_training_tf_enhanced`, `training_range_learning_with_tf_init`,
  `training_range_learning_with_tf_enhanced_init`, `training_range_learning`,
  `post_training_percentile`.
- `compute_encodings()` is overloaded; passing an `Iterable[Dict[str, np.ndarray]]`
  is a supported form (runs `self.session.run(None, item)` per item). Current usage
  is correct.
- `to_onnx_qdq()` exists.
- `QcQuantizeOp.enabled` is a property **with a working setter** (sets
  `quant_info.enabled`; `False` overrides OpMode to passThrough). The restricted-scope
  path will run — note it appears **never to have been exercised**, since the only
  artifacts on disk are `*_cleallops.*` from a `--quantize-all` run.
- `QcQuantizeOp.get_encodings()` exists, returns `Optional[List[libpymo.TfEncoding]]`,
  and returns `None` when uninitialized or when `data_type == float`.
- `fold_all_batch_norms_to_weight(model: ModelProto) -> Tuple[List, List]`
- `equalize_model(model: ModelProto)`

---

## Current artifacts on disk

```
image_encoder_int8_aimet_cleallops.onnx    346 MB  Aug  4   0.0000
text_encoder_int8_aimet_cleallops.onnx     255 MB  Aug  4
image_encoder_int8_aimet_clampallops.onnx  346 MB  Aug  5   0.0000
text_encoder_int8_aimet_clampallops.onnx   255 MB  Aug  5
image_encoder_int8_aimet_defscope.onnx     330 MB  Aug 10   0.8610  <- BEST
text_encoder_int8_aimet_defscope.onnx      244 MB  Aug 10
```

`htpscope` and `htpallops` were byte-for-byte duplicates of `defscope` and
`clampallops` respectively (verified with `cmp`) and were **deleted 2026-08-11**,
reclaiming 1.1 GB. They are reproducible from the Phase 0 commands if ever
needed, but there is no reason to: the config that produced them is inert.

---

## Reproducing everything in this document

All analysis scripts are committed under `src/local/analysis/`. Every command needs
both the env var and the conda interpreter (base `python3` has no torch/aimet):

```bash
cd /mnt/rama_ml/projects/LPCVC2026-Edge-Retrieval-XR2
export LPCVC_DATA_DIR=/mnt/rama_ml/data/lpcvc_track1_sample_data
PY=/mnt/rama_ml/conda_envs/lpcvc/bin/python
```

**1. CLE / BN-folding no-op check** (Finding 1) — ~2 min

```bash
$PY src/local/analysis/cle_fp32_check.py
# expect: folded pairs ([], []) for both encoders, Recall@10 = 0.8728 unchanged
```

**2. Confirm the -inf initializers** (Finding 2) — instant

```bash
$PY - <<'EOF'
import onnx, numpy as np
from onnx import numpy_helper
m = onnx.load('exported_onnx/text_encoder.onnx')
for i in m.graph.initializer:
    a = numpy_helper.to_array(i)
    if a.dtype.kind == 'f' and a.size and not np.isfinite(a).all():
        print(i.name, a.shape, 'min', a.min(), 'n_inf', int((~np.isfinite(a)).sum()))
EOF
# expect: 12 x (1,1,77,77), min -inf, n_inf 2926
```

**3. Op-count breakdown** (the ~200-float-ops table) — instant

```bash
$PY - <<'EOF'
import onnx
from collections import Counter
ALLOWED={'Conv','ConvTranspose','MatMul','Gemm'}
MOVEMENT={'Reshape','Transpose','Squeeze','Unsqueeze','Identity','Cast','Shape','Concat','Slice'}
for f in ['exported_onnx/image_encoder.onnx','exported_onnx/text_encoder.onnx']:
    c=Counter(n.op_type for n in onnx.load(f).graph.node); tot=sum(c.values())
    allowed=sum(v for k,v in c.items() if k in ALLOWED)
    move=sum(v for k,v in c.items() if k in MOVEMENT)
    print(f, 'total', tot, '| quantized', allowed, '| movement', move,
          '| FLOAT COMPUTE', tot-allowed-move)
EOF
```

**4. Measure the pre-mask logit range** — ~2 min, read-only

```bash
$PY src/local/analysis/measure_logit_range.py
# expect: L_min=-11.268  L_max=15.479  D=26.746, plus per-layer table
```

**5. Mask-value sweep vs FP32 Recall@10** — ~10 min

```bash
$PY src/local/analysis/sweep_mask_value.py
# expect: |M| >= 20 -> 0.8728 exactly; -10/-15 perturb
```

Note: `measure_logit_range.py` must run ORT with
`GraphOptimizationLevel.ORT_DISABLE_ALL`, otherwise the `Add(mask)` nodes get fused
and the tensors it exposes disappear.

---

## Phase 0 commands (2x2 config x scope matrix) — EXECUTED 2026-08-10

> **These commands have been run. For what happened, jump to
> "PHASE 0 — RESULTS (RAN 2026-08-10)" below.** Headline: `defscope` = **0.8610**
> (best INT8 to date), `htp_v69` turned out to be inert, `--quantize-all` still
> 0.0000. This section is retained as the reproduction recipe.

**Three** quantization builds + one sweep — the fourth cell of the matrix is
already measured:

| | scope-restricted (Conv/MatMul/Gemm) | `--quantize-all` |
|---|---|---|
| generic `default` config | **A** — never run | `clampallops` = **0.0000** (done 2026-08-05) |
| `htp_v69` | **B** | **C** — the on-device-speed candidate |

(The originally-planned build D, "htp_v69 plain vs per-channel," is dropped:
the two config files are byte-identical, so it would have rebuilt C.)

Each build quantizes both encoders (~2-4 min each), so budget ~15 min for the
builds and ~30-40 min for the sweep, which also re-scores the two existing
`*allops` builds.

Common preamble:

```bash
cd /mnt/rama_ml/projects/LPCVC2026-Edge-Retrieval-XR2
export LPCVC_DATA_DIR=/mnt/rama_ml/data/lpcvc_track1_sample_data
PY=/mnt/rama_ml/conda_envs/lpcvc/bin/python
TXT=exported_onnx/text_encoder_maskclamp.onnx    # mask-clamped export (M = -25)
```

All four use the clamped text encoder so the mask is not a confounder, and
`--skip-cle --skip-bnf` because both are measured no-ops (Finding 1).

```bash
# A. generic default config + scope-restricted  <- the never-run baseline
$PY src/local/quantize_aimet.py --skip-cle --skip-bnf \
    --config-file default --text-onnx $TXT --variant defscope

# B. htp_v69 + scope-restricted
$PY src/local/quantize_aimet.py --skip-cle --skip-bnf \
    --config-file htp_v69 --text-onnx $TXT --variant htpscope

# C. htp_v69 + quantize-all                  <- the on-device-speed candidate
$PY src/local/quantize_aimet.py --skip-cle --skip-bnf --quantize-all \
    --config-file htp_v69 --text-onnx $TXT --variant htpallops
```

Then benchmark everything discovered in `exported_onnx/`:

```bash
$PY src/local/inference_onnx.py --sweep --inspect-embeddings
```

Reference points for reading the table:

| | Recall@10 |
|---|---|
| FP32 | 0.8728 |
| ORT static INT8, Conv/MatMul/Gemm only | 0.8256 |
| AIMET quantize-all, default config (`cleallops` / `clampallops`) | 0.0000 |

What each outcome means:

- **A ≈ 0.82-0.87** — scope restriction works in AIMET too; the config was never
  the issue. Proceed to Phase 1 to find what full quantization breaks.
- **A low but B high** — the HTP config is what matters, and the earlier collapse
  was largely an artifact of using the generic default.
- **C usable** — the most valuable outcome: full int8 with no float
  boundaries, i.e. an actual candidate for an on-device latency win. Push it
  through `compile_and_profile.py` against **Samsung Galaxy S22 (Family)**.
- **C >> `clampallops` (0.0000)** — since supergroups are ruled out and the
  configs differ only in per-channel weights, the `Gather` output rule, and
  LayerNorm weight symmetry, the credit belongs to one of those three. The
  `Gather` rule is the prime suspect (it covers the token-embedding tensor);
  confirm by re-running C with `Gather` forced back to quantized rather than
  assuming.
- **All still ~0.0** — the config hypothesis is dead too; go straight to Phase 1
  QuantAnalyzer on the image encoder.

Watch for the `check_embedding_health()` warnings in the sweep output — but
remember it detects *collapse*, not *scrambling*: the 0.0137 image encoder passed
it silently. Trust the Recall@10 number over the absence of a warning.

---

## PHASE 0 — pre-run rationale (SUPERSEDED — kept for the record)

> **Historical.** This section argues *why* Phase 0 was worth running. It was
> run on 2026-08-10 and the config hypothesis came back **inert** — see
> "PHASE 0 — RESULTS" above. Nothing below is a pending action.

### Target device, confirmed

XR2 Gen 2 is deprecated from July 2026. LPCVC recommends **Samsung Galaxy S22
(Family)**. Queried via `qai_hub.get_devices()` on 2026-08-10 — not assumed:

```
Samsung Galaxy S22 (Family)   os:android
  chipset:qualcomm-snapdragon-8gen1      chipset:sm8450
  hexagon:v69
  htp-supports-fp16:true
  framework:qnn / onnx / tflite
```

So the AIMET backend config family is **v69** — not the v73/v75 used in AIMET's
own example notebook. Also note `htp-supports-fp16:true`: native FP16 on this HTP
is a real alternative to INT8 and `export_onnx.py --dtype fp16` already exists.

### Why this is the next experiment

Every AIMET run to date passed `config_file=None`.

### Phase 0 config facts, corrected (measured 2026-08-10)

Read directly from the installed JSONs in
`aimet_onnx/common/quantsim_config/`. An earlier draft of this section asserted
v69 had 18 op_type overrides and 14 supergroups and built the whole rationale on
supergroup fusion. **Both numbers were wrong** — they belong to v73/v75/v79/v81:

| Config | op_type overrides | supergroups |
|---|---|---|
| `default_config.json` (what we used) | 2 | 5 |
| `htp_quantsim_config_v69.json` | **15** | **3** |
| `htp_quantsim_config_v69_per_channel_linear.json` | 15 | 3 (identical file) |
| `htp_quantsim_config_v73/v75/v79/v81.json` | 18 | 14 |

Two corrections follow, both of which shrink the hypothesis:

1. **The supergroup argument is dead.** v69's three supergroups are
   `(ConvTranspose,Relu)`, `(Add,Relu)`, `(Gemm,Relu)`; the generic default's
   five are those plus `(Conv,Relu)` and `(Conv,Clip)`. CLIP ViT contains **no
   Relu and no Clip** — its activation is QuickGELU (`Mul`+`Sigmoid`). Zero
   supergroups match under either config, so this mechanism cannot explain
   anything here. v69 in fact has *fewer* supergroups than the default.
2. **`htp_v69` and `htp_v69_pc` are byte-identical** in AIMET 2.34 (verified by
   sorted-key diff). Base v69 already carries the per-channel overrides. There
   is no per-channel A/B to run — the planned build D was a duplicate of C.

### What actually differs for this model

All of it comes from the op_type overrides, and all three are real:

| v69 override | Ops in image enc | Ops in text enc |
|---|---|---|
| `per_channel_quantization=True` for Conv/Gemm/MatMul | 1 + 12 + 61 | 12 + 61 |
| **`Gather: is_output_quantized=False`** | 37 | 37 |
| `LayerNormalization: params.weight.is_symmetric=False` | 26 | 25 |

The `Gather` rule is the interesting one: it directly addresses the outstanding
`token_embedding.weight_scale: shape=()` observation. The token-embedding lookup
is a `Gather`, and v69 says the backend does not quantize Gather outputs at all
— so the tensor the 2026-08-05 review flagged as the leading suspect is one the
target hardware would have left alone, while `default_config.json` quantized it.

**This is a hypothesis, not a finding.** Four hypotheses in this investigation
have already been wrong, and the first draft of *this* one was wrong twice over
before a single run. It is worth running first only because it is cheap.

### API gotchas found while reading AIMET's quant_analyzer.ipynb

Relevant to Phase 1/2, recorded here so they are not rediscovered:

1. `from aimet_common.utils import CallbackFunc` **raises ImportError** in our env
   (both `aimet_onnx` and `aimet_torch` installed, `aimet_common` deprecated since
   v2.20). Use `from aimet_onnx.common.utils import CallbackFunc`.
2. `QuantAnalyzer.__init__` in 2.34 is typed
   `(model, dummy_input, forward_pass_callback: Callable[[InferenceSession], Any],
   eval_callback: Callable[[InferenceSession], float])` — **plain callables**, not
   the notebook's `CallbackFunc` wrappers. `CallbackFunc` still exists; confirm
   which the implementation accepts before writing the script.
3. The notebook's `config_file="htp_v75"` **does not resolve**.
   `get_path_for_target_config()` builds `"{name}.json"`, but the real filenames
   are `htp_quantsim_config_v75.json`. Hence `resolve_config_file()` in
   `quantize_aimet.py`, which accepts shorthands (`htp_v69`, `htp_v69_pc`), bare
   names, or full paths, and lists the available configs on a miss.

### Phase 1/2 design note (QuantAnalyzer)

`eval_callback` receives **one** InferenceSession, but Recall@10 needs both
encoders. Resolution: precompute the *other* encoder's FP32 embeddings once, and
score the session under test against them — i.e. exactly the existing
`int8_img_fp32_txt` / `fp32_img_int8_txt` modes. Run the image encoder first: no
causal mask, no embedding table, no EOS path.

Runtime risk: `analyze()` runs the eval callback once per layer, twice (enable and
disable sweeps). With ~74 quantized layers and quantsim being materially slower
than plain ORT, budget 1-3 h per encoder. `enable_per_layer_mse_loss()` is much
cheaper (no eval) and may localize the problem on its own.

---

## Next steps (resume here)

1. ~~Measure the pre-mask logit range~~ — **DONE 2026-08-05.** D = 26.75.
2. ~~Determine the clamp value~~ — **DONE 2026-08-05.** Use **`M = -25`**.
3. ~~Clamp the mask at export~~ — **DONE 2026-08-05.** `export_onnx.py` gained
   `--attn-mask-clamp M`; `exported_onnx/text_encoder_maskclamp.onnx` verified at
   FP32 0.8728 with 0 non-finite initializers. **Keep this** — it is free and
   removes non-finite values the QNN compiler would have to handle.
4. ~~Run `--quantize-all` on the clamped export~~ — **DONE 2026-08-05 → 0.0000.**
   Hypothesis falsified. See the negative-result section.
5. ~~**PHASE 0 — the 2x2 config/scope matrix**~~ — **DONE 2026-08-10.**
   `defscope` = `htpscope` = **0.8610** (best INT8 to date); `htpallops` =
   0.0000. The config lever is inert — see "PHASE 0 — RESULTS". Do not spend
   more time on quantsim config files for this model.
6. **Leave-one-out bisection** to find what full quantization actually breaks:
   from all-quantized, return one op type at a time to float
   (`Gather`, `LayerNormalization`, `Softmax`, `Mul`/`Sigmoid`, `Add`).
   Run it on the image encoder first — it has no mask, no embedding table, and no
   EOS path, so it is the simpler of the two failures to isolate.
7. Check the token-embedding per-tensor quantization candidate (per-channel, or
   exclude that one tensor) — but only after the bisection points somewhere.
8. Drop CLE/BNF from the variant tag; collapse to a single variant.
9. Add a scrambling signal to `check_embedding_health()` (cosine vs FP32 reference)
   — a 0.0137 model passed it silently.
10. Whichever variant wins locally, **compile and profile it on XR2** — that is the
    only thing that tests the dequant/requant hypothesis.

---

## 2026-08-05 EXPERIMENT: mask clamp + `--quantize-all` — **NEGATIVE RESULT**

The decisive experiment ran. **The clamp does not rescue `--quantize-all`.**

```
Model      Format       Activation    Recall@10   vs FP32
ViT-B/16   FP32         —                0.8728
ViT-B/16   aimet        clampallops      0.0000   -0.8728
ViT-B/16   aimet        cleallops        0.0000   -0.8728
```

Pipeline that produced this (all gates passed):

1. `export_onnx.py --encoder text --attn-mask-clamp -25` → clamped 35112 non-finite
   entries across 12 blocks; exported ONNX verified to have **0 non-finite initializers**.
2. FP32 gate on the clamped export: **0.8728 exactly** — the clamp is genuinely free
   through the real export path, not just via initializer patching.
3. `quantize_aimet.py --quantize-all --skip-cle --skip-bnf --variant clampallops`
   — completed cleanly, no degenerate-range warnings (nothing left for
   `sanitize_encodings` to catch).
4. Benchmark: **0.0000**.

### The mask was real but secondary

Text encoder mean pairwise cosine: **1.0000** unclamped → **0.9938** clamped.
Measurably less degenerate, functionally identical.

The decisive detail — first text embedding, both builds:

```
clampallops  text embed[0]  min=-1.827  max=0.758  norm=5.515
cleallops    text embed[0]  min=-1.827  max=0.758  norm=5.515
```

**Bit-identical.** Removing 35112 `-inf` entries changed the first embedding not at
all. Whatever destroys the text encoder acts upstream of, or independently of, the
attention mask.

### Both encoders are destroyed, not just text

```
INT8_img(clampallops) + FP32_txt  →  Recall@10 = 0.0137
```

The image encoder — which has **no causal mask at all** — is equally destroyed by
full quantization. It simply failed to trip `check_embedding_health()` because its
embeddings are *scrambled* (mean pairwise cosine < 0.9) rather than *collapsed*
(> 0.99). An earlier note in this session speculated the image side might be fine
under `--quantize-all`; **that was wrong** — it scores 0.0137.

### What this actually establishes

Quantizing **every activation** with per-tensor int8 PTQ destroys CLIP, in **two
independent toolchains**:

| Toolchain | All ops quantized | Conv/MatMul/Gemm only |
|---|---|---|
| ONNX Runtime static (`quantize.py`) | 0.1003 | 0.8256 |
| AIMET `QuantizationSimModel` | 0.0000 | **0.8610** (measured 2026-08-10) |

The op-type restriction in `DEFAULT_QUANT_OP_TYPES` exists for a good reason. The
"reframe" argued earlier in this document — *exclusions should be counted in ~12-24
tensors, not ~200 ops* — is *not supported by this result* and should be treated as
refuted until a bisection shows otherwise.

### Candidate found by static inspection (NOT confirmed)

The token embedding table is quantized **per-tensor**, not per-channel:

```
token_embedding.weight_scale: shape=()  value=0.00088786   (scalar, no axis attribute)
token_embedding.weight_zero_point: 76 (uint8)
```

Implied calibrated range ≈ `[-0.068, +0.159]` against an actual table range of
`[-0.442, +0.463]` — `post_training_tf_enhanced` trimmed hard and clips the tails of
a 49408x512 vocabulary through one shared scale.

**Candidate only.** Resolution alone looks survivable (~14 levels per row-sigma);
the asymmetric clipping is what makes it suspicious. Given that three prior
hypotheses in this investigation were wrong, this one gets no weight until measured.

### Next: leave-one-out bisection

Start from fully-quantized and return one op type at a time to float
(`Gather`, `LayerNormalization`, `Softmax`, `Mul`/`Sigmoid`, `Add`) until each
encoder recovers. This finds the culprit set instead of guessing it, and is what
`QuantAnalyzer` (step 5 of `aimet_quantization_plan.md`) is for.

### Side finding: `check_embedding_health()` needs a second signal

A model scoring 0.0137 Recall@10 passed the health check silently. The mean-pairwise-
cosine test catches *collapse* but not *scrambling*. Worth adding a cheap second
signal — e.g. cosine against the FP32 reference embedding per sample, which would
have flagged both encoders immediately.

---

## What is still NOT demonstrated

Important before drawing conclusions or writing this up publicly:

| Claim | Status |
|---|---|
| CLE/BN-folding are no-ops on CLIP ViT | **Measured** |
| 12 baked `-inf` mask initializers exist | **Measured** |
| ~200 compute ops stay float under op-type restriction | **Measured** |
| Mask wiring is additive (`MatMul→Add→Softmax`) | **Measured** |
| Pre-mask logit range D = 26.75 | **Measured** |
| `\|M\| ≥ 20` preserves FP32 Recall@10 exactly | **Measured** |
| `step = (D+\|M\|)/255` resolution arithmetic | **Derived** (simple, sound) |
| `-inf` → inf quantizer step → attention collapse | **Partially refuted.** The mask does affect the model (mean cosine 1.0000 → 0.9938 when clamped), but it is not the dominant cause — see the negative result above |
| **Clamping the mask makes `--quantize-all` viable** | **TESTED 2026-08-05 → FALSE.** Recall@10 = 0.0000 clamped, same as unclamped |
| Full activation quantization destroys CLIP (both encoders, two toolchains) | **Measured** |
| Scope-restricted AIMET INT8 is usable (0.8610, beats ORT's 0.8256) | **Measured 2026-08-10** |
| Target-hardware (`htp_v69`) config changes the result | **TESTED → FALSE, and untestable as posed.** Byte-identical artifacts and identical sim state vs the generic default; every differing rule is a no-op on this graph |
| AIMET 2.34 applies per-channel weights without being asked | **Measured** — 50 per-channel scale vectors in a default-config artifact |
| Token embedding per-tensor quantization is the culprit | **Candidate only, unmeasured.** Note it is *not* explained by config choice — Gather outputs are unquantized under both configs |
| Dequant/requant boundaries cause the on-device slowdown | **Hypothesis only.** Only an XR2 profiling run can test it |

The headline story tried in this session — "a `-inf` in the exported graph is what
blocks INT8 quantization of CLIP's text encoder" — **did not survive testing.**
The `-inf` is a genuine defect worth fixing (it will trouble the QNN compiler
regardless, and the clamp is provably free), but it is **not** what blocks
quantization. Do not write this up as a fix until a bisection identifies the real
cause.

---

## Reference numbers

| Variant | Local Recall@10 | On-device Recall@10 | On-device latency |
|---|---|---|---|
| FP32 | 0.8728 | 0.7299 | 31.4 ms |
| FP32 + BNfold + CLE (no quant) | 0.8728 | — | — |
| FP32, mask clamped to −20 / −25 / −30 / −40 / −50 / −60 / −100 | 0.8728 (all) | — | — |
| FP32, mask clamped to −15 | 0.8683 | — | — |
| FP32, mask clamped to −10 | 0.8796 | — | — |
| AIMET INT8 `clampallops` (quantize-all, mask clamped −25) | **0.0000** | — | — |
| AIMET INT8 `clampallops` image only + FP32 text | **0.0137** | — | — |
| ORT static INT8 (`quantize.py`) | 0.8256 | 0.6804 | 39.1 ms (slower than FP32) |
| AIMET INT8 `cleallops` (quantize-all, unclamped mask) | 0.0000 | — | — |
| **AIMET INT8 `defscope` (scope-restricted, default config)** | **0.8610** | — | not yet profiled |
| **AIMET INT8 `htpscope` (scope-restricted, htp_v69)** | **0.8610** (byte-identical to `defscope`) | — | — |
| AIMET INT8 `htpallops` (quantize-all, htp_v69) | 0.0000 (byte-identical to `clampallops`) | — | — |

**Best INT8 to date: `defscope` / `htpscope` at 0.8610.** It is the obvious
candidate for the next on-device profiling run — but note it is scope-restricted,
so it carries the same ~200 float-compute ops that are the suspected cause of the
39.1 ms ORT INT8 result. It should be expected to fix Recall@10, **not** latency.

---

## PHASE 0 — RESULTS (RAN 2026-08-10)

Three builds (A/B/C), six invocations, all exit 0; then
`inference_onnx.py --sweep --inspect-embeddings`.

```
Model      Format   Activation     Recall@10   vs FP32
ViT-B/16   FP32     —                 0.8728
ViT-B/16   aimet    defscope          0.8610   -0.0118   <- A, NEW
ViT-B/16   aimet    htpscope          0.8610   -0.0118   <- B, NEW
ViT-B/16   aimet    htpallops         0.0000   -0.8728   <- C, NEW
ViT-B/16   aimet    clampallops       0.0000   -0.8728   (2026-08-05)
ViT-B/16   aimet    cleallops         0.0000   -0.8728   (2026-08-04)
```

### Result 1 — scope-restricted AIMET works, and beats ORT

**`defscope` = 0.8610**, the configuration that had never been run. This is the
best INT8 number the project has: **+0.0354 over ORT static INT8 at the same
scope (0.8256)**, and only **−0.0118** below the FP32 ONNX baseline. Embeddings
are healthy (no collapse warning, `cos(img[0],txt[0]) = 0.2545` vs FP32's
0.2565). The op-type restriction is sound in AIMET, and AIMET is the better
toolchain at equal scope.

### Result 2 — `--config-file htp_v69` is INERT (not "refuted" — inert)

The config never changed anything. Three independent checks:

1. **Byte-identical artifacts.** `md5sum` of the quantized ONNX:
   ```
   f32dea45…  image_encoder_int8_aimet_defscope.onnx
   f32dea45…  image_encoder_int8_aimet_htpscope.onnx     <- same
   4df4691c…  image_encoder_int8_aimet_htpallops.onnx
   4df4691c…  image_encoder_int8_aimet_clampallops.onnx  <- same as a DEFAULT-config
                                                            build from 2026-08-05
   ```
   Same pattern for both text encoders.
2. **Identical sim state.** Building `QuantizationSimModel` on
   `text_encoder_maskclamp.onnx` with `config_file=None` vs the v69 JSON gives
   **371 total quantizers, 297 enabled** in both, with an identical
   enabled-by-op-type breakdown
   (`Mul 48, Gemm 48, Add 37, MatMul 25, LayerNormalization 25, Softmax 12, Sigmoid 12`).
3. The build banner confirms the resolved path was passed, so this is not a
   plumbing bug in `resolve_config_file()`.

**Why it is inert** — each of the three differences identified above turns out to
be a no-op on this graph, for a *different* reason:

| v69 override | Why it changes nothing here |
|---|---|
| per-channel for Conv/Gemm/MatMul | AIMET 2.34 already does this by default. The `defscope` artifact (built with the generic default) contains **50 per-channel scale vectors** — `visual.conv1.weight_scale (768,)`, `…out_proj.weight_scale (768,)`, `(2304,)`, `(3072,)`. We never call `enable_per_channel_quantization()`; AIMET does it internally. |
| `Gather: is_output_quantized=False` | Gather outputs are **already unquantized** under the default config — `Gather` does not appear in the enabled-output-quantizer list for either config. The rule asks for behaviour that was already in place. |
| `LayerNormalization` weight asymmetry | Under scope restriction the LayerNorm quantizers are disabled anyway; under `--quantize-all` the artifacts are byte-identical, so it is not being applied. |
| supergroups | Already ruled out: no Relu/Clip in CLIP ViT. |

So the honest statement is **not** "the target-hardware config doesn't help."
It is: **for these two graphs, `htp_quantsim_config_v69.json` and AIMET 2.34's
generic default describe the same quantization contract**, so Phase 0 could not
have distinguished them. The hypothesis was untestable as posed, and the
`token_embedding` per-tensor observation is *not* explained by config choice.

**This is the fifth hypothesis in this investigation to fail.** It is also the
second one (after `-1e4`) that failed for a reason visible by inspection before
running anything — the per-channel scales were already in the artifacts on disk.

### Result 3 — full activation quantization still destroys the model

`htpallops` = 0.0000, byte-identical to `clampallops`. Text encoder collapses
(mean pairwise cosine 0.9938). The three-way tie
`cleallops`/`clampallops`/`htpallops` ≈ 0.0000 now covers unclamped mask, clamped
mask, and target-hardware config. **`--quantize-all` is dead by every lever tried
so far**; only the leave-one-out bisection can localize it.

### Measured peak RSS (new, useful for scheduling)

| Build | image | text |
|---|---|---|
| `defscope` | 5.24 GB | 5.10 GB |
| `htpscope` | 5.40 GB | 3.97 GB |
| `htpallops` | **5.80 GB** | 4.96 GB |

ViT-B/16 quantsim needs **4–5.8 GB per encoder** — far above the ~3 GB assumed
earlier (that figure was where the OOM killer struck, not the requirement). On
the 7.4 GB ThinkPad this fits only one encoder at a time with the IDE closed;
hence `quantize_aimet.py --encoder image|text|both`. Two runs were OOM-killed on
2026-08-10 before this was understood, one of which took down the tmux session.
