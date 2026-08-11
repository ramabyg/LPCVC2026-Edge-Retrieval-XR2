"""
AIMET-based INT8 Static Quantization for CLIP encoders using aimet_onnx.

Follows the recommended AIMET PTQ workflow:
  1. Batch Norm Folding (expected no-op on ViT — has LayerNorm, not BatchNorm)
  2. Cross-Layer Equalization (CLE) — optional, skip with --skip-cle
  3. QuantizationSimModel — int8 quantization with per-layer control
  4. Compute encodings over calibration data
  5. Export to QDQ ONNX (ready for QAI Hub compile or local inference via ORT)

Output naming follows the same convention as quantize.py so that
inference_onnx.py auto-discovers the models:

  exported_onnx/image_encoder{slug}_int8_aimet_{variant}.onnx
  exported_onnx/text_encoder{slug}_int8_aimet_{variant}.onnx

where {variant} encodes the preprocessing flags:
  cle          — CLE + BN folding (default)
  nocle        — --skip-cle
  clenobnf     — --skip-bnf
  noclenobnf   — --skip-cle --skip-bnf

Examples:
  image_encoder_int8_aimet_cle.onnx
  text_encoder_vitl14_int8_aimet_nocle.onnx

Calibration data: 56 sample images, 211 text prompts (same as quantize.py)
Quantization scheme: int8 weights, int8 activations, per-channel weights,
                     post_training_tf_enhanced (better for transformers than min_max)

Usage:
  python src/local/quantize_aimet.py                     # default: AIMET INT8 with CLE
  python src/local/quantize_aimet.py --skip-cle          # skip cross-layer equalization
  python src/local/quantize_aimet.py --skip-bnf          # skip batch norm folding
  python src/local/quantize_aimet.py --model ViT-L/14    # quantize the ViT-L/14 export
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict

import onnx
import clip as clip_lib

# AIMET imports
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from aimet_onnx.cross_layer_equalization import equalize_model
from aimet_onnx.common.defs import QuantScheme

from src.common.config import ONNX_DIR, IMAGE_DIR, IMG_LIST, TXT_LIST, ensure_output_dirs

ensure_output_dirs()

# ─────────────────────────────────────────────────────────────────────────────
# Which op types keep their quantizers.
#
# AIMET quantizes *every* activation by default, which destroys CLIP.  Measured
# with --quantize-all (2026-08-05):
#
#   text encoder   Recall@10 0.0000, mean pairwise cosine 1.0000 (fully collapsed)
#   image encoder  Recall@10 0.0137 (scrambled, not collapsed)
#
# ROOT CAUSE IS NOT YET IDENTIFIED.  An earlier version of this comment blamed the
# text encoder's -inf causal mask.  That was tested and is wrong: re-exporting with
# the mask clamped to a finite -25 (export_onnx.py --attn-mask-clamp) still scores
# 0.0000, the first text embedding is bit-identical to the unclamped build, and the
# image encoder — which has no causal mask at all — is destroyed just the same.
# The mask is a real defect worth fixing, but it is not what breaks quantization.
#
# What IS established: full activation quantization destroys CLIP in two
# independent toolchains, and restricting to the matmul family recovers it —
# ONNX Runtime static INT8 went 0.1003 (all ops) → 0.8256 (Conv/MatMul/Gemm only),
# see CLAUDE.md.  The AIMET equivalent of that restricted run has not been
# executed yet.
#
# Next diagnostic step is a leave-one-out bisection (return one op type at a time
# to float, starting with the image encoder since it has no mask, no embedding
# table and no EOS path).  See plans_notes/aimet_quantization_review_2026-08-04.md
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_QUANT_OP_TYPES = ("Conv", "ConvTranspose", "MatMul", "Gemm")

# Any calibrated range wider than this is a broken tensor (e.g. an -inf mask),
# not a real activation distribution.
MAX_SANE_ENCODING = 1e6


# ─────────────────────────────────────────────────────────────────────────────
# Target hardware config.
#
# LPCVC 2026 target device (confirmed via qai_hub.get_devices(), 2026-08-10):
#
#   Samsung Galaxy S22 (Family)
#     chipset:qualcomm-snapdragon-8gen1 / sm8450
#     hexagon:v69
#     htp-supports-fp16:true
#
# (XR2 Gen 2 was the original target but is deprecated from July 2026; LPCVC now
# recommends the S22 family.)  So the correct AIMET backend config is the
# **v69** family, NOT v73/v75.
#
# Every AIMET run before 2026-08-10 passed config_file=None, i.e. AIMET's generic
# default_config.json.  Measured differences vs v69 (verified 2026-08-10 by
# reading the installed JSONs — an earlier draft quoted "18 op_type / 14
# supergroups", which is v73/v75/v79/v81, NOT v69):
#
#                              op_type overrides   supergroups
#   default_config.json                2               5
#   htp_quantsim_config_v69.json      15               3
#
# Supergroups turn out to be IRRELEVANT here.  v69's three are
# (ConvTranspose,Relu), (Add,Relu), (Gemm,Relu); default's five are those plus
# (Conv,Relu) and (Conv,Clip).  CLIP ViT contains no Relu and no Clip — its
# activation is QuickGELU (Mul+Sigmoid) — so zero supergroups match either way.
#
# What actually differs for THIS model, all from the op_type overrides:
#
#   1. per_channel_quantization=True for Conv / Gemm / MatMul.
#      Our graphs: image = 1 Conv + 61 MatMul + 12 Gemm, text = 61 MatMul + 12
#      Gemm.  The generic default is per-tensor for all of them.
#   2. Gather: is_output_quantized=False.  We have 37 Gather ops per encoder,
#      and the token embedding lookup is one of them — so v69 says the backend
#      does not quantize the tensor that the 2026-08-05 review flagged as the
#      leading suspect for the text-encoder collapse.
#   3. LayerNormalization: params.weight.is_symmetric=False (26 / 25 LayerNorms).
#
# (1) and (2) are the substantive hypothesis; the supergroup argument is dead.
# Untested as of this writing.
#
# NOTE: htp_quantsim_config_v69.json and
# htp_quantsim_config_v69_per_channel_linear.json are BYTE-IDENTICAL in this
# AIMET build (2.34) — base v69 already carries the per-channel overrides.  The
# "htp_v69_pc" shorthand is kept only so the notebook-style name resolves; it
# selects the same contract as "htp_v69", so do not treat the two as an A/B on
# per-channel weights.
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_SHORTHANDS = {
    "default": None,                                    # AIMET generic default
    "htp_v69": "htp_quantsim_config_v69",               # Galaxy S22 / SD 8 Gen 1
    # identical file to htp_v69 in AIMET 2.34 — alias, not a distinct variant
    "htp_v69_pc": "htp_quantsim_config_v69_per_channel_linear",
}


def resolve_config_file(spec):
    """
    Resolve --config-file into a path AIMET accepts (or None).

    Accepts, in order of precedence:
      * None or "default"      -> None (AIMET's generic default_config.json)
      * a shorthand key        -> see CONFIG_SHORTHANDS
      * an existing file path  -> used as-is
      * a bare config name     -> resolved inside aimet_onnx's quantsim_config dir

    Note: AIMET's own get_path_for_target_config() builds "{name}.json", so the
    notebook-style "htp_v75" does NOT resolve — the real filenames are
    "htp_quantsim_config_v75.json".  Hence this resolver.
    """
    import glob
    from aimet_onnx.common.quantsim_config import config_utils

    cfg_dir = os.path.dirname(os.path.abspath(config_utils.__file__))

    if spec is None or spec == "default":
        return None
    if spec in CONFIG_SHORTHANDS:
        name = CONFIG_SHORTHANDS[spec]
        if name is None:
            return None
        spec = name
    if os.path.isfile(spec):
        return spec

    candidate = os.path.join(cfg_dir, spec if spec.endswith(".json") else spec + ".json")
    if os.path.isfile(candidate):
        return candidate

    available = sorted(
        os.path.basename(p)[:-5] for p in glob.glob(os.path.join(cfg_dir, "*.json"))
    )
    raise SystemExit(
        f"Error: could not resolve --config-file '{spec}'.\n"
        f"  Shorthands: {', '.join(CONFIG_SHORTHANDS)}\n"
        f"  Configs in {cfg_dir}:\n    " + "\n    ".join(available)
    )


def configure_quantizers(sim, allowed_op_types, encoder_type):
    """
    Disable quantizers on every op whose type is not in `allowed_op_types`.

    Leaves the compute-heavy Conv/MatMul/Gemm ops (weights + outputs) quantized
    so the model still gets the int8 speedup, while elementwise/normalization/
    softmax tensors stay in float.
    """
    kept, disabled = 0, 0
    for op in sim.connected_graph.get_all_ops().values():
        in_q, out_q, param_q = sim.get_op_quantizers(op)
        quantizers = list(in_q) + list(out_q) + list(param_q.values())
        if op.type in allowed_op_types:
            kept += sum(1 for q in quantizers if q is not None and q.enabled)
            continue
        for q in quantizers:
            if q is not None and q.enabled:
                q.enabled = False
                disabled += 1
    print(f"    Quantizers: {kept} kept ({', '.join(allowed_op_types)}), {disabled} disabled")
    return kept


def sanitize_encodings(sim, encoder_type):
    """
    Safety net: after calibration, disable any quantizer whose learned range is
    non-finite or absurdly wide.  Catches -inf/NaN tensors (attention masks,
    masked-fill sentinels) that would silently destroy the model otherwise.
    """
    bad = []
    for name, q in sim.get_qc_quantize_op().items():
        if not getattr(q, "enabled", False):
            continue
        try:
            encodings = q.get_encodings()
        except Exception:
            continue
        if encodings is None:
            continue
        if not isinstance(encodings, (list, tuple)):
            encodings = [encodings]
        for enc in encodings:
            lo, hi = getattr(enc, "min", None), getattr(enc, "max", None)
            if lo is None or hi is None:
                continue
            if not (np.isfinite(lo) and np.isfinite(hi)) or max(abs(lo), abs(hi)) > MAX_SANE_ENCODING:
                bad.append((name, float(lo), float(hi)))
                q.enabled = False
                break

    if bad:
        print(f"    WARNING: disabled {len(bad)} quantizer(s) with degenerate ranges "
              f"(non-finite or |range| > {MAX_SANE_ENCODING:g}):")
        for name, lo, hi in bad[:8]:
            print(f"      {name}: [{lo:.4g}, {hi:.4g}]")
        if len(bad) > 8:
            print(f"      ... and {len(bad) - 8} more")
    return bad


class ImageCalibrationReader:
    """Feeds competition-style /255 images (no extra normalization — baked into model)."""

    def __init__(self):
        df = pd.read_csv(IMG_LIST)
        filenames = df.iloc[:, 0].tolist()
        print(f"    Loading {len(filenames)} calibration images...")

        self._data = []
        for f in filenames:
            img = Image.open(os.path.join(IMAGE_DIR, f)).convert("RGB").resize((224, 224))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))[np.newaxis, :]  # (1, 3, 224, 224)
            self._data.append({"image": arr})

    def __iter__(self):
        """Yield calibration batches as dicts."""
        return iter(self._data)


class TextCalibrationReader:
    """Feeds tokenized text prompts as int64."""

    def __init__(self):
        df = pd.read_csv(TXT_LIST)
        prompts = df.iloc[:, 1].dropna().tolist()
        print(f"    Tokenizing {len(prompts)} calibration prompts...")

        import torch
        tokens = clip_lib.tokenize(prompts)  # (M, 77) int64
        self._data = []
        for i in range(len(tokens)):
            self._data.append({"text": tokens[i:i+1].numpy().astype(np.int64)})

    def __iter__(self):
        """Yield calibration batches as dicts."""
        return iter(self._data)


def quantize_encoder(
    model_onnx_path: str,
    output_path: str,
    encoder_type: str,
    calibration_data: List[Dict[str, np.ndarray]],
    skip_bnf: bool = False,
    skip_cle: bool = False,
    quant_op_types=DEFAULT_QUANT_OP_TYPES,
    config_file: str = None,
) -> None:
    """
    Quantize a CLIP encoder using AIMET's QuantizationSimModel.

    Args:
        model_onnx_path: Path to FP32 ONNX file
        output_path: Path to output quantized ONNX file
        encoder_type: "image" or "text" (for logging)
        calibration_data: List of calibration batches as dicts (e.g., [{"image": arr}, ...])
        skip_bnf: If True, skip batch norm folding
        skip_cle: If True, skip cross-layer equalization
        quant_op_types: Op types that keep their quantizers. None quantizes
            everything (AIMET default — known to break CLIP, see notes above).
        config_file: Path to an AIMET quantsim config JSON. None uses AIMET's
            generic default_config.json (per-tensor weights, 2 op_type overrides,
            5 supergroups) — which is NOT the target hardware's behaviour. See
            resolve_config_file().
    """
    print(f"\n[{encoder_type.upper()} Encoder] Loading ONNX model...")
    model = onnx.load(model_onnx_path)

    # Step 1: Batch Norm Folding (expected no-op on ViT — has LayerNorm, not BatchNorm)
    if not skip_bnf:
        print(f"[{encoder_type.upper()} Encoder] Batch Norm Folding...")
        _ = fold_all_batch_norms_to_weight(model)  # modifies in-place
        print(f"    BatchNorm folding complete (likely no-op on ViT)")

    # Step 2: Cross-Layer Equalization (CLE)
    if not skip_cle:
        print(f"[{encoder_type.upper()} Encoder] Cross-Layer Equalization...")
        equalize_model(model)  # modifies in-place
        print(f"    CLE complete")

    # Step 3: Create QuantizationSimModel
    print(f"[{encoder_type.upper()} Encoder] Creating QuantizationSimModel...")

    # Dummy input (matching calibration data structure)
    if encoder_type == "image":
        dummy_input = {"image": np.random.randn(1, 3, 224, 224).astype(np.float32)}
    else:  # text
        dummy_input = {"text": np.random.randint(0, 49408, (1, 77), dtype=np.int64)}

    # Create quantization simulator
    # Use post_training_tf_enhanced for better transformer support
    sim = QuantizationSimModel(
        model,
        dummy_input=dummy_input,
        param_type="int8",
        activation_type="int8",
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        config_file=config_file,
    )

    # Step 3b: Restrict which ops are quantized (must happen before calibration
    # so disabled quantizers never gather stats)
    if quant_op_types:
        print(f"[{encoder_type.upper()} Encoder] Restricting quantization scope...")
        configure_quantizers(sim, quant_op_types, encoder_type)
    else:
        print(f"[{encoder_type.upper()} Encoder] Quantizing ALL ops (AIMET default)")

    # Step 4: Compute encodings (quantization ranges) over calibration data
    print(f"[{encoder_type.upper()} Encoder] Computing encodings over {len(calibration_data)} {encoder_type} samples...")
    sim.compute_encodings(calibration_data)
    print(f"    Encodings computed")

    # Step 4b: Reject degenerate ranges (-inf masks etc.)
    sanitize_encodings(sim, encoder_type)

    # Step 5: Export to QDQ ONNX
    print(f"[{encoder_type.upper()} Encoder] Exporting to QDQ ONNX...")

    # Use to_onnx_qdq() to get the QDQ model, then save it
    qdq_model = sim.to_onnx_qdq()

    # Protobuf can't serialize >2GB in a single file — spill weights to
    # {output_path}.data (same convention as export_onnx.py).  ORT resolves the
    # reference by relative path, so both files must stay in the same directory.
    if qdq_model.ByteSize() > 2_000_000_000:
        data_name = os.path.basename(output_path) + ".data"
        onnx.save(
            qdq_model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
        )
    else:
        onnx.save(qdq_model, output_path)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"    Saved: {output_path}  ({size_mb:.1f} MB)")

    # Check for .onnx.data file (weights are stored separately for large models)
    data_path = output_path + ".data"
    if os.path.exists(data_path):
        data_size_mb = os.path.getsize(data_path) / 1e6
        print(f"    Weights: {data_path}  ({data_size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AIMET INT8 Static Quantization for CLIP"
    )
    parser.add_argument(
        "--model", default="ViT-B/16", choices=["ViT-B/16", "ViT-L/14"],
        help="CLIP model variant to quantize (must already be exported to ONNX)",
    )
    parser.add_argument(
        "--skip-bnf",
        action="store_true",
        help="Skip Batch Norm Folding (expected no-op on ViT, but included for completeness)",
    )
    parser.add_argument(
        "--skip-cle",
        action="store_true",
        help="Skip Cross-Layer Equalization (for A/B testing CLE impact)",
    )
    parser.add_argument(
        "--quantize-all", action="store_true",
        help=(
            "Quantize every op (raw AIMET default). Known to collapse the text "
            "encoder via the -inf causal mask — for A/B comparison only."
        ),
    )
    parser.add_argument(
        "--quant-op-types", default=",".join(DEFAULT_QUANT_OP_TYPES),
        help="Comma-separated op types that keep their quantizers.",
    )
    parser.add_argument(
        "--config-file", default="default",
        help=(
            "AIMET quantsim config. 'default' (AIMET generic, what every run "
            "before 2026-08-10 used), 'htp_v69' (Samsung Galaxy S22 / Snapdragon "
            "8 Gen 1 -- the LPCVC target, confirmed via qai_hub), 'htp_v69_pc' "
            "(alias -- byte-identical to htp_v69 in AIMET 2.34), a bare config "
            "name, or a path to a JSON. An unresolvable value prints the "
            "available list."
        ),
    )
    parser.add_argument(
        "--encoder", choices=("image", "text", "both"), default="both",
        help=(
            "Which encoder to quantize. Default 'both'. Use 'image'/'text' as "
            "separate invocations on memory-tight hosts -- MEASURED peak RSS is "
            "4.0-5.8 GB per encoder for ViT-B/16 (2026-08-10), so 'both' in one "
            "process does not fit in 7.5 GB. Also makes a failed run cheap to "
            "resume."
        ),
    )
    parser.add_argument(
        "--image-onnx", default=None,
        help="Explicit FP32 image-encoder ONNX path (overrides the --model-derived name).",
    )
    parser.add_argument(
        "--text-onnx", default=None,
        help=(
            "Explicit FP32 text-encoder ONNX path (overrides the --model-derived "
            "name). Use with a mask-clamped export, e.g. "
            "exported_onnx/text_encoder_maskclamp.onnx"
        ),
    )
    parser.add_argument(
        "--variant", default=None,
        help=(
            "Override the {variant} tag in the output filename "
            "(lowercase letters/digits only). Default is derived from "
            "--skip-cle / --skip-bnf, e.g. 'cle', 'nocle', 'clenobnf'."
        ),
    )
    args = parser.parse_args()

    model_name = args.model
    # Same slug convention as quantize.py / export_onnx.py: ViT-B/16 is canonical
    # (empty slug), everything else gets "_vitl14"-style suffix.
    slug = "" if model_name == "ViT-B/16" else "_" + model_name.lower().replace("/", "").replace("-", "")

    # Filename tag consumed by inference_onnx.py's discovery regex:
    #   {image,text}_encoder{slug}_int8_{format}_{activation}.onnx
    # We use format="aimet" and activation=<preprocessing variant>.
    quant_op_types = None if args.quantize_all else tuple(
        t.strip() for t in args.quant_op_types.split(",") if t.strip()
    )

    fmt_name = "aimet"
    if args.variant:
        act_name = args.variant.lower()
    else:
        act_name = "cle" if not args.skip_cle else "nocle"
        if args.skip_bnf:
            act_name += "nobnf"
        if args.quantize_all:
            act_name += "allops"

    config_path = resolve_config_file(args.config_file)

    image_onnx = args.image_onnx or os.path.join(ONNX_DIR, f"image_encoder{slug}.onnx")
    text_onnx = args.text_onnx or os.path.join(ONNX_DIR, f"text_encoder{slug}.onnx")

    image_out = os.path.join(ONNX_DIR, f"image_encoder{slug}_int8_{fmt_name}_{act_name}.onnx")
    text_out = os.path.join(ONNX_DIR, f"text_encoder{slug}_int8_{fmt_name}_{act_name}.onnx")

    # Check FP32 inputs exist (only the ones this invocation will actually read)
    required = []
    if args.encoder in ("image", "both"):
        required.append(image_onnx)
    if args.encoder in ("text", "both"):
        required.append(text_onnx)
    for path in required:
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run: python src/platform/export_onnx.py --model ViT-B/16")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"AIMET INT8 Quantization — {model_name}")
    print(f"  BN Folding: {not args.skip_bnf}")
    print(f"  CLE: {not args.skip_cle}")
    print(f"  Quantized ops: {'ALL' if quant_op_types is None else ', '.join(quant_op_types)}")
    print(f"  Config file: {config_path or 'AIMET default (generic, per-tensor)'}")
    print(f"  Encoder(s): {args.encoder}")
    print(f"  Output tag: int8_{fmt_name}_{act_name}")
    print(f"{'='*60}")

    # Quantize the requested encoder(s).
    #
    # Each encoder is handled in its own scope and its calibration set is loaded
    # only when needed, so peak RSS is one encoder's worth rather than both.
    #
    # MEASURED peak RSS per encoder, ViT-B/16, 2026-08-10 (/usr/bin/time -v):
    #   defscope   image 5.24 GB   text 5.10 GB
    #   htpscope   image 5.40 GB   text 3.97 GB
    #   htpallops  image 5.80 GB   text 4.96 GB
    # On a 7.5 GB host that fits only one encoder at a time, and only with the
    # IDE closed -- an earlier single-process run of both was OOM-killed.
    if args.encoder in ("image", "both"):
        print(f"\n[Calibration Data] Loading images...")
        image_calib_list = list(ImageCalibrationReader())
        quantize_encoder(
            image_onnx,
            image_out,
            "image",
            image_calib_list,
            skip_bnf=args.skip_bnf,
            skip_cle=args.skip_cle,
            quant_op_types=quant_op_types,
            config_file=config_path,
        )
        del image_calib_list

    if args.encoder in ("text", "both"):
        print(f"\n[Calibration Data] Loading text...")
        text_calib_list = list(TextCalibrationReader())
        quantize_encoder(
            text_onnx,
            text_out,
            "text",
            text_calib_list,
            skip_bnf=args.skip_bnf,
            skip_cle=args.skip_cle,
            quant_op_types=quant_op_types,
            config_file=config_path,
        )
        del text_calib_list

    print("\n" + "="*60)
    print("AIMET Quantization complete")
    print("="*60)
    if args.encoder in ("image", "both"):
        print(f"  {image_out}")
    if args.encoder in ("text", "both"):
        print(f"  {text_out}")
    print(f"\nNext: python src/local/inference_onnx.py --model {model_name} --sweep --inspect-embeddings")
