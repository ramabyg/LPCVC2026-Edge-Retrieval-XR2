"""
Score an arbitrary INT8 *image* encoder against the FP32 text encoder.

The --sweep discovery in inference_onnx.py requires a matching text_encoder
artifact for the same tag, so it cannot score an image-only bisection build.
This driver reuses that module's own loaders/inference so the numbers are
directly comparable, and computes the FP32/FP32 reference in the SAME process
rather than trusting a remembered 0.8728.

Run from the repo root:
    python src/local/analysis/score_image_build.py <image_int8.onnx> [more.onnx ...]
"""
import os
import sys

sys.path.insert(0, os.getcwd())

from src.common.config import ONNX_DIR, TXT_LIST, IMG_LIST
from src.common.eval import evaluate_track1
from src.local.inference_onnx import load_images, load_text_tokens, run_inference

IMAGE_FP32 = os.path.join(ONNX_DIR, "image_encoder.onnx")
TEXT_FP32 = os.path.join(ONNX_DIR, "text_encoder.onnx")


def main(paths):
    images = load_images()
    tokens = load_text_tokens()
    print(f"  {len(images)} images, {len(tokens)} prompts\n")

    # In-process FP32/FP32 reference — validates the harness before any INT8 claim.
    img_f32, txt_f32 = run_inference(IMAGE_FP32, TEXT_FP32, images, tokens, "FP32")
    ref = evaluate_track1(img_f32, txt_f32, TXT_LIST, IMG_LIST)
    print(f"\n  FP32/FP32 reference Recall@10 = {ref:.4f}\n")
    del img_f32

    results = []
    for p in paths:
        label = os.path.basename(p)
        img_int8, _ = run_inference(p, TEXT_FP32, images, tokens, label)
        r = evaluate_track1(img_int8, txt_f32, TXT_LIST, IMG_LIST)
        results.append((label, r))
        print(f"\n  {label}: INT8_img+FP32_txt Recall@10 = {r:.4f}  (vs {ref:.4f})\n")
        del img_int8

    print("=" * 70)
    print(f"  {'build':<55} {'R@10':>8}")
    print(f"  {'FP32 reference':<55} {ref:>8.4f}")
    for label, r in results:
        print(f"  {label:<55} {r:>8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
