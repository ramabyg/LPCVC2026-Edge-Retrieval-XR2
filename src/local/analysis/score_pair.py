"""
Score an arbitrary (image, text) INT8 encoder pair.

The per-tag discovery in inference_onnx.py assumes both encoders share one tag.
The best scope differs per encoder, so the shippable build is a MIXED-tag pair
and needs this driver. FP32/FP32 reference is computed in the same process.

Run from the repo root:
    python src/local/analysis/score_pair.py <image_int8.onnx> <text_int8.onnx>
"""
import os
import sys

sys.path.insert(0, os.getcwd())

from src.common.config import ONNX_DIR, TXT_LIST, IMG_LIST
from src.common.eval import evaluate_track1
from src.local.inference_onnx import load_images, load_text_tokens, run_inference

IMAGE_FP32 = os.path.join(ONNX_DIR, "image_encoder.onnx")
TEXT_FP32 = os.path.join(ONNX_DIR, "text_encoder.onnx")


def main(img_int8, txt_int8):
    images = load_images()
    tokens = load_text_tokens()
    print(f"  {len(images)} images, {len(tokens)} prompts\n")

    img_f32, txt_f32 = run_inference(IMAGE_FP32, TEXT_FP32, images, tokens, "FP32")
    ref = evaluate_track1(img_f32, txt_f32, TXT_LIST, IMG_LIST)
    print(f"\n  FP32/FP32 reference = {ref:.4f}\n")

    img_q, txt_q = run_inference(img_int8, txt_int8, images, tokens, "INT8 pair")

    rows = [
        ("FP32_img + FP32_txt (reference)", ref),
        ("INT8_img + FP32_txt", evaluate_track1(img_q, txt_f32, TXT_LIST, IMG_LIST)),
        ("FP32_img + INT8_txt", evaluate_track1(img_f32, txt_q, TXT_LIST, IMG_LIST)),
        ("INT8_img + INT8_txt  <-- shippable", evaluate_track1(img_q, txt_q, TXT_LIST, IMG_LIST)),
    ]

    print("=" * 72)
    print(f"  image: {os.path.basename(img_int8)}")
    print(f"  text : {os.path.basename(txt_int8)}")
    print("-" * 72)
    for label, r in rows:
        delta = "" if r == ref else f"  ({r - ref:+.4f})"
        print(f"  {label:<40} {r:>8.4f}{delta}")
    print("=" * 72)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
