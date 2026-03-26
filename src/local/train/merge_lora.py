"""
Merge LoRA weights back into the base CLIP model and save as a plain state dict.

Track B6: After LoRA fine-tuning with finetune_lora.py, this script merges the
adapter weights into the base model so the result can be used directly without
PEFT at inference time.

Usage:
    python merge_lora.py --checkpoint lora_checkpoints/best --output clip_vitb16_lora_finetuned.pt
    python merge_lora.py --checkpoint lora_checkpoints/epoch_3  # default output name
"""

import sys
import os
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, "clip_model")
import clip as clip_lib

from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into base CLIP and save merged state dict"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to LoRA checkpoint directory (e.g. lora_checkpoints/best)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for merged state dict .pt file "
             "(default: merged_<checkpoint_dirname>.pt)",
    )
    return parser.parse_args()


def count_params(model):
    """Return (total_params, size_mb) for a model."""
    total = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    return total, size_mb


def main():
    args = parse_args()

    checkpoint_path = args.checkpoint
    if not os.path.isdir(checkpoint_path):
        print(f"ERROR: Checkpoint directory not found: {checkpoint_path}")
        sys.exit(1)

    # Derive output path
    if args.output is not None:
        output_path = args.output
    else:
        dirname = os.path.basename(os.path.normpath(checkpoint_path))
        output_path = f"merged_{dirname}.pt"

    # ------------------------------------------------------------------
    # 1. Load base CLIP model
    # ------------------------------------------------------------------
    print("Loading base CLIP ViT-B/16...")
    base_model, _ = clip_lib.load("ViT-B/16", device="cpu")
    base_model = base_model.float()

    # ------------------------------------------------------------------
    # 2. Load LoRA adapter
    # ------------------------------------------------------------------
    print(f"Loading LoRA adapter from: {checkpoint_path}")
    lora_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    lora_model = lora_model.float()
    lora_model.eval()

    # ------------------------------------------------------------------
    # 3. Generate reference embeddings before merge (for sanity check)
    # ------------------------------------------------------------------
    print("Computing reference embeddings for sanity check...")
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_text = clip_lib.tokenize(["a photo of a cat"])

    with torch.no_grad():
        ref_img_emb = lora_model.encode_image(dummy_image).float()
        ref_txt_emb = lora_model.encode_text(dummy_text).float()

    # ------------------------------------------------------------------
    # 4. Merge and unload
    # ------------------------------------------------------------------
    print("Merging LoRA weights into base model...")
    merged_model = lora_model.merge_and_unload()
    merged_model.eval()

    # ------------------------------------------------------------------
    # 5. Save merged state dict
    # ------------------------------------------------------------------
    print(f"Saving merged state dict to: {output_path}")
    torch.save(merged_model.state_dict(), output_path)

    # Print model info
    total_params, size_mb = count_params(merged_model)
    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"\nMerged model summary:")
    print(f"  Parameters:  {total_params:,}")
    print(f"  Model size:  {size_mb:.1f} MB (in memory)")
    print(f"  File size:   {file_size_mb:.1f} MB (on disk)")

    # ------------------------------------------------------------------
    # 6. Sanity check — reload and compare embeddings
    # ------------------------------------------------------------------
    print("\nRunning sanity check...")
    verify_model, _ = clip_lib.load("ViT-B/16", device="cpu")
    verify_model = verify_model.float()

    state_dict = torch.load(output_path, map_location="cpu", weights_only=True)
    verify_model.load_state_dict(state_dict)
    verify_model.eval()

    with torch.no_grad():
        check_img_emb = verify_model.encode_image(dummy_image).float()
        check_txt_emb = verify_model.encode_text(dummy_text).float()

    img_cos = F.cosine_similarity(ref_img_emb, check_img_emb).item()
    txt_cos = F.cosine_similarity(ref_txt_emb, check_txt_emb).item()

    print(f"  Image embedding cosine similarity: {img_cos:.6f}")
    print(f"  Text  embedding cosine similarity: {txt_cos:.6f}")

    if img_cos > 0.9999 and txt_cos > 0.9999:
        print("  PASS: Merged model matches LoRA model.")
    else:
        print("  WARNING: Embeddings diverge — check merge quality.")
        print(f"  Image L2 diff: {(ref_img_emb - check_img_emb).norm().item():.6f}")
        print(f"  Text  L2 diff: {(ref_txt_emb - check_txt_emb).norm().item():.6f}")

    print(f"\nDone. Merged weights saved to: {output_path}")


if __name__ == "__main__":
    main()
