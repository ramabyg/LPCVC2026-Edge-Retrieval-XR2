import sys
import argparse
import torch
import os

import clip as clip_lib

from src.common.config import ONNX_DIR

device = torch.device("cpu")  # use CPU to export onnx model to avoid GPU device issues

parser = argparse.ArgumentParser(description="Export CLIP encoder(s) to ONNX")
parser.add_argument(
    "--model", default="ViT-B/16", choices=["ViT-B/16", "ViT-L/14"],
    help="CLIP model variant to export (default: ViT-B/16)",
)
parser.add_argument(
    "--encoder", default="both", choices=["image", "text", "both"],
    help="Which encoder to export (default: both)",
)
parser.add_argument(
    "--weights", default=None, type=str,
    help="Path to merged fine-tuned weights (.pt file). If omitted, uses base CLIP weights.",
)
parser.add_argument(
    "--dtype", default="fp32", choices=["fp32", "fp16"],
    help="Export dtype: fp32 (default) or fp16 for native Hexagon FP16 compute.",
)
args = parser.parse_args()

# Derive output filename suffix — ViT-B/16 keeps canonical names for backward compat
dtype_suffix = "_fp16" if args.dtype == "fp16" else ""
if args.model == "ViT-B/16":
    image_onnx_path = os.path.join(ONNX_DIR, f"image_encoder{dtype_suffix}.onnx")
    text_onnx_path  = os.path.join(ONNX_DIR, f"text_encoder{dtype_suffix}.onnx")
else:
    slug = args.model.lower().replace("/", "").replace("-", "")  # "vitl14"
    image_onnx_path = os.path.join(ONNX_DIR, f"image_encoder_{slug}{dtype_suffix}.onnx")
    text_onnx_path  = os.path.join(ONNX_DIR, f"text_encoder_{slug}{dtype_suffix}.onnx")

# -----------------------------
# 1. Prepare Environment
# -----------------------------
os.makedirs(ONNX_DIR, exist_ok=True)
print(f"Model:  {args.model}  Dtype: {args.dtype}")
print(f"Saving ONNX files to directory: {os.path.abspath(ONNX_DIR)}")

# -----------------------------
# 2. Dummy inputs
# -----------------------------
DUMMY_IMAGE_INPUT = torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device)
DUMMY_TEXT_INPUT = torch.randint(0, 49408, (1, 77), dtype=torch.int64, device=device)

# -----------------------------
# 3. Load CLIP model from local clip_model module
# -----------------------------
print(f"Loading CLIP model ({args.model}) from clip_model...")
clip_model, _ = clip_lib.load(args.model, device=device)
export_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
clip_model = clip_model.to(export_dtype)
if args.weights:
    state = torch.load(args.weights, map_location=device)
    clip_model.load_state_dict(state, strict=False)
    print(f"Loaded fine-tuned weights from: {args.weights}")
clip_model.eval()

class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        # CLIP-specific normalization baked in — competition inputs are /255 only
        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1))

    def forward(self, images):
        images = images.to(self.mean.dtype)  # cast input to model dtype (fp16 or fp32)
        images = (images - self.mean) / self.std
        return self.visual(images)

class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        eos_index = token_ids.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_index]
        x = x @ self.text_projection
        return x

# -----------------------------
# 4. Create wrapper instances
# -----------------------------
image_encoder = ImageEncoderWrapper(clip_model)
text_encoder = TextEncoderWrapper(clip_model)
image_encoder.eval()
text_encoder.eval()


# -----------------------------
# 5. Export Image Encoder
# -----------------------------
if args.encoder in ("image", "both"):
    print(f"\nExporting Image Encoder to {image_onnx_path}...")

    torch.onnx.export(
        image_encoder,
        DUMMY_IMAGE_INPUT,
        image_onnx_path,
        input_names=["image"],
        output_names=["embedding"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=True,
    )
    print(f"  Saved: {image_onnx_path}")

# -----------------------------
# 6. Export Text Encoder
# -----------------------------
if args.encoder in ("text", "both"):
    print(f"\nExporting Text Encoder to {text_onnx_path}...")

    torch.onnx.export(
        text_encoder,
        DUMMY_TEXT_INPUT,
        text_onnx_path,
        input_names=["text"],
        output_names=["text_embedding"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=True,
    )
    print(f"  Saved: {text_onnx_path}")

print("\nExport complete.")
