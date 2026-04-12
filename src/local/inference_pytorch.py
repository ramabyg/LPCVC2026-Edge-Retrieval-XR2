import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, InterpolationMode

import clip as clip_lib

from src.common.eval import evaluate_track1
from src.common.config import IMAGE_DIR, IMG_LIST, TXT_LIST, CLIP_MEAN, CLIP_STD

parser = argparse.ArgumentParser(description="Local FP32 CLIP inference")
parser.add_argument(
    "--model", default="ViT-B/16",
    choices=["ViT-B/16", "ViT-L/14", "EVA02-B-16", "SigLIP-2-B-16"],
    help="Model variant to evaluate (default: ViT-B/16)",
)
parser.add_argument(
    "--weights", default=None, type=str,
    help="Path to merged fine-tuned weights (.pt file). If omitted, uses base model weights.",
)
args = parser.parse_args()

MODEL = args.model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load model
print(f"Loading model ({MODEL})...")

if MODEL in ("ViT-B/16", "ViT-L/14"):
    # OpenAI CLIP
    model, _ = clip_lib.load(MODEL, device=device)
    model = model.float()
    if args.weights:
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded fine-tuned weights from: {args.weights}")
    model.eval()
    norm_mean = CLIP_MEAN
    norm_std = CLIP_STD
    tokenize_fn = clip_lib.tokenize
    encode_image_fn = lambda imgs: model.encode_image(imgs)
    encode_text_fn = lambda toks: model.encode_text(toks)

elif MODEL == "EVA02-B-16":
    import open_clip
    model, _, preprocess_eva = open_clip.create_model_and_transforms(
        'EVA02-B-16', pretrained='merged2b_s8b_b131k'
    )
    model = model.float().to(device).eval()
    if args.weights:
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded fine-tuned weights from: {args.weights}")
    tokenizer = open_clip.get_tokenizer('EVA02-B-16')
    norm_mean = CLIP_MEAN  # EVA-CLIP uses same normalization as OpenAI CLIP
    norm_std = CLIP_STD
    tokenize_fn = tokenizer
    encode_image_fn = lambda imgs: model.encode_image(imgs)
    encode_text_fn = lambda toks: model.encode_text(toks)

elif MODEL == "SigLIP-2-B-16":
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
    model_id = "google/siglip2-base-patch16-224"
    siglip_model = AutoModel.from_pretrained(model_id).to(device).eval()
    siglip_tokenizer = AutoTokenizer.from_pretrained(model_id)
    siglip_processor = AutoImageProcessor.from_pretrained(model_id)
    # SigLIP uses its own normalization constants
    norm_mean = list(siglip_processor.image_mean)
    norm_std = list(siglip_processor.image_std)
    tokenize_fn = None  # handled separately below
    encode_image_fn = lambda imgs: siglip_model.get_image_features(pixel_values=imgs)
    encode_text_fn = None  # handled separately below

# Competition-style preprocessing: resize + /255 + model-specific normalization
preprocess_competition = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    ToTensor(),  # converts HWC uint8 → CHW float32 and divides by 255
])

MEAN = torch.tensor(norm_mean, device=device).reshape(1, 3, 1, 1)
STD  = torch.tensor(norm_std,  device=device).reshape(1, 3, 1, 1)

# 2. Load and preprocess images in img_list.csv order
df_img = pd.read_csv(IMG_LIST)
image_filenames = df_img.iloc[:, 0].tolist()
print(f"Loading {len(image_filenames)} images...")
images = torch.stack([
    preprocess_competition(Image.open(os.path.join(IMAGE_DIR, f)).convert("RGB"))
    for f in image_filenames
]).to(device)  # shape: [N, 3, 224, 224], values in [0, 1]
images = (images - MEAN) / STD  # apply model-specific normalization

# 3. Load and tokenize text prompts
df_txt = pd.read_csv(TXT_LIST)
prompts = df_txt.iloc[:, 1].dropna().tolist()
print(f"Tokenizing {len(prompts)} text prompts...")

if MODEL == "SigLIP-2-B-16":
    text_inputs = siglip_tokenizer(prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    text_tokens = {k: v.to(device) for k, v in text_inputs.items()}
else:
    text_tokens = tokenize_fn(prompts).to(device)  # shape: [M, 77]

# 4. Run encoders locally
print("Running inference...")
with torch.no_grad():
    img_embeddings = encode_image_fn(images).cpu().numpy()      # [N, embed_dim]
    if MODEL == "SigLIP-2-B-16":
        txt_embeddings = siglip_model.get_text_features(**text_tokens).cpu().numpy()
    else:
        txt_embeddings = encode_text_fn(text_tokens).cpu().numpy()  # [M, embed_dim]

# 5. Reshape to list-of-arrays to match evaluate_track1() expected format
img_output = [img_embeddings[i:i+1] for i in range(len(img_embeddings))]
txt_output = [txt_embeddings[i:i+1] for i in range(len(txt_embeddings))]

# 6. Compute Recall@10
result = evaluate_track1(img_output, txt_output, TXT_LIST, IMG_LIST)
print(f"Local Recall@10: {result:.4f}")
