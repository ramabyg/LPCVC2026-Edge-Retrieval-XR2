import sys
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "clip_model")
import clip as clip_lib

from inference import evaluate_track1

# --- Config ---
DATA_DIR  = r"C:\rama\projects\data\lpcvc_track1_sample_data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
IMG_LIST  = os.path.join(DATA_DIR, "img_list.csv")
TXT_LIST  = os.path.join(DATA_DIR, "txt_list.csv")
MODEL     = "ViT-B/16"
# --------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load model and correct CLIP preprocessing pipeline
print(f"Loading CLIP model ({MODEL})...")
model, preprocess = clip_lib.load(MODEL, device=device)
model.eval()

# 2. Load and preprocess images in img_list.csv order
df_img = pd.read_csv(IMG_LIST)
image_filenames = df_img.iloc[:, 0].tolist()
print(f"Loading {len(image_filenames)} images...")
images = torch.stack([
    preprocess(Image.open(os.path.join(IMAGE_DIR, f)).convert("RGB"))
    for f in image_filenames
]).to(device)  # shape: [N, 3, 224, 224]

# 3. Load and tokenize text prompts
df_txt = pd.read_csv(TXT_LIST)
prompts = df_txt.iloc[:, 1].dropna().tolist()
print(f"Tokenizing {len(prompts)} text prompts...")
text_tokens = clip_lib.tokenize(prompts).to(device)  # shape: [M, 77]

# 4. Run encoders locally
print("Running inference...")
with torch.no_grad():
    img_embeddings = model.encode_image(images).cpu().numpy()      # [N, 512]
    txt_embeddings = model.encode_text(text_tokens).cpu().numpy()  # [M, 512]

# 5. Reshape to list-of-arrays to match evaluate_track1() expected format
img_output = [img_embeddings[i:i+1] for i in range(len(img_embeddings))]
txt_output = [txt_embeddings[i:i+1] for i in range(len(txt_embeddings))]

# 6. Compute Recall@10
result = evaluate_track1(img_output, txt_output, TXT_LIST, IMG_LIST)
print(f"Local Recall@10: {result:.4f}")
