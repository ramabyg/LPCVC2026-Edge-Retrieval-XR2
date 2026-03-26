import sys
import pandas as pd
import numpy as np
import os
import qai_hub
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, InterpolationMode

sys.path.insert(0, "clip_model")
import clip as clip_lib

# Competition-style preprocessing: matches inference_local.py exactly
preprocess = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    ToTensor(),  # uint8 HWC → float32 CHW, divides by 255
])

def process_image(image_path):
    """Loads and processes an image to competition format: float32 (1, C, H, W), values in [0, 1]."""
    img = preprocess(Image.open(image_path).convert("RGB"))  # [3, 224, 224]
    return img.numpy()[np.newaxis, :]  # [1, 3, 224, 224]

image_folder = "C:\\rama\\projects\\data\\lpcvc_track1_sample_data\\images"
img_list_csv = "C:\\rama\\projects\\data\\lpcvc_track1_sample_data\\img_list.csv"

# Load images in img_list.csv row order — must match evaluate_track1() expectations
df_img = pd.read_csv(img_list_csv)
image_filenames = df_img.iloc[:, 0].tolist()
input_image = [process_image(os.path.join(image_folder, f)) for f in image_filenames]
print(f"Processed {len(input_image)} images.")
print(f"First image shape: {input_image[0].shape}")  # Should be (1, 3, 224, 224)
print(f"First 3 filenames: {image_filenames[:3]}")  # Verify CSV order

# Upload dataset
print(qai_hub.upload_dataset({"image": input_image}))

# TODO: Load txt CSV
csv_path = "C:\\rama\\projects\\data\\lpcvc_track1_sample_data\\txt_list.csv"
df = pd.read_csv(csv_path)

# Get unique text prompts in order from the second column, drop NaN
prompts = df.iloc[:, 1].dropna().tolist()

# Tokenize using OpenAI CLIP tokenizer — matches inference_local.py
tokenized_texts = []
for prompt in prompts:
    tokens = clip_lib.tokenize([prompt])  # int64 tensor [1, 77]
    tokenized_texts.append(tokens.numpy().astype(np.int32))  # int32 for QAI Hub

# Example: check first element
print(tokenized_texts[0].shape)  # (1, 77)
print(tokenized_texts[0].dtype)  # int32

print(qai_hub.upload_dataset({"text": tokenized_texts}))
