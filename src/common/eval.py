"""
Shared evaluation utilities for LPCVC 2026 Track 1.

evaluate_track1() and parse_ground_truth() are used by all inference scripts
(local, on-device, diagnostic). Extracted here so scripts in src/local/,
src/platform/, and src/debug/ can import from a single location without
pulling in qai_hub or any platform-specific dependencies.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def parse_ground_truth(txt_list, img_list):
    """Load text IDs and per-image ground-truth text ID lists from CSV files."""
    df_img = pd.read_csv(img_list)
    df_txt = pd.read_csv(txt_list)

    txt_id = df_txt.iloc[:, 0].dropna().astype(np.int16).tolist()
    gt = df_img.iloc[:, 1].dropna().tolist()  # list of txt id for each image
    return txt_id, gt


def evaluate_track1(img_output, txt_output, txt_list, img_list, k=10):
    """
    Compute Recall@K between image and text embeddings.

    Args:
        img_output: list of numpy arrays, each shape (1, D) — image embeddings
        txt_output: list of numpy arrays, each shape (1, D) — text embeddings
        txt_list (str): path to txt_list.csv
        img_list (str): path to img_list.csv
        k (int): Top-K for recall computation

    Returns:
        float: Mean Recall@K across all images
    """
    # Stack into 2D arrays: [N, D] and [M, D]
    img_embeds = np.vstack([x for x in img_output])
    txt_embeds = np.vstack([x for x in txt_output])

    # Normalize
    img_embeds = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
    txt_embeds = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)

    sim_matrix = cosine_similarity(img_embeds, txt_embeds)

    txt_id, gt = parse_ground_truth(txt_list, img_list)

    recalls = []
    for i in range(len(img_embeds)):
        gt_ids = [int(x) for x in gt[i].split(';')]

        # Top-K text indices by similarity
        k = 10
        top_k = np.argsort(-sim_matrix[i])[:k]

        # Map to real text IDs
        predicted_txt_ids = [txt_id[idx] for idx in top_k]

        # Fractional recall: how many GTs appear in top-K
        matched = len(set(predicted_txt_ids) & set(gt_ids))
        recall_i = matched / len(gt_ids)
        recalls.append(recall_i)

    return np.mean(recalls)
