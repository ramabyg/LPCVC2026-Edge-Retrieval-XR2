"""
LoRA fine-tuning for CLIP ViT-B/16 using PEFT library.

Track B4: Fine-tune CLIP with LoRA adapters on COCO and/or Flickr30k
using Karpathy splits. Targets attention output projections and MLP
layers in both image and text encoders.

Usage:
    python finetune_lora.py --epochs 3 --batch-size 64       # Quick run
    python finetune_lora.py --epochs 10 --amp                # Full training
    python finetune_lora.py --datasets both                  # COCO + Flickr30k
"""

import sys
import os
import json
import time
import math
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image

sys.path.insert(0, "clip_model")
import clip as clip_lib

from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
from src.common.config import (
    CLIP_MEAN, CLIP_STD,
    COCO_JSON, COCO_DATA_DIR, FLICKR30K_JSON, FLICKR30K_IMG_DIR,
    CHECKPOINTS_DIR,
)

DATASET_PATHS = {
    "coco":     {"json": COCO_JSON,     "image_root": COCO_DATA_DIR},
    "flickr30k":{"json": FLICKR30K_JSON,"image_root": FLICKR30K_IMG_DIR},
}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            224, scale=(0.7, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset — Karpathy JSON format
# ---------------------------------------------------------------------------
class CaptionDataset(Dataset):
    """Dataset for COCO / Flickr30k in Karpathy JSON format.

    Each item in the JSON ``images`` list has:
        - ``filepath`` or ``filename`` — relative image path
        - ``sentences`` — list of dicts with ``raw`` caption strings
        - ``split`` — one of ``train``, ``val``, ``restval``, ``test``

    For training we randomly sample one caption per image per call.
    For validation we return the first caption (deterministic).
    """

    def __init__(self, json_path, image_root, split, transform, max_text_len=77):
        super().__init__()
        self.image_root = image_root
        self.transform = transform
        self.max_text_len = max_text_len
        self.is_train = split == "train"

        logger.info("Loading Karpathy JSON: %s (split=%s)", json_path, split)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Karpathy convention: "restval" images are added to training set
        target_splits = {split}
        if split == "train":
            target_splits.add("restval")

        self.samples = []  # list of (image_path, [caption_str, ...])
        for entry in data["images"]:
            if entry["split"] not in target_splits:
                continue
            # Resolve image path: try "filepath" first, then "filename"
            rel_path = entry.get("filepath", entry.get("filename", ""))
            img_path = os.path.join(image_root, rel_path)
            captions = [s["raw"] for s in entry["sentences"]]
            if captions:
                self.samples.append((img_path, captions))

        logger.info("  Loaded %d images for split '%s'", len(self.samples), split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, captions = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Pick caption
        if self.is_train:
            caption = captions[torch.randint(len(captions), (1,)).item()]
        else:
            caption = captions[0]

        # Tokenize (returns shape [1, 77], squeeze to [77])
        tokens = clip_lib.tokenize([caption], truncate=True).squeeze(0)

        return image, tokens


# ---------------------------------------------------------------------------
# Loss — symmetric InfoNCE (CLIP loss)
# ---------------------------------------------------------------------------
def infonce_loss(img_feat, txt_feat, logit_scale):
    """Symmetric contrastive loss used by CLIP.

    Args:
        img_feat: (B, D) image embeddings (unnormalized OK)
        txt_feat: (B, D) text embeddings (unnormalized OK)
        logit_scale: scalar (exp of learned temperature)

    Returns:
        Scalar loss (mean of image-to-text and text-to-image cross-entropy).
    """
    img_feat = F.normalize(img_feat, dim=-1)
    txt_feat = F.normalize(txt_feat, dim=-1)
    logits = logit_scale * img_feat @ txt_feat.t()  # (B, B)
    labels = torch.arange(len(img_feat), device=img_feat.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2


# ---------------------------------------------------------------------------
# Validation — Recall@K on Karpathy val split
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, val_loader, device, logit_scale_val=None):
    """Compute image-to-text Recall@1/5/10 on the validation set.

    For each image, we check if its ground-truth caption(s) appear in
    the top-K most similar texts by cosine similarity.

    NOTE: This is a simplified 1-caption-per-image evaluation (since the
    val dataloader returns one caption per image). For full 5-caption
    evaluation, a dedicated function would be needed.
    """
    model.eval()
    all_img_feats = []
    all_txt_feats = []

    for images, tokens in val_loader:
        images = images.to(device)
        tokens = tokens.to(device)

        img_feat = model.encode_image(images).float()
        txt_feat = model.encode_text(tokens).float()

        all_img_feats.append(F.normalize(img_feat, dim=-1).cpu())
        all_txt_feats.append(F.normalize(txt_feat, dim=-1).cpu())

    img_feats = torch.cat(all_img_feats, dim=0)  # (N, D)
    txt_feats = torch.cat(all_txt_feats, dim=0)  # (N, D)

    # Cosine similarity matrix: (N_img, N_txt)
    sim = img_feats @ txt_feats.t()

    # Each image i has ground-truth text i (1-to-1 in this simplified setup)
    n = sim.size(0)
    ranks = torch.zeros(n)
    for i in range(n):
        sorted_indices = torch.argsort(sim[i], descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        ranks[i] = rank

    r1 = (ranks < 1).float().mean().item() * 100
    r5 = (ranks < 5).float().mean().item() * 100
    r10 = (ranks < 10).float().mean().item() * 100

    model.train()
    return r1, r5, r10


# ---------------------------------------------------------------------------
# Learning rate scheduler with linear warmup + cosine annealing
# ---------------------------------------------------------------------------
def get_lr(step, total_steps, warmup_steps, base_lr):
    """Linear warmup for ``warmup_steps``, then cosine decay to 0."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr_main, lr_logit_scale):
    """Set learning rates for the two parameter groups."""
    optimizer.param_groups[0]["lr"] = lr_main
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]["lr"] = lr_logit_scale


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for CLIP ViT-B/16"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4, help="LR for LoRA params")
    parser.add_argument("--lr-logit-scale", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--scheduler", default="cosine", choices=["cosine"])
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--logit-scale-clamp", type=float, default=4.6052,
                        help="Max value for logit_scale (ln(100))")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--drop-last", action="store_true", default=True,
                        help="Drop last incomplete batch (default: True)")
    parser.add_argument("--no-drop-last", dest="drop_last", action="store_false")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="Evaluate every N epochs")
    parser.add_argument("--save-dir", default=CHECKPOINTS_DIR)
    parser.add_argument("--datasets", default="coco",
                        choices=["coco", "flickr30k", "both"],
                        help="Training dataset(s)")
    parser.add_argument("--val-max-images", type=int, default=1000,
                        help="Max images for validation (0 = all)")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs without R@10 improvement)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_datasets(args):
    """Build train and val datasets based on --datasets flag."""
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_datasets = []
    dataset_names = ["coco", "flickr30k"] if args.datasets == "both" else [args.datasets]

    for name in dataset_names:
        cfg = DATASET_PATHS[name]
        ds = CaptionDataset(cfg["json"], cfg["image_root"], "train", train_transform)
        train_datasets.append(ds)
        logger.info("Train dataset '%s': %d samples", name, len(ds))

    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

    # Validation: always use COCO Karpathy val
    coco_cfg = DATASET_PATHS["coco"]
    val_dataset_full = CaptionDataset(
        coco_cfg["json"], coco_cfg["image_root"], "val", val_transform
    )

    # Optionally limit validation size
    if args.val_max_images > 0 and len(val_dataset_full) > args.val_max_images:
        val_dataset = torch.utils.data.Subset(
            val_dataset_full, list(range(args.val_max_images))
        )
        logger.info("Val dataset: %d / %d images (truncated)", args.val_max_images, len(val_dataset_full))
    else:
        val_dataset = val_dataset_full
        logger.info("Val dataset: %d images", len(val_dataset))

    return train_dataset, val_dataset


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 1. Load CLIP model
    # ------------------------------------------------------------------
    logger.info("Loading CLIP ViT-B/16...")
    model, _ = clip_lib.load("ViT-B/16", device=device)
    # CRITICAL: CLIP loads fp16 on CUDA -> must cast to fp32 for training
    model = model.float()

    # ------------------------------------------------------------------
    # 2. Apply LoRA
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["out_proj", "c_fc", "c_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    # NOTE: in_proj_weight (Q/K/V fused) is a raw Parameter, not nn.Linear,
    # so PEFT cannot target it directly.

    peft_model = get_peft_model(model, lora_config)

    # CRITICAL: unfreeze logit_scale so temperature can be learned
    for name, p in peft_model.named_parameters():
        if "logit_scale" in name:
            p.requires_grad_(True)
            logger.info("Unfroze parameter: %s", name)

    # Report trainable parameters
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info("Trainable parameters: %d / %d (%.2f%%)",
                trainable, total, 100.0 * trainable / total)

    peft_model.train()

    # ------------------------------------------------------------------
    # 3. Build datasets and dataloaders
    # ------------------------------------------------------------------
    train_dataset, val_dataset = build_datasets(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=args.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    total_steps = len(train_loader) * args.epochs
    logger.info("Total training steps: %d (%d batches x %d epochs)",
                total_steps, len(train_loader), args.epochs)

    # ------------------------------------------------------------------
    # 4. Optimizer — two param groups: LoRA params + logit_scale
    # ------------------------------------------------------------------
    lora_params = []
    logit_scale_params = []
    for name, p in peft_model.named_parameters():
        if not p.requires_grad:
            continue
        if "logit_scale" in name:
            logit_scale_params.append(p)
        else:
            lora_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": logit_scale_params, "lr": args.lr_logit_scale, "weight_decay": 0.0},
    ])

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    best_r10 = 0.0
    patience_counter = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        peft_model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        t_start = time.time()

        for batch_idx, (images, tokens) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            # LR schedule
            lr_main = get_lr(global_step, total_steps, args.warmup_steps, args.lr)
            lr_ls = get_lr(global_step, total_steps, args.warmup_steps, args.lr_logit_scale)
            set_lr(optimizer, lr_main, lr_ls)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=args.amp):
                img_feat = peft_model.encode_image(images).float()
                txt_feat = peft_model.encode_text(tokens).float()
                logit_scale = peft_model.logit_scale.exp()
                loss = infonce_loss(img_feat, txt_feat, logit_scale)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Clamp logit_scale after optimizer step
            with torch.no_grad():
                peft_model.logit_scale.clamp_(0.0, args.logit_scale_clamp)

            batch_size = images.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size
            global_step += 1

            # Log every 50 steps
            if global_step % 50 == 0:
                elapsed = time.time() - t_start
                throughput = epoch_samples / max(elapsed, 1e-6)
                logger.info(
                    "Epoch %d  Step %d/%d  loss=%.4f  lr=%.2e  logit_scale=%.3f  "
                    "throughput=%.1f img/s",
                    epoch, batch_idx + 1, len(train_loader),
                    loss.item(), lr_main,
                    peft_model.logit_scale.item(),
                    throughput,
                )

        # End of epoch
        epoch_time = time.time() - t_start
        avg_loss = epoch_loss / max(epoch_samples, 1)
        throughput = epoch_samples / max(epoch_time, 1e-6)
        logger.info(
            "Epoch %d complete  avg_loss=%.4f  logit_scale=%.3f  "
            "time=%.1fs  throughput=%.1f img/s",
            epoch, avg_loss, peft_model.logit_scale.item(),
            epoch_time, throughput,
        )

        # ------------------------------------------------------------------
        # 6. Validation
        # ------------------------------------------------------------------
        if epoch % args.eval_every == 0:
            logger.info("Running validation...")
            r1, r5, r10 = validate(peft_model, val_loader, device)
            logger.info(
                "Val  Recall@1=%.2f  Recall@5=%.2f  Recall@10=%.2f",
                r1, r5, r10,
            )

            # Early stopping check
            if r10 > best_r10:
                best_r10 = r10
                patience_counter = 0
                # Save best checkpoint
                best_dir = os.path.join(args.save_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                peft_model.save_pretrained(best_dir)
                logger.info("New best R@10=%.2f — saved to %s", r10, best_dir)
            else:
                patience_counter += 1
                logger.info(
                    "No R@10 improvement (%d/%d patience)",
                    patience_counter, args.patience,
                )

            if patience_counter >= args.patience:
                logger.info("Early stopping triggered after %d epochs.", epoch)
                break

        # ------------------------------------------------------------------
        # 7. Save epoch checkpoint
        # ------------------------------------------------------------------
        epoch_dir = os.path.join(args.save_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        peft_model.save_pretrained(epoch_dir)
        logger.info("Saved checkpoint: %s", epoch_dir)

    logger.info("Training complete. Best val Recall@10: %.2f", best_r10)
    logger.info("Best checkpoint: %s", os.path.join(args.save_dir, "best"))


if __name__ == "__main__":
    main()
