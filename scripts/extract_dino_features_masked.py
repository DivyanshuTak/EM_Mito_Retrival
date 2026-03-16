#!/usr/bin/env python3
"""
DINO feature extraction with Masked mean Pooling.

Uses ground-truth segmentation masks to pool only foreground patch tokens
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel

from dataset_utils import EMPatchMaskDataset, collate_image_mask_meta

# DINOv3 patch size (spatial tokens = (H/16) * (W/16))
DINO_PATCH_SIZE = 16

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--config",
        type=Path,
        default="/media/sdb/divyanshu/divyanshu/hhmi/config/feature_extraction.yaml",
        help="Path to config YAML",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    model_name = cfg["model"]["name"]
    model_slug = model_name.split("/")[-1].replace("-", "_")
    dl_cfg = cfg["dataloader"]
    device = torch.device(cfg["device"])
    patch_size = cfg.get("patch_size", 256)
    dataset_name = cfg.get("dataset_name", "dataset")

    print("Masked token pooling - ")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    ds = EMPatchMaskDataset(paths["input_npz"])
    loader = DataLoader(
        ds,
        batch_size=dl_cfg["batch_size"],
        num_workers=dl_cfg["num_workers"],
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_image_mask_meta,
    )

    features_list = []
    metadata_list = []
    # set the grid 
    grid_h = patch_size // DINO_PATCH_SIZE
    grid_w = patch_size // DINO_PATCH_SIZE
    n_tokens = grid_h * grid_w

    with torch.inference_mode():
        for batch_imgs, batch_masks, batch_meta in loader:
            imgs_np = batch_imgs.permute(0, 2, 3, 1).numpy()
            mx = max(imgs_np.max(), 1e-6)
            if mx > 255:
                imgs_np = (imgs_np / mx * 255).astype(np.uint8)
            elif mx <= 1.0 and mx > 0:
                imgs_np = (imgs_np * 255).astype(np.uint8)
            else:
                imgs_np = imgs_np.astype(np.uint8)


            inputs = processor(
                images=list(imgs_np),
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            # (B, 1+4+N, H) -> (B, N, H)
            patch_tokens = outputs.last_hidden_state[:, 5:, :]

            batch_masks = batch_masks.to(device)

            for b in range(patch_tokens.shape[0]):
                target_id = batch_meta[b].get("target_label_id", 1)
                mask_2d = (batch_masks[b] == target_id).float()

                # Downsample mask to token grid: (H, W) -> (1, 1, grid_h, grid_w)
                mask_4d = mask_2d.unsqueeze(0).unsqueeze(0)
                mask_down = F.interpolate(
                    mask_4d,
                    size=(grid_h, grid_w),
                    mode="nearest",
                )
                mask_flat = mask_down.view(n_tokens, 1)

                # Zero out background tokens, sum foreground, normalize
                masked_tokens = patch_tokens[b] * mask_flat
                mask_sum = mask_flat.sum()
                if mask_sum > 0:
                    feat = masked_tokens.sum(dim=0) / mask_sum
                else:
                    feat = patch_tokens[b].mean(dim=0)

                features_list.append(feat.cpu().numpy())
                metadata_list.append(batch_meta[b])

    features = np.stack(features_list, axis=0)
    metadata_arr = np.empty(len(metadata_list), dtype=object)
    metadata_arr[:] = metadata_list

    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"features_maskedmeanpooled_{model_slug}_masked_{dataset_name}_patch{patch_size}.npz"     # save name

    np.savez_compressed(
        out_path,
        features=features,
        metadata=metadata_arr,
    )
    print("Saved masked features")
   


if __name__ == "__main__":
    main()
