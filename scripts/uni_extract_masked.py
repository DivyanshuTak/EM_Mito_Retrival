#!/usr/bin/env python3
"""
Extract masked-token-pooled features for UNI FM.
"""

import argparse
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset_utils import EMPatchMaskDataset, collate_image_mask_meta


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


## imagenet normalization for UNI 
def normalize_batch_for_imagenet(batch_imgs: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of images for imagenet-pretrained ViT
    Expected input shape: (B, 3, H, W)
    """
    batch_imgs = batch_imgs.float()

    max_val = batch_imgs.max()
    if max_val > 1.0:
        batch_imgs = batch_imgs / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406], device=batch_imgs.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=batch_imgs.device).view(1, 3, 1, 1)

    batch_imgs = (batch_imgs - mean) / std
    return batch_imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--config",
        type=Path,
        default="/media/sdb/divyanshu/divyanshu/hhmi/config/pathology.yaml",
        help="Path to pathology config YAML",
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

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    
    # load from timm
    model = timm.create_model(
        f"hf-hub:{model_name}",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    ).to(device).eval()

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

    with torch.inference_mode():
        for batch_imgs, batch_masks, batch_meta in loader:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            batch_masks = batch_masks.to(device, non_blocking=True)

            # Fixed preprocessing: direct tensor normalization
            img_tensor = normalize_batch_for_imagenet(batch_imgs)

            tokens = model.forward_features(img_tensor)

            # take the patch tokens 
            patch_tokens = tokens[:, 1:, :]  # (B, N, D)

            n_spatial_tokens = patch_tokens.shape[1]
            grid_size = int(n_spatial_tokens ** 0.5)

            for b in range(patch_tokens.shape[0]):
                target_id = batch_meta[b].get("target_label_id", 1)

                # Binary mask for the target mitochondria
                mask_2d = (batch_masks[b] == target_id).float()

                # Downsample mask to token grid
                mask_4d = mask_2d.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                mask_down = F.interpolate(
                    mask_4d,
                    size=(grid_size, grid_size),
                    mode="nearest",
                )
                mask_flat = mask_down.view(n_spatial_tokens, 1)  # (N,1)

                # Masked mean pooling over patch tokens
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
    out_path = out_dir / f"features_{model_slug}_masked_{dataset_name}_patch{patch_size}.npz"  # save name 

    np.savez_compressed(
        out_path,
        features=features,
        metadata=metadata_arr,
    )

    print("Saved masked features")
    


if __name__ == "__main__":
    main()
