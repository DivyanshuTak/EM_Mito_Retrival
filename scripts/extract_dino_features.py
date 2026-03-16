#!/usr/bin/env python3
"""
DINO feature extraction for EM patches.

Loads grayscale patches from .npz, converts to 3-channel for DINO,
extract CLS token features, and saves to .npz.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel

from dataset_utils import EMPatchDataset, collate_image_meta


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
    dl_cfg = cfg["dataloader"]
    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    ds = EMPatchDataset(paths["input_npz"])
    loader = DataLoader(
        ds,
        batch_size=dl_cfg["batch_size"],
        num_workers=dl_cfg["num_workers"],
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_image_meta,
    )

    features_list = []
    metadata_list = []

    with torch.inference_mode():
        for batch_imgs, batch_meta in loader:
            imgs_np = batch_imgs.permute(0, 2, 3, 1).numpy()
            mx = max(imgs_np.max(), 1e-6)
            if mx > 255:
                imgs_np = (imgs_np / mx * 255).astype(np.uint8)
            elif mx <= 1.0 and mx > 0:
                imgs_np = (imgs_np * 255).astype(np.uint8)
            else:
                imgs_np = imgs_np.astype(np.uint8)
            inputs = processor(images=list(imgs_np), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            cls_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()     # cls token 

            features_list.append(cls_features)
            for m in batch_meta:
                metadata_list.append(m)

    features = np.concatenate(features_list, axis=0)
    metadata_arr = np.empty(len(metadata_list), dtype=object)
    metadata_arr[:] = metadata_list

    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_size = cfg.get("patch_size", 256)
    dataset_name = cfg.get("dataset_name", "dataset")
    model_slug = model_name.split("/")[-1].replace("-", "_")
    out_path = out_dir / f"features_background_{model_slug}_patch{patch_size}_{dataset_name}.npz"    # save name 

    np.savez_compressed(
        out_path,
        features=features,
        metadata=metadata_arr,
    )
    print("Saved cls features")
    


if __name__ == "__main__":
    main()
