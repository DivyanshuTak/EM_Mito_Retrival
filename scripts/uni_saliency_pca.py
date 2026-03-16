#!/usr/bin/env python3
"""
PCA-RGB semantic maps for UNI
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import matplotlib.pyplot as plt
import yaml
from sklearn.decomposition import PCA
from torchvision import transforms


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_meta(m):
    return m if isinstance(m, dict) else m.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--config",
        type=Path,
        default="/media/sdb/divyanshu/divyanshu/hhmi/config/pathology.yaml",
        help="Path to pathology config YAML",
    )
    ap.add_argument(
        "-n",
        "--num-patches",
        type=int,
        default=5,
        help="Number of patches to visualize",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    model_slug = model_name.split("/")[-1].replace("-", "_")
    dataset_name = cfg.get("dataset_name", "dataset")
    patch_size = cfg.get("patch_size", 256)
    input_npz = cfg["paths"]["input_npz"]
    output_dir = Path(cfg["paths"]["saliency_output_dir"] )
    device = torch.device(cfg["device"])
    num_patches = args.num_patches

    name_prefix = f"pca_rgb_{model_slug}_{dataset_name}_patch{patch_size}" # save name

    # same norm
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    # spin up model 
    model = timm.create_model(
        f"hf-hub:{model_name}",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    ).to(device).eval()

    data = np.load(input_npz, allow_pickle=True)
    images = data["images"]
    masks = data["masks"]
    metadata = data["metadata"]
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images.squeeze(-1)

    n = min(num_patches, len(images))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(images), size=n, replace=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    batch_imgs = []
    batch_meta = []
    for i in indices:
        img = images[i]
        x = torch.from_numpy(np.asarray(img, dtype=np.float32))
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.expand(3, -1, -1)
        img_np = x.permute(1, 2, 0).numpy()
        mx = max(img_np.max(), 1e-6)
        if mx > 255:
            img_np = (img_np / mx * 255).astype(np.uint8)
        elif mx <= 1.0 and mx > 0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        batch_imgs.append(img_np)
        batch_meta.append(load_meta(metadata[i]))

    tensors = []
    for img_np in batch_imgs:
        tensors.append(transform(img_np))
    img_tensor = torch.stack(tensors).to(device)

    with torch.no_grad():
        tokens = model.forward_features(img_tensor)
        patch_tokens = tokens[:, 1:, :].cpu().numpy()

    batch_sz, n_tokens, hidden_dim = patch_tokens.shape
    grid_size = int(np.sqrt(n_tokens))

    tokens_flat = patch_tokens.reshape(-1, hidden_dim)
    pca = PCA(n_components=3)
    pca_components = pca.fit_transform(tokens_flat)
    pca_min = pca_components.min(axis=0)
    pca_max = pca_components.max(axis=0)
    pca_range = np.where(pca_max - pca_min > 1e-8, pca_max - pca_min, 1.0)
    pca_scaled = (pca_components - pca_min) / pca_range
    pca_rgb = pca_scaled.reshape(batch_sz, grid_size, grid_size, 3)

    for idx, i in enumerate(indices):
        img = images[i]
        mask = masks[i]
        meta = batch_meta[idx]

        rgb_grid = pca_rgb[idx]
        h, w = img.shape[:2]
        rgb_upscaled = cv2.resize(
            rgb_grid,
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )
        rgb_upscaled = np.clip(rgb_upscaled, 0, 1)

        target_id = meta.get("target_label_id", 1)
        mask_binary = (np.asarray(mask).squeeze() == target_id).astype(np.float32)

        #plotting 
        img_display = np.asarray(img).squeeze()
        if img_display.max() > 255:
            img_display = (img_display / img_display.max() * 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(img_display, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Raw EM")
        axes[0].axis("off")

        axes[1].imshow(img_display, cmap="gray", vmin=0, vmax=255)
        axes[1].imshow(
            np.ma.masked_where(mask_binary < 0.5, mask_binary),
            cmap="Greens",
            alpha=0.5,
        )
        axes[1].set_title("Target mitochondrion")
        axes[1].axis("off")

        axes[2].imshow(rgb_upscaled)
        axes[2].set_title("PCA-RGB (patch features)")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = output_dir / f"{name_prefix}_{idx:03d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved at {out_path}")




if __name__ == "__main__":
    main()
