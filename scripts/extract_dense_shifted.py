#!/usr/bin/env python3
"""
Dense per-pixel embeddings via sliding-window aggregation.

Uses stride=4 for both UNI (timm) and DINOv3 (HuggingFace). 
Output shape: (Batch, 256, 256, D).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel

VIT_PATCH_SIZE = 16
PADDING = 16
STRIDE = 4
SHIFTS = [0, 4, 8, 12]


class EMPatchDataset(Dataset):
    """Dataset for EM patches from .npz. 
    Returns (image, metadata)."""

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path, allow_pickle=True)
        self.images = data["images"]
        self.metadata = data["metadata"]
        if self.images.ndim == 4 and self.images.shape[-1] == 1:
            self.images = self.images.squeeze(-1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img = self.images[idx]
        m = self.metadata[idx]
        meta = dict(m.item() if hasattr(m, "item") else m)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32))
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.expand(3, -1, -1)
        return x, meta

# custom collate
def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, list[dict]]:
    imgs = torch.stack([b[0] for b in batch])
    metas = [b[1] for b in batch]
    return imgs, metas


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_sharded_features(meta_path: str | Path):
    """
    Load sharded dense features. 
    Returns (loader, metadata).
    """
    meta = np.load(meta_path, allow_pickle=True)
    metadata = meta["metadata"]
    shape = tuple(meta["shape"])
    shard_offsets = meta["shard_offsets"]
    shards_dir = Path(meta["shards_dir"].item() if meta["shards_dir"].ndim == 0 else str(meta["shards_dir"]))

    class ShardedLoader:
        def __len__(self):
            return shape[0]

        def __getitem__(self, idx):
            shard_id = np.searchsorted(shard_offsets, idx, side="right") - 1
            local_idx = idx - shard_offsets[shard_id]
            shard = np.load(shards_dir / f"shard_{shard_id:04d}.npy")
            return shard[local_idx]

        def iter_shards(self):
            #Yield (features_array, start_idx, end_idx) for each shard use for pooling without loading all
            for i in range(len(shard_offsets) - 1):
                arr = np.load(shards_dir / f"shard_{i:04d}.npy")
                yield arr, int(shard_offsets[i]), int(shard_offsets[i + 1])

        def concatenate(self):
            #Load all shards into one array, might run into OOM
            return np.concatenate([np.load(p) for p in sorted(shards_dir.glob("shard_*.npy"))], axis=0)

    return ShardedLoader(), metadata


def create_gaussian_weight(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    #Create a 2D Gaussian tile of shape (size, size), centered, for patch weighting
    half = (size - 1) / 2.0
    grid = torch.arange(size, device=device, dtype=torch.float32)
    x = grid.unsqueeze(0).expand(size, size)
    y = grid.unsqueeze(1).expand(size, size)
    g = torch.exp(-((x - half) ** 2 + (y - half) ** 2) / (2 * sigma**2))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--config",
        type=Path,
        default="/media/sdb/divyanshu/divyanshu/hhmi/config/feature_extraction.yaml",
        help="Path to config (feature_extraction.yaml for DINOv3, pathology.yaml for UNI)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    model_name = cfg["model"]["name"]
    model_slug = model_name.split("/")[-1].replace("-", "_")
    dl_cfg = cfg["dataloader"]
    device = torch.device(cfg["device"])
    img_size = cfg.get("patch_size", 256)
    dataset_name = cfg.get("dataset_name", "dataset")

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Image size: {img_size}x{img_size}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    is_uni = "UNI" in model_name or "MahmoodLab" in model_name

    if is_uni:
        import timm
        model = timm.create_model(
            f"hf-hub:{model_name}",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        ).to(device).eval()
        hidden_dim = model.embed_dim
    else:
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        hidden_dim = model.config.hidden_size

    
    # Use 16 for dense extraction (matches UNI)
    padded_size = img_size + 2 * PADDING
    batch_size = 16

    ds = EMPatchDataset(paths["input_npz"])
    nw = dl_cfg.get("dense_num_workers", 0) 
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=nw,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    grid_size = padded_size // VIT_PATCH_SIZE

    gaussian_tile = create_gaussian_weight(VIT_PATCH_SIZE, sigma=4.0, device=device)
    weight_map = gaussian_tile.unsqueeze(0).unsqueeze(0).repeat(
        1, 1, grid_size, grid_size
    )
    weight_map = F.interpolate(
        weight_map,
        size=(padded_size, padded_size),
        mode="bilinear",
        align_corners=False,
    )

    total_samples = len(ds)
    all_metadata = []
    shard_size = cfg.get("shard_size", 32)  # samples per shard, write as they come to save memory 

    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / f"dense_features_shifted_{model_slug}_{dataset_name}_p{img_size}_shards"
    shards_dir.mkdir(exist_ok=True)

    shard_buf = []
    shard_meta_buf = []
    shard_offsets = [0]
    shard_idx = 0

    def flush_shard():
        nonlocal shard_idx
        if not shard_buf:
            return
        arr = np.concatenate(shard_buf, axis=0)
        shard_path = shards_dir / f"shard_{shard_idx:04d}.npy"
        np.save(shard_path, arr)
        shard_offsets.append(shard_offsets[-1] + arr.shape[0])
        all_metadata.extend(shard_meta_buf)
        shard_idx += 1
        shard_buf.clear()
        shard_meta_buf.clear()

    with torch.inference_mode():
        for batch_idx, (batch_imgs, batch_meta) in enumerate(loader):
            imgs_np = batch_imgs.permute(0, 2, 3, 1).numpy()
            mx = max(imgs_np.max(), 1e-6)
            if mx > 255:
                imgs_np = (imgs_np / mx * 255).astype(np.uint8)
            elif mx <= 1.0 and mx > 0:
                imgs_np = (imgs_np * 255).astype(np.uint8)
            else:
                imgs_np = imgs_np.astype(np.uint8)

            tensors = []
            for j in range(imgs_np.shape[0]):
                tensors.append(transform(imgs_np[j]))
            img_tensor = torch.stack(tensors).to(device)

            padded = F.pad(img_tensor, (PADDING,) * 4, mode="reflect")

            B, _, Hp, Wp = padded.shape

            sum_features = torch.zeros(B, hidden_dim, Hp, Wp, device=device, dtype=torch.float32)
            sum_weights = torch.zeros(B, 1, Hp, Wp, device=device, dtype=torch.float32)

            for dy in SHIFTS:
                for dx in SHIFTS:
                    shifted = torch.roll(padded, shifts=(dy, dx), dims=(2, 3))

                    if is_uni:
                        tokens = model.forward_features(shifted)[:, 1:, :]
                    else:
                        outputs = model(pixel_values=shifted)
                        tokens = outputs.last_hidden_state[:, 5:, :]

                    n_tokens = tokens.shape[1]
                    gh = gw = int(n_tokens**0.5)
                    feat_grid = tokens.reshape(B, gh, gw, -1).permute(0, 3, 1, 2)

                    upsampled = F.interpolate(
                        feat_grid,
                        size=(Hp, Wp),
                        mode="bilinear",
                        align_corners=False,
                    )

                    aligned = torch.roll(upsampled, shifts=(-dy, -dx), dims=(2, 3))

                    w = weight_map.expand(B, -1, -1, -1)
                    sum_features = sum_features + aligned * w
                    sum_weights = sum_weights + w

            avg_features = sum_features / (sum_weights + 1e-8)

            cropped = avg_features[:, :, PADDING : PADDING + img_size, PADDING : PADDING + img_size]
            cropped = cropped.permute(0, 2, 3, 1).cpu().numpy()

            B = cropped.shape[0]
            shard_buf.append(cropped)
            shard_meta_buf.extend(batch_meta)
            if sum(a.shape[0] for a in shard_buf) >= shard_size:
                flush_shard()

            print(f"  Batch {batch_idx + 1}/{len(loader)}: {cropped.shape}")

    flush_shard()  # write remaining

    metadata_arr = np.empty(len(all_metadata), dtype=object)
    metadata_arr[:] = all_metadata

    meta_path = out_dir / f"dense_features_shifted_{model_slug}_{dataset_name}_p{img_size}_meta.npz"
    features_shape = (total_samples, img_size, img_size, hidden_dim)
    shard_offsets_arr = np.array(shard_offsets)
    np.savez_compressed(
        meta_path,
        metadata=metadata_arr,
        shape=np.array(features_shape),
        shard_offsets=shard_offsets_arr,
        shards_dir=str(shards_dir),
    )

    print(f"Saved {total_samples} dense features in {shard_idx} shards to {shards_dir}")



if __name__ == "__main__":
    main()
