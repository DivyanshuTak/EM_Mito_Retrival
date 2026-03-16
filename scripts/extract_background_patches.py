#!/usr/bin/env python3
"""
Extract 256x256 background patches (hard negatives) from EM slices.

Samples patches where the segmentation mask is completely empty (zero mitochondria).
Output format matches extract_mito_patches.py: images, masks, metadata.
"""

# -----------------------------------------------------------------------------
# CONFIGURATION 
# -----------------------------------------------------------------------------
DATA_DIR = "/media/sdb/divyanshu/divyanshu/hhmi/data/em_slices"
DATASETS = ["jrc_hela-2"]
PATCH_SIZE = 256
NUM_PATCHES = 527  # 626 for hela2 # 527 for macrophage
OUTPUT_PATH = "/media/sdb/divyanshu/divyanshu/hhmi/data/patches/jrc_hela2_background_256.npz"
# -----------------------------------------------------------------------------

import argparse
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np


def get_slices_with_both(data_dir: Path, datasets: list[str]) -> list[tuple[str, Path, Path]]:
    """Return list of (dataset_name, em_path, mask_path) for slices that have both."""
    out = []
    for ds in datasets:
        em_dir = data_dir / ds
        label_dir = em_dir / "labels"
        if not label_dir.exists():
            continue
        em_names = {p.name for p in em_dir.glob("slice_*.npy")}
        for p in sorted(label_dir.glob("slice_*.npy")):
            if p.name in em_names:
                out.append((ds, em_dir / p.name, p))
    return out


def extract_patch(img: np.ndarray, cy: float, cx: float, size: int, pad_val: int | float = 0) -> np.ndarray:
    """Extract size x size patch centered at (cy, cx). Pad with pad_val if out of bounds or on the edge"""
    half = size // 2
    y0, y1 = int(cy) - half, int(cy) + half
    x0, x1 = int(cx) - half, int(cx) + half
    h, w = img.shape

    sy0 = max(0, y0)
    sy1 = min(h, y1)
    sx0 = max(0, x0)
    sx1 = min(w, x1)

    out = np.full((size, size), pad_val, dtype=img.dtype)
    dy0 = sy0 - y0
    dy1 = dy0 + (sy1 - sy0)
    dx0 = sx0 - x0
    dx1 = dx0 + (sx1 - sx0)
    out[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return out


def parse_slice_z(filename: str) -> int:
    """Extract z-index from slice filename (e.g. slice_00707.npy)"""
    m = re.match(r"slice_(\d+)\.npy", filename)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-n", "--num-patches",
        type=int,
        default=NUM_PATCHES,
        help=f"Number of background patches to extract (default: {NUM_PATCHES})",
    )
    ap.add_argument(
        "-o", "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Path to save the output .npz file (default: {OUTPUT_PATH})",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Base dir with dataset/labels/ (default: {DATA_DIR})",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help=f"Dataset subdir names (default: {DATASETS})",
    )
    ap.add_argument(
        "--patch-size",
        type=int,
        default=PATCH_SIZE,
        help=f"Patch size in pixels (default: {PATCH_SIZE})",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    datasets = args.datasets
    patch_size = args.patch_size
    num_patches = args.num_patches
    out_path = Path(args.output)

    slices = get_slices_with_both(data_dir, datasets)
    

    images_list = []
    masks_list = []
    metadata_list = []
    patch_id = 0
    attempts = 0
    max_attempts = num_patches * 1000  # safety limit

    
    
    pbar = tqdm(total=num_patches, desc="Background patches", unit="patch")
    

    rng = np.random.default_rng()

    while patch_id < num_patches:
        attempts += 1
        if attempts > max_attempts:
            print("max attempts reached ")
            break

        ds, em_path, mask_path = slices[rng.integers(0, len(slices))]
        img = np.load(em_path)
        mask = np.load(mask_path)
        H, W = img.shape

        x_max = max(0, W - patch_size)
        y_max = max(0, H - patch_size)

        x = int(rng.integers(0, x_max + 1))
        y = int(rng.integers(0, y_max + 1))

        cx = x + patch_size / 2.0
        cy = y + patch_size / 2.0

        mask_patch = extract_patch(mask, cy, cx, patch_size, pad_val=0)
        if np.any(mask_patch > 0):
            continue

        patch_img = extract_patch(img, cy, cx, patch_size, pad_val=0)
        patch_mask = np.zeros((patch_size, patch_size), dtype=mask.dtype)

        # append images and mask placeholders 
        images_list.append(patch_img)
        masks_list.append(patch_mask)
        z = parse_slice_z(em_path.name)
        metadata_list.append({              # list metadata
            "patch_id": patch_id,
            "dataset_name": ds,
            "slice_filename": em_path.name,
            "center_x": float(cx),
            "center_y": float(cy),
            "target_label_id": 0,
            "is_background": True,
            "z": z,
            "x": x,
            "y": y,
        })
        patch_id += 1
        

    images = np.stack(images_list, axis=0)
    masks = np.stack(masks_list, axis=0)
    metadata_arr = np.empty(len(metadata_list), dtype=object)
    metadata_arr[:] = metadata_list

    # save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        images=images,
        masks=masks,
        metadata=metadata_arr,
    )

    print(f"Saved {len(images_list)} background patches in {out_path}")
    


if __name__ == "__main__":
    main()
