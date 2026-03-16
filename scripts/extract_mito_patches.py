#!/usr/bin/env python3
"""
Extract sizexsize patches centered on each mitochondrion from EM slices and masks.
Patches near edges are padded with zeros 
Output: npz files with images, masks, metadata.
"""

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_DIR = "/media/sdb/divyanshu/divyanshu/hhmi/data/em_slices"
DATASETS = ["jrc_macrophage-2"]
OUTPUT_PATH = "/media/sdb/divyanshu/divyanshu/hhmi/data/patches/jrc_macrophage2_mito_dataset_256.npz"
PATCH_SIZE = 256
# -----------------------------------------------------------------------------

import numpy as np
from pathlib import Path

from skimage.measure import regionprops


def get_slices_with_both(data_dir: Path, datasets: list[str]) -> list[tuple[str, Path, Path]]:
    """Yield (dataset_name, em_path, mask_path) for slices that have both"""
    for ds in datasets:
        em_dir = data_dir / ds
        label_dir = em_dir / "labels"
        if not label_dir.exists():
            continue
        em_names = {p.name for p in em_dir.glob("slice_*.npy")}
        for p in sorted(label_dir.glob("slice_*.npy")):
            if p.name in em_names:
                yield ds, em_dir / p.name, p


def extract_patch(img: np.ndarray, cy: float, cx: float, size: int, pad_val: int | float = 0) -> np.ndarray:
    """Extract size x size patch centered at (cy, cx). Pad with pad_val if out of bounds"""
    half = size // 2
    y0, y1 = int(cy) - half, int(cy) + half
    x0, x1 = int(cx) - half, int(cx) + half
    h, w = img.shape

    # Bounds in source image
    sy0 = max(0, y0)
    sy1 = min(h, y1)
    sx0 = max(0, x0)
    sx1 = min(w, x1)

    out = np.full((size, size), pad_val, dtype=img.dtype)
    # Destination region in output (where we put the valid crop)
    dy0 = sy0 - y0
    dy1 = dy0 + (sy1 - sy0)
    dx0 = sx0 - x0
    dx1 = dx0 + (sx1 - sx0)
    out[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return out


def main():
    data_dir = Path(DATA_DIR)
    patch_size = PATCH_SIZE

    images_list = []
    masks_list = []
    metadata_list = []
    patch_id = 0

    for ds, em_path, mask_path in get_slices_with_both(data_dir, DATASETS):
        img = np.load(em_path)
        mask = np.load(mask_path)

        # regionprops on labeled image: 0 = background, each int = one region
        props = regionprops(mask)
        for region in props:
            cy, cx = region.centroid
            patch_img = extract_patch(img, cy, cx, patch_size, pad_val=0)
            patch_mask = extract_patch(mask, cy, cx, patch_size, pad_val=0)

            images_list.append(patch_img)
            masks_list.append(patch_mask)
            metadata_list.append({         #  list metadata 
                "patch_id": patch_id,
                "dataset_name": ds,
                "slice_filename": em_path.name,
                "center_x": float(cx),
                "center_y": float(cy),
                "target_label_id": int(region.label),
            })
            patch_id += 1

    

    images = np.stack(images_list, axis=0)
    masks = np.stack(masks_list, axis=0)
    metadata_arr = np.empty(len(metadata_list), dtype=object)
    metadata_arr[:] = metadata_list

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        images=images,
        masks=masks,
        metadata=metadata_arr,
    )
    print("Saved patches")


if __name__ == "__main__":
    main()
