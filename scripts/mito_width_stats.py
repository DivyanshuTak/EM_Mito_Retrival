#!/usr/bin/env python3
"""
Mitochondria max-width statistics from instance segmentation masks.

Computes the maximum extent (larger of bbox height/width) for each mitochondria
across all slices to select patch/tile size.

"""

import argparse
import json
from pathlib import Path

import numpy as np


def max_extent_px(labels: np.ndarray, uid: int) -> float:
    """Max of (height, width) of bounding box for instance uid"""
    mask = labels == uid
    rows, cols = np.where(mask)
    if rows.size == 0:
        return 0.0
    h = int(rows.max()) - int(rows.min()) + 1
    w = int(cols.max()) - int(cols.min()) + 1
    return float(max(h, w))


def widths_from_slice(path: Path) -> list[float]:
    """Extract max extent for each mitochondrion in one label slice"""
    arr = np.load(path)
    ids = np.unique(arr)
    ids = ids[ids > 0]
    return [max_extent_px(arr, i) for i in ids]


def run(data_dir: Path, datasets: list[str]) -> np.ndarray:
    """Compute all mitochondria widths. Returns 1D array of widths in pixels"""
    all_widths: list[float] = []
    for ds in datasets:
        em_dir = data_dir / ds
        label_dir = em_dir / "labels"
        if not label_dir.exists():
            print("Skipping for no labels dir")
            continue
        em_names = {p.name for p in em_dir.glob("slice_*.npy")}
        label_paths = sorted(label_dir.glob("slice_*.npy"))
        valid = [p for p in label_paths if p.name in em_names]
        for p in valid:
            all_widths.extend(widths_from_slice(p))
        
    return np.array(all_widths, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        type=Path,
        default="/media/sdb/divyanshu/divyanshu/hhmi/data/em_slices",
        help="Base dir with dataset/labels/",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["jrc_hela-2", "jrc_macrophage-2"],
        help="Dataset subdir names",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save list of all max-widths (one per mitochondrion) to JSON",
    )
    args = ap.parse_args()

    widths = run(args.data_dir, args.datasets)

    p95 = float(np.percentile(widths, 95)) # 95 percentile 
    print(f"95th percentile  {p95}")
    print(f"Max: {widths.max()}  Min: {widths.min()}  Mean: {widths.mean()}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # save json 
        out = {
            "widths": widths.tolist(),
            "stats": {
                "min": float(widths.min()),
                "max": float(widths.max()),
                "mean": float(widths.mean()),
                "p95": float(np.percentile(widths, 95)),
            },
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print("Saved")


if __name__ == "__main__":
    main()
