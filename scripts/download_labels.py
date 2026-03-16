#!/usr/bin/env python3
"""
Download mito labels (mito_seg) for existing EM slices.
"""

import argparse
from pathlib import Path

import numpy as np
import zarr
from zarr.n5 import N5FSStore

S3_BASE = "s3://janelia-cosem-datasets"
LABEL_PATH = "labels/mito_seg/s0"


def get_label_array(dataset: str):
    """Open mito label array for one dataset"""
    url_n5 = f"{S3_BASE}/{dataset}/{dataset}.n5"
    store = N5FSStore(url_n5, anon=True)
    return zarr.open(store, mode="r", path=LABEL_PATH)


def download_labels_for_dataset(dataset: str, em_dir: Path) -> int:
    """Download labels for existing EM slices"""
    existing = sorted(em_dir.glob("slice_*.npy"))
    if not existing:
        print(f"Skipping {dataset}: no EM slices in {em_dir}")
        return 0

    # get the z indices for the existing images 
    z_indices = [int(f.stem.split("_")[1]) for f in existing]
    label_arr = get_label_array(dataset)
    label_dir = em_dir / "labels"                   # save in labaels subdir 
    label_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, z in enumerate(z_indices):
        label_slice = np.asarray(label_arr[z, :, :])
        out_path = label_dir / f"slice_{z:05d}.npy"
        np.save(out_path, label_slice)
        saved += 1
        # log
        if (i + 1) % 20 == 0 or (i + 1) == len(z_indices):
            print(f"{i + 1}/{len(z_indices)}")

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["jrc_hela-2", "jrc_macrophage-2"],
        help="Dataset names",
    )
    parser.add_argument(
        "-o",
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "em_slices",
        help="Base dir containing dataset subdirs (e.g. data/em_slices)",
    )
    args = parser.parse_args()

    total = 0
    # main loop
    for ds in args.datasets:
        em_dir = args.data_dir / ds
        if not em_dir.exists():
            print(f"Skipping {ds}: {em_dir} does not exist")
            continue
        try:
            n = download_labels_for_dataset(ds, em_dir)
            total += n
        except Exception as e:
            print(f"Failed {ds}: {e}")
            raise SystemExit(1) from e

    print("Done.")


if __name__ == "__main__":
    main()
