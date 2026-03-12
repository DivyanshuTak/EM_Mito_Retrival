#!/usr/bin/env python3
"""
Task 1: Programmatic download of EM slices and mitochondria labels from OpenOrganelle.

Uses fsspec + zarr + dask per OpenOrganelle FAQ:
https://openorganelle.janelia.org/faq#python

Data is stored as N5/Zarr on S3 (janelia-cosem-datasets).

Slice sampling: Random z-indices for better diversity across the volume
(mitochondria from different regions, morphologies). Contiguous chunks would
bias toward one region.
"""

import argparse
import logging
from pathlib import Path

import dask.array as da
import fsspec
import numpy as np
import zarr
from zarr.n5 import N5FSStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

S3_BASE = "s3://janelia-cosem-datasets"
EM_PATH = "em/fibsem-uint16/s0"  # full-res EM
MITO_PATHS = [
    "volumes/labels/mitochondria",
    "labels/mitochondria",
    "labels/mito",
    "mito/s0",
]  # try in order; datasets vary


def get_store(dataset: str):
    """Open N5 root group for dataset on S3 using N5FSStore (anon read).

    Mirrors the OpenOrganelle FAQ example:
        group = zarr.open(zarr.N5FSStore('s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5', anon=True))
    """
    url = f"{S3_BASE}/{dataset}/{dataset}.n5"
    store = N5FSStore(url, anon=True)
    return zarr.open(store, mode="r")


def _find_array(root, paths: list[str], name: str):
    """Return first existing array from paths, or None. Handles nested keys like volumes/labels/mito."""
    for p in paths:
        try:
            obj = root
            for key in p.split("/"):
                obj = obj[key]
            return obj
        except (KeyError, TypeError):
            continue
    return None


def download_slices(
    dataset: str,
    output_dir: Path,
    num_slices: int = 200,
    random: bool = True,
    seed: int | None = 42,
) -> tuple[list[Path], list[Path] | None]:
    """
    Download EM slices (and mitochondria labels if available) from one dataset.
    Slices are sampled randomly for diversity across the volume.
    Returns (em_paths, label_paths or None).
    """
    em_dir = output_dir / dataset
    label_dir = output_dir / dataset / "labels"
    em_dir.mkdir(parents=True, exist_ok=True)

    root = get_store(dataset)

    em = _find_array(root, [EM_PATH, "em/s0", "images/fibsem-uint16/s0"], "EM")
    if em is None:
        raise KeyError(f"EM data not found in {dataset}")

    mito = _find_array(root, MITO_PATHS, "mitochondria")
    has_labels = mito is not None
    if has_labels:
        label_dir.mkdir(parents=True, exist_ok=True)
    else:
        log.warning("%s: no mitochondria labels found (tried %s)", dataset, MITO_PATHS)

    z_arr = da.from_array(em, chunks=em.chunks)
    nz, ny, nx = z_arr.shape

    if random:
        rng = np.random.default_rng(seed)
        z_indices = rng.choice(nz, size=min(num_slices, nz), replace=False)
        z_indices.sort()  
        log.info("%s: shape %s, downloading %d random slices", dataset, (nz, ny, nx), len(z_indices))
    else:
        z_indices = np.arange(min(num_slices, nz))
        log.info("%s: shape %s, downloading slices [0:%d]", dataset, (nz, ny, nx), len(z_indices))

    em_saved = []
    label_saved = [] if has_labels else None

    for i, z in enumerate(z_indices):
        slice_arr = z_arr[z, :, :].compute()
        em_path = em_dir / f"slice_{z:05d}.npy"
        np.save(em_path, slice_arr)
        em_saved.append(em_path)

        if has_labels:
            label_slice = mito[z, :, :]
            if hasattr(label_slice, "compute"):
                label_slice = label_slice.compute()
            else:
                label_slice = np.asarray(label_slice)
            label_path = label_dir / f"slice_{z:05d}.npy"
            np.save(label_path, label_slice)
            label_saved.append(label_path)

        if (i + 1) % 50 == 0:
            log.info("  %d/%d slices done", i + 1, len(z_indices))

    return em_saved, label_saved


def main():
    parser = argparse.ArgumentParser(description="Download EM slices + labels from OpenOrganelle (Task 1)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["jrc_hela-3", "jrc_macrophage-2"],
        help="Dataset names",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=200,
        help="Slices per dataset",
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Use contiguous slices instead of random",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for slice selection",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data/em_slices"),
        help="Output directory",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for ds in args.datasets:
        try:
            em_paths, label_paths = download_slices(
                ds,
                args.output_dir,
                args.num_slices,
                random=not args.no_random,
                seed=args.seed,
            )
            total += len(em_paths)
            if label_paths:
                log.info("  + %d label slices in %s/labels", len(label_paths), ds)
        except Exception as e:
            log.error("Failed %s: %s", ds, e)
            raise SystemExit(1) from e

    log.info("Done. %d slices saved to %s", total, args.output_dir)


if __name__ == "__main__":
    main()
