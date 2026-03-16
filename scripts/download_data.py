#!/usr/bin/env python3
"""
Downloads the slices from 
https://openorganelle.janelia.org/faq#python

Slice sampling: Random z-indices
"""

import argparse
import logging
from pathlib import Path

import dask.array as da
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
def get_store(dataset: str):
    """Open N5 root group for dataset on S3 using N5FSStore (anon read). """
    
    url = f"{S3_BASE}/{dataset}/{dataset}.n5"
    store = N5FSStore(url, anon=True)
    return zarr.open(store, mode="r")


def download_slices(
    dataset: str,
    output_dir: Path,
    num_slices: int = 200,
    random: bool = True,
    seed: int | None = 42,
 ) -> list[Path]:
    """
    Download randomle sampled EM slices from one dataset.
    """
    em_dir = output_dir / dataset
    em_dir.mkdir(parents=True, exist_ok=True)

    root = get_store(dataset)

    # EM volume (full resolution)
    
    em = root[EM_PATH]
    

    z_arr = da.from_array(em, chunks=em.chunks)
    nz, ny, nx = z_arr.shape

    if random:
        rng = np.random.default_rng(seed)                    # random x slices 
        z_indices = rng.choice(nz, size=min(num_slices, nz), replace=False)
        z_indices.sort()  
        log.info("downloading slices")
    else:                                                    # top x slices
        z_indices = np.arange(min(num_slices, nz))
        log.info("donwloading slices")

    em_saved = []

    for i, z in enumerate(z_indices):
        slice_arr = z_arr[z, :, :].compute()
        em_path = em_dir / f"slice_{z:05d}.npy"
        np.save(em_path, slice_arr)
        em_saved.append(em_path)

    return em_saved


def main():
    parser = argparse.ArgumentParser()
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
        # Default to <project_root>/data/em_slices regardless of cwd
        default=Path(__file__).resolve().parents[1] / "data" / "em_slices",
        help="Output directory",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    # main loop
    for ds in args.datasets:
        try:
            em_paths = download_slices(
                ds,
                args.output_dir,
                args.num_slices,
                random=not args.no_random,
                seed=args.seed,
            )
            total += len(em_paths)
        except Exception as e:
            log.error("Failed %s: %s", ds, e)
            raise SystemExit(1) from e

    log.info("Done.")


if __name__ == "__main__":
    main()
