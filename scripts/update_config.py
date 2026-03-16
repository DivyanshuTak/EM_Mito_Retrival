#!/usr/bin/env python3
"""Update config YAML with experimenttion parameters"""

import argparse
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, help="Config file path")
    ap.add_argument("--patch-size", type=int, required=True)
    ap.add_argument("--dataset-name", type=str, required=True)
    ap.add_argument("--input-npz", type=str, required=True)
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["patch_size"] = args.patch_size
    cfg["dataset_name"] = args.dataset_name
    cfg["paths"]["input_npz"] = args.input_npz

    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir
        cfg["paths"]["saliency_output_dir"] = f"{args.output_dir}/saliency"
        cfg["paths"]["pca_output_dir"] = f"{args.output_dir}/pca_rgb"

    with open(args.config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()
