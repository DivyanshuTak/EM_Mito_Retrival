#!/usr/bin/env python3
"""
Visualize top-K image retrieval results for EM patch embeddings.

"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset_utils import (
    build_db_coord_map,
    get_patch_key,
    is_background,
    load_feature_image_database,
    load_feature_image_dataset,
    prepare_image_for_plot,
)


# ============================================================================
# Configuration
# ============================================================================
QUERY_FEATURE_NPZ = "/media/sdb/divyanshu/divyanshu/hhmi/results/features_UNI_masked_jrc_macrophage2_patch256.npz"
QUERY_IMAGE_NPZ = "/media/sdb/divyanshu/divyanshu/hhmi/data/patches/jrc_macrophage2_mito_dataset_256.npz"
DB_DATASETS = [
    {
        "feature_npz": "/media/sdb/divyanshu/divyanshu/hhmi/results/features_UNI_masked_jrc_hela2_patch256.npz",
        "image_npz": "/media/sdb/divyanshu/divyanshu/hhmi/data/patches/jrc_hela2_mito_dataset_256.npz",
    },
    #{  # uncomment for cls 
    #    "feature_npz": "/media/sdb/divyanshu/divyanshu/hhmi/results/features_background_UNI_cls_jrc_hela2_patch256.npz",
    #    "image_npz": "/media/sdb/divyanshu/divyanshu/hhmi/data/patches/jrc_hela2_background_256.npz",
    #},
]

K = 5
NUM_QUERIES = 5
RANDOM_SEED = 0
OUTPUT_PREFIX = "Uni_maskmeanpooled_macrophage2query_hela2database"


def main() -> None:

    # load features and metadata 
    query_features_np, query_metadata, query_images = load_feature_image_dataset(
        QUERY_FEATURE_NPZ,
        QUERY_IMAGE_NPZ,
    )
    db_features_np, db_metadata, db_images = load_feature_image_database(DB_DATASETS)

    valid_query_indices = [
        idx for idx, meta in enumerate(query_metadata) if not is_background(meta)
    ]
    

    num_queries = min(NUM_QUERIES, len(valid_query_indices))
    rng = random.Random(RANDOM_SEED)
    sampled_query_indices = rng.sample(valid_query_indices, k=num_queries)

    query_features = torch.from_numpy(query_features_np).float()
    db_features = torch.from_numpy(db_features_np).float()

    # normalize to bring onto unit hypersphere 
    query_features = F.normalize(query_features, p=2, dim=-1)
    db_features = F.normalize(db_features, p=2, dim=-1)

    # cosine similarity 
    sim_matrix = query_features @ db_features.T

    # mask self match
    db_coord_map = build_db_coord_map(db_metadata)
    for query_idx, meta in enumerate(query_metadata):
        key = get_patch_key(meta)
        if key is None:
            continue
        db_matches = db_coord_map.get(key, [])
        if not db_matches:
            continue
        sim_matrix[query_idx, db_matches] = -1.0

    topk = min(K, db_features.shape[0])
    if topk == 0:
        raise ValueError("Database is empty.")

    ## plotting 
    for query_idx in sampled_query_indices:
        scores, db_indices = torch.topk(sim_matrix[query_idx], k=topk, largest=True)
        scores_np = scores.cpu().numpy()
        db_indices_np = db_indices.cpu().numpy()

        fig, axes = plt.subplots(1, topk + 1, figsize=(3 * (topk + 1), 3.2))

        query_img, query_cmap = prepare_image_for_plot(query_images[query_idx])
        axes[0].imshow(query_img, cmap=query_cmap)
        axes[0].set_title("Query")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        for rank, (db_idx, score) in enumerate(zip(db_indices_np, scores_np), start=1):
            retrieved_img, retrieved_cmap = prepare_image_for_plot(db_images[db_idx])
            axes[rank].imshow(retrieved_img, cmap=retrieved_cmap)
            label = "Mito" if not is_background(db_metadata[db_idx]) else "Background"
            axes[rank].set_title(f"Sim: {score:.2f}\n{label}")
            axes[rank].set_xticks([])
            axes[rank].set_yticks([])

        plt.tight_layout()
        output_path = Path(
            f"/media/sdb/divyanshu/divyanshu/hhmi/results/retrival_viz/{OUTPUT_PREFIX}_query_{query_idx}.png"   # save name 
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved at {output_path}")


if __name__ == "__main__":
    main()
