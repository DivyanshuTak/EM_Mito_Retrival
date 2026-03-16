#!/usr/bin/env python3
"""
Retrieval evaluation for EM patch embeddings.

- Separate query and database feature files (.npz with `features`, `metadata`)
- L2-normalized cosine similarity 
- Two modes:
  * semantic: positive = mitochondrion vs. background
  * instance: positive = same instance ID, with frequency filter N>=2 in DB
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# Default metadata keys used in this project
INSTANCE_KEY = "target_label_id"  # instance ID for mitochondria
DATASET_KEY = "dataset_name"
SLICE_FILENAME_KEY = "slice_filename"
CENTER_X_KEY = "center_x"
CENTER_Y_KEY = "center_y"
IS_BACKGROUND_KEY = "is_background"


def to_py_dict(obj: Any) -> Dict[str, Any]:
    """make metadata entries plain python dicts"""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "item"):
        v = obj.item()
        if isinstance(v, dict):
            return v
    


def load_npz_features(path: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Load (features, metadata_list) from the .npz file"""
    data = np.load(path, allow_pickle=True)
    feats = data["features"]
    meta_arr = data["metadata"]
    metadata: List[Dict[str, Any]] = []
    for m in meta_arr:
        try:
            md = to_py_dict(m)
        except TypeError:
            continue
        metadata.append(md)
    return feats, metadata


def is_background(meta: Dict[str, Any]) -> bool:
    """Return true if a patch is background (non-mito)"""
    if IS_BACKGROUND_KEY in meta:
        try:
            return bool(meta[IS_BACKGROUND_KEY])
        except Exception:
            pass
    inst = meta.get(INSTANCE_KEY, None)
    if inst is None:
        return True
    try:
        return int(inst) == 0
    except (TypeError, ValueError):
        return False


def get_patch_key(meta: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """
    Return a unique key for this patch, or None.
    Queries are always mito patchesm use (dataset_name, slice_filename, center_x, center_y).
    """
    ds = meta.get(DATASET_KEY, None)
    sf = meta.get(SLICE_FILENAME_KEY, None)
    cx = meta.get(CENTER_X_KEY, None)
    cy = meta.get(CENTER_Y_KEY, None)
    if ds is None or sf is None or cx is None or cy is None:
        return None
    return (ds, sf, cx, cy)


def build_self_retrieval_map(
    db_metadata: List[Dict[str, Any]],
) -> Dict[Tuple[Any, ...], List[int]]:
    """
    Build mapping from patch key -> list of database indices.
    Key is (dataset_name, slice_filename, center_x, center_y). 
    """
    coord_map: Dict[Tuple[Any, ...], List[int]] = defaultdict(list)
    for idx, m in enumerate(db_metadata):
        key = get_patch_key(m)
        if key is None:
            continue
        coord_map[key].append(idx)
    return coord_map

# mask the score of the same patch for retrieval 
def mask_self_retrievals(   
    sim_matrix: torch.Tensor,
    query_metadata: List[Dict[str, Any]],
    coord_map: Dict[Tuple[Any, ...], List[int]],
) -> None:
    """In-place: set similarity to -1.0 where query and DB patch keys exactly match."""
    if not coord_map:
        return
    for qi, m in enumerate(query_metadata):
        key = get_patch_key(m)
        if key is None:
            continue
        db_idxs = coord_map.get(key)
        if not db_idxs:
            continue
        sim_matrix[qi, db_idxs] = -1.0


def average_precision(relevant: np.ndarray) -> float:
    """
    Compute AP from a binary relevance array over a ranking.
    relevant[i] = 1 if item at rank i is relevant, else 0.
    """
    assert relevant.ndim == 1
    n_rel = int(relevant.sum())
    if n_rel == 0:
        return 0.0
    cumsum = np.cumsum(relevant)
    ranks = np.arange(1, len(relevant) + 1)
    precision_at_i = cumsum / ranks
    ap = float(np.sum(precision_at_i * relevant) / n_rel)
    return ap


def evaluate_semantic(
    sim_matrix: torch.Tensor,
    db_metadata: List[Dict[str, Any]],
    k: int,
) -> Tuple[float, float, int]:
    """
    Semantic retrieval:
    - Queries assumed positive.
    - Database item is positive if not background.
    """
    device = sim_matrix.device
    n_q, n_d = sim_matrix.shape
    k = min(k, n_d)

    db_pos = torch.tensor(
        [not is_background(m) for m in db_metadata],
        dtype=torch.bool,
        device=device,
    )
    n_pos_db = int(db_pos.sum().item())
    if n_pos_db == 0:
        return 0.0, 0.0, 0

    # Full ranking for AP, top-k for precision
    sorted_scores, sorted_idx = torch.sort(sim_matrix, dim=1, descending=True)
    topk_idx = sorted_idx[:, :k]

    precisions: List[float] = []
    aps: List[float] = []

    for qi in range(n_q):
        # Top-k stats
        tk = topk_idx[qi]
        tk_pos = db_pos[tk]
        tp_k = int(tk_pos.sum().item())
        precisions.append(tp_k / float(k))

        # AP over full ranking
        rel_full = db_pos[sorted_idx[qi]].cpu().numpy().astype(np.int32)
        aps.append(average_precision(rel_full))

    mean_p_at_k = float(np.mean(precisions))
    map_score = float(np.mean(aps))
    return map_score, mean_p_at_k, n_q


def evaluate_instance(
    sim_matrix: torch.Tensor,
    query_metadata: List[Dict[str, Any]],
    db_metadata: List[Dict[str, Any]],
    k: int,
    coord_map: Dict[Tuple[Any, ...], List[int]],
) -> Tuple[float, float, int, int]:
    """
    Instance retrieval:
    - Queries assumed positive.
    - True positive = match on INSTANCE_KEY.
    - Only evaluate queries whose instance appears >= 2 times in DB.  Metrics invalidate if only one patch present 
    - Query patch is excluded from ground-truth targets (via coord_map).
    """
    device = sim_matrix.device
    n_q, n_d = sim_matrix.shape
    k = min(k, n_d)

    # Instance frequency in DB (ignore background / null)
    inst_freq: Dict[Any, int] = defaultdict(int)
    db_inst: List[Any] = []
    for m in db_metadata:
        inst = m.get(INSTANCE_KEY, None)
        db_inst.append(inst)
        if inst is None:
            continue
        try:
            inst_int = int(inst)
        except (TypeError, ValueError):
            continue
        if inst_int == 0:
            continue
        inst_freq[inst_int] += 1

    db_inst_tensor = torch.tensor(
        [int(i) if i is not None else -1 for i in db_inst],
        device=device,
        dtype=torch.long,
    )

    sorted_scores, sorted_idx = torch.sort(sim_matrix, dim=1, descending=True)
    topk_idx = sorted_idx[:, :k]

    precisions: List[float] = []
    aps: List[float] = []

    n_total_queries = n_q
    n_valid_queries = 0

    for qi, qm in enumerate(query_metadata):
        inst = qm.get(INSTANCE_KEY, None)
        try:
            inst_int = int(inst)
        except (TypeError, ValueError):
            continue
        if inst_int == 0:
            continue
        if inst_freq.get(inst_int, 0) < 2:
            # skip instances that only appear once in DB
            continue

        n_valid_queries += 1
        # Relevance over DB: same instance ID
        curr_rel_db = (db_inst_tensor == inst_int).clone()

        # Remove query patch from ground-truth (it's masked in sim_matrix)
        key = get_patch_key(qm)
        if key is not None and key in coord_map:
            curr_rel_db[coord_map[key]] = False

        n_rel_db = int(curr_rel_db.sum().item())
        if n_rel_db == 0:
            n_valid_queries -= 1
            continue

        # Top-k
        tk = topk_idx[qi]
        tk_rel = curr_rel_db[tk]
        tp_k = int(tk_rel.sum().item())

        precisions.append(tp_k / float(k))

        # AP over full ranking
        rel_full = curr_rel_db[sorted_idx[qi]].cpu().numpy().astype(np.int32)
        aps.append(average_precision(rel_full))

    if n_valid_queries == 0:
        return 0.0, 0.0, n_total_queries, 0

    mean_p_at_k = float(np.mean(precisions))
    map_score = float(np.mean(aps))
    return map_score, mean_p_at_k, n_total_queries, n_valid_queries


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate global retrieval for EM patch embeddings"
    )
    ap.add_argument(
        "-q",
        "--query",
        type=Path,
        required=True,
        help="Path to .npz file with query features + metadata",
    )
    ap.add_argument(
        "-d",
        "--database",
        type=Path,
        nargs="+",
        required=True,
        help="One or more .npz files forming the database search space",
    )
    ap.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["semantic", "instance"],
        required=True,
        help="Retrieval mode: 'semantic' or 'instance'",
    )
    ap.add_argument(
        "-k",
        type=int,
        default=10,
        help="K for Precision@K (default: 10)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional path to save results as JSON",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"K: {args.k}")
    print(f"Query file: {args.query}")
    print("Database files:")
    for p in args.database:
        print(f"  - {p}")

    # Load query
    q_feats_np, q_meta = load_npz_features(args.query)
    print(f"Loaded query features: {q_feats_np.shape}")

    # Load and concatenate database
    db_feats_list: List[np.ndarray] = []
    db_meta: List[Dict[str, Any]] = []
    for db_path in args.database:
        feats, meta = load_npz_features(db_path)
        db_feats_list.append(feats)
        db_meta.extend(meta)
        print(f"Loaded database chunk {db_path}: {feats.shape}")

    db_feats_np = np.concatenate(db_feats_list, axis=0)
    print(f"Concatenated database features: {db_feats_np.shape}")

    # Move to torch and normalize
    q_feats = torch.from_numpy(q_feats_np).to(device=device, dtype=torch.float32)
    db_feats = torch.from_numpy(db_feats_np).to(device=device, dtype=torch.float32)

    q_feats = F.normalize(q_feats, p=2, dim=-1)
    db_feats = F.normalize(db_feats, p=2, dim=-1)

    # Cosine similarity via matrix multiplication
    print("Computing similarity matrix...")
    sim_matrix = q_feats @ db_feats.t()  # (N_q, N_d)

    # Self-retrieval masking
    coord_map = build_self_retrieval_map(db_meta)
    mask_self_retrievals(sim_matrix, q_meta, coord_map)

    # Evaluation
    if args.mode == "semantic":
        map_score, mean_p_at_k, n_q = evaluate_semantic(
            sim_matrix, db_meta, k=args.k
        )
        n_total_queries = n_q
        n_valid_queries = n_q  # all queries used in semantic mode
    else:
        map_score, mean_p_at_k, n_total_queries, n_valid_queries = (
            evaluate_instance(sim_matrix, q_meta, db_meta, k=args.k, coord_map=coord_map)
        )

    print("=" * 50)
    print("Retrieval Evaluation Summary")
    print("=" * 70)
    print(f"Mode                      : {args.mode}")
    print(f"K                         : {args.k}")
    print(f"Total queries (input)     : {n_total_queries}")
    print(f"Valid queries (used)      : {n_valid_queries}")
    print(f"mAP                       : {map_score:.4f}")
    print(f"Mean Precision@{args.k:<3}     : {mean_p_at_k:.4f}")

    # save the output to jsom 
    if args.output is not None:
        results = {
            "config": {
                "query": str(args.query),
                "database": [str(p) for p in args.database],
                "mode": args.mode,
                "k": args.k,
                "total_queries": n_total_queries,
                "valid_queries": n_valid_queries,
                "device": str(device),
            },
            "metrics": {
                "mAP": map_score,
                f"Precision@{args.k}": mean_p_at_k,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()

