#!/usr/bin/env python3
"""
Shared dataset + collate helpers for DINO and UNI scripts 
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

INSTANCE_KEY = "target_label_id"
DATASET_KEY = "dataset_name"
SLICE_FILENAME_KEY = "slice_filename"
CENTER_X_KEY = "center_x"
CENTER_Y_KEY = "center_y"
IS_BACKGROUND_KEY = "is_background"



#------------------------------------------------
# HELPERS FOR FEATURE EXTRACTION SCRIPTS 
#------------------------------------------------

def _to_meta_dict(meta_obj: Any) -> dict:
    return dict(meta_obj.item() if hasattr(meta_obj, "item") else meta_obj)


def _to_3ch_float_tensor(img: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(img, dtype=np.float32))
    if x.ndim == 2:
        x = x.unsqueeze(0)
    return x.expand(3, -1, -1)


class EMPatchDataset(Dataset):
    """Dataset for EM patches from .npz. Returns (image, metadata)."""

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
        meta = _to_meta_dict(self.metadata[idx])
        return _to_3ch_float_tensor(img), meta


class EMPatchMaskDataset(Dataset):
    """Dataset for EM patches from .npz. Returns (image, mask, metadata)."""

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path, allow_pickle=True)
        self.images = data["images"]
        self.masks = data["masks"]
        self.metadata = data["metadata"]
        if self.images.ndim == 4 and self.images.shape[-1] == 1:
            self.images = self.images.squeeze(-1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        img = self.images[idx]
        mask = self.masks[idx]
        meta = _to_meta_dict(self.metadata[idx])
        img_t = _to_3ch_float_tensor(img)
        mask_t = torch.from_numpy(np.asarray(mask, dtype=np.float32)).squeeze()
        return img_t, mask_t, meta


def collate_image_meta(
    batch: list[tuple[torch.Tensor, dict]],
) -> tuple[torch.Tensor, list[dict]]:
    imgs = torch.stack([b[0] for b in batch], dim=0)
    metas = [b[1] for b in batch]
    return imgs, metas


def collate_image_mask_meta(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    imgs = torch.stack([b[0] for b in batch], dim=0)
    masks = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return imgs, masks, metas


def to_py_dict(obj: Any) -> Dict[str, Any]:
    """Ensure metadata entries are plain Python dicts."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "item"):
        value = obj.item()
        if isinstance(value, dict):
            return value
    raise TypeError(f"Unsupported metadata entry type: {type(obj)}")



#-=------------------------------------------
# HELPERS FOR RETRIVAL VIZ SCRIPT 
#-------------------------------------------


def load_feature_image_dataset(
    feature_npz_path: str | Path,
    image_npz_path: str | Path,
) -> tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
    """Load features, metadata, and images for one dataset split."""
    feature_path = Path(feature_npz_path)
    image_path = Path(image_npz_path)

    with np.load(feature_path, allow_pickle=True) as feature_data:
        features = np.asarray(feature_data["features"])
        metadata = [to_py_dict(item) for item in feature_data["metadata"]]

    with np.load(image_path, allow_pickle=True) as image_data:
        images = np.asarray(image_data["images"])

        if "metadata" in image_data:
            image_metadata = [to_py_dict(item) for item in image_data["metadata"]]

    return features, metadata, images


def load_feature_image_database(
    db_datasets: List[Dict[str, str]],
) -> tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
    """Load and concatenate multiple database datasets into one search space"""
    

    feature_blocks = []
    metadata_blocks: List[Dict[str, Any]] = []
    image_blocks = []

    for entry in db_datasets:
        features, metadata, images = load_feature_image_dataset(
            entry["feature_npz"],
            entry["image_npz"],
        )
        feature_blocks.append(features)
        metadata_blocks.extend(metadata)
        image_blocks.append(images)

    return (
        np.concatenate(feature_blocks, axis=0),
        metadata_blocks,
        np.concatenate(image_blocks, axis=0),
    )


def is_background(meta: Dict[str, Any]) -> bool:
    """Return True if a patch is background."""
    if IS_BACKGROUND_KEY in meta:
        try:
            return bool(meta[IS_BACKGROUND_KEY])
        except Exception:
            pass

    inst = meta.get(INSTANCE_KEY)
    if inst is None:
        return True

    try:
        return int(inst) == 0
    except (TypeError, ValueError):
        return False


def get_patch_key(meta: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """Return a unique spatial identifier for a patch, or None if unavailable."""
    dataset_name = meta.get(DATASET_KEY)
    slice_filename = meta.get(SLICE_FILENAME_KEY)
    center_x = meta.get(CENTER_X_KEY)
    center_y = meta.get(CENTER_Y_KEY)
    if dataset_name is None or slice_filename is None or center_x is None or center_y is None:
        return None
    return (dataset_name, slice_filename, center_x, center_y)


def build_db_coord_map(db_metadata: List[Dict[str, Any]]) -> Dict[Tuple[Any, ...], List[int]]:
    """Map spatial patch keys to database indices."""
    coord_map: Dict[Tuple[Any, ...], List[int]] = {}
    for idx, meta in enumerate(db_metadata):
        key = get_patch_key(meta)
        if key is None:
            continue
        coord_map.setdefault(key, []).append(idx)
    return coord_map


def prepare_image_for_plot(image: np.ndarray) -> tuple[np.ndarray, Optional[str]]:
    """Convert an image array into a Matplotlib-friendly format."""
    arr = np.asarray(image)

    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim == 2:
        return arr, "gray"

    return arr, None
