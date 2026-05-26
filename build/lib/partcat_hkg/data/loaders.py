from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

from partcat_hkg.config import ProjectConfig
from .partimagenet import (
    RoleAwarePartImageNetDataset,
    Stage2ImageOnlyDataset,
    resolve_partimagenet_split_paths,
)
from .collate import collate_part_batch, collate_stage2_image_only


def resolve_partimagenet_split(root: str | Path, split: str, annotation_cfg: str = "", image_root_cfg: str = ""):
    """Compatibility helper returning ``(annotation_json, image_root)`` for one split."""
    paths = resolve_partimagenet_split_paths(root, split, annotation_cfg or None, image_root_cfg or None)
    return paths.annotation_json, paths.image_root


def resolve_partimagenet_annotation(root: str | Path, annotation_cfg: str = "", split: str = "train") -> Path:
    """Compatibility helper used by tests/notebooks to inspect the JSON path."""
    paths = resolve_partimagenet_split_paths(root, split, annotation_cfg or None, None)
    return paths.annotation_json


def resolve_partimagenet_image_root(root: str | Path, image_root_cfg: str = "", split: str = "train") -> Path:
    """Compatibility helper used by tests/notebooks to inspect the image root."""
    paths = resolve_partimagenet_split_paths(root, split, None, image_root_cfg or None)
    return paths.image_root


def make_datasets(cfg: ProjectConfig):
    """Build train/val datasets using the native PartImageNet directory layout.

    The v51 notebook used:
        TRAIN_JSON = PARTIMAGENET_ROOT / "annotations" / "train" / "train.json"
        VAL_JSON   = PARTIMAGENET_ROOT / "annotations" / "val" / "val.json"
        IMG_ROOT_* = PARTIMAGENET_ROOT / "images" / split

    This function mirrors that layout while retaining compatibility with an
    explicit JSON path supplied in the config/CLI.
    """

    root = Path(cfg.paths.partimagenet_root)
    train_paths = resolve_partimagenet_split_paths(
        root,
        "train",
        cfg.paths.train_annotations,
        getattr(cfg.paths, "train_image_root", None),
    )
    val_paths = resolve_partimagenet_split_paths(
        root,
        "val",
        cfg.paths.val_annotations,
        getattr(cfg.paths, "val_image_root", None),
    )

    train = RoleAwarePartImageNetDataset(
        train_paths.annotation_json,
        train_paths.image_root,
        img_size=cfg.data.img_size,
        train=True,
        max_samples=cfg.data.max_train_samples or None,
    )
    val = RoleAwarePartImageNetDataset(
        val_paths.annotation_json,
        val_paths.image_root,
        img_size=cfg.data.img_size,
        train=False,
        schema=train.schema,
        max_samples=cfg.data.max_val_samples or None,
    )
    print(
        "[datasets] train:", train_paths.annotation_json,
        "| images:", train_paths.image_root,
        "| samples:", len(train),
    )
    print(
        "[datasets] val:", val_paths.annotation_json,
        "| images:", val_paths.image_root,
        "| samples:", len(val),
    )
    return train, val


def make_loaders(cfg: ProjectConfig, train_ds, val_ds):
    common = dict(
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_part_batch,
    )
    if cfg.data.num_workers > 0:
        common.update(persistent_workers=cfg.data.persistent_workers, prefetch_factor=cfg.data.prefetch_factor)
    sampler = WeightedRandomSampler(train_ds.sample_weight, len(train_ds.sample_weight), replacement=True)
    stage1_train = DataLoader(train_ds, batch_size=cfg.training.batch_size_stage1, sampler=sampler, **common)
    stage1_val = DataLoader(val_ds, batch_size=cfg.training.batch_size_stage1, shuffle=False, **common)

    if cfg.data.use_stage2_image_only_loader:
        s2_train_ds = Stage2ImageOnlyDataset(train_ds, train=True)
        s2_val_ds = Stage2ImageOnlyDataset(val_ds, train=False)
        s2_common = dict(num_workers=cfg.data.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_stage2_image_only)
        if cfg.data.num_workers > 0:
            s2_common.update(persistent_workers=cfg.data.persistent_workers, prefetch_factor=cfg.data.prefetch_factor)
        stage2_train = DataLoader(s2_train_ds, batch_size=cfg.training.batch_size_stage2, shuffle=True, **s2_common)
        stage2_val = DataLoader(s2_val_ds, batch_size=cfg.training.batch_size_stage2, shuffle=False, **s2_common)
    else:
        stage2_train = DataLoader(train_ds, batch_size=cfg.training.batch_size_stage2, shuffle=True, **common)
        stage2_val = DataLoader(val_ds, batch_size=cfg.training.batch_size_stage2, shuffle=False, **common)
    return stage1_train, stage1_val, stage2_train, stage2_val
