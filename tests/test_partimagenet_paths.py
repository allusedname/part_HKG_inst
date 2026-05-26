from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from partcat_hkg.config import ProjectConfig
from partcat_hkg.data.loaders import make_datasets
from partcat_hkg.data.partimagenet import (
    RoleAwarePartImageNetDataset,
    resolve_partimagenet_split_paths,
)


def _write_tiny_partimagenet(root: Path, split: str) -> None:
    ann_dir = root / "annotations" / split
    img_dir = root / "images" / split / "n00000001"
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    file_name = "n00000001/n00000001_0001.JPEG"
    Image.new("RGB", (16, 16), (128, 128, 128)).save(img_dir / "n00000001_0001.JPEG")
    payload = {
        "images": [{"id": 1, "file_name": file_name, "width": 16, "height": 16}],
        "categories": [{"id": 1, "name": "car wheel", "supercategory": "Car"}],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [4, 4, 8, 8],
            "segmentation": [[4, 4, 12, 4, 12, 12, 4, 12]],
            "area": 64,
            "iscrowd": 0,
        }],
    }
    (ann_dir / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_resolves_notebook_partimagenet_layout(tmp_path: Path):
    root = tmp_path / "PartImageNet"
    _write_tiny_partimagenet(root, "train")
    resolved = resolve_partimagenet_split_paths(root, "train", "train.json")
    assert resolved.annotation_json == root / "annotations" / "train" / "train.json"
    assert resolved.image_root == root / "images" / "train"


def test_dataset_loads_nested_annotation_and_images(tmp_path: Path):
    root = tmp_path / "PartImageNet"
    _write_tiny_partimagenet(root, "train")
    resolved = resolve_partimagenet_split_paths(root, "train", "annotations/train/train.json")
    ds = RoleAwarePartImageNetDataset(resolved.annotation_json, resolved.image_root, img_size=32, train=False)
    item = ds[0]
    assert len(ds) == 1
    assert item["image"].shape[-2:] == (32, 32)
    assert item["part_masks"].sum() > 0
    assert item["role_masks"].sum() > 0
    assert item["meta"]["obj_name"] == "car"


def test_make_datasets_uses_nested_partimagenet_layout(tmp_path: Path):
    root = tmp_path / "PartImageNet"
    _write_tiny_partimagenet(root, "train")
    _write_tiny_partimagenet(root, "val")
    cfg = ProjectConfig()
    cfg.paths.partimagenet_root = str(root)
    cfg.paths.train_annotations = "annotations/train/train.json"
    cfg.paths.val_annotations = "annotations/val/val.json"
    cfg.paths.train_image_root = "images/train"
    cfg.paths.val_image_root = "images/val"
    cfg.data.img_size = 32
    cfg.data.max_train_samples = 1
    cfg.data.max_val_samples = 1
    train_ds, val_ds = make_datasets(cfg)
    assert len(train_ds) == 1
    assert len(val_ds) == 1
    assert train_ds.samples[0]["img_path"].endswith("n00000001_0001.JPEG")
