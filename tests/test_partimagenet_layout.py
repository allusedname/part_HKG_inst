from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from partcat_hkg.data.partimagenet import (
    RoleAwarePartImageNetDataset,
    resolve_partimagenet_image_path,
    resolve_partimagenet_split_paths,
)


def _write_split(root: Path, split: str, image_id: int, file_name: str) -> None:
    ann_dir = root / "annotations" / split
    img_dir = root / "images" / split / file_name.split("_")[0]
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(120, 130, 140)).save(img_dir / Path(file_name).name)
    data = {
        "images": [{"id": image_id, "file_name": file_name, "width": 16, "height": 16}],
        "categories": [
            {"id": 1, "name": "body", "supercategory": "bird"},
            {"id": 2, "name": "wing", "supercategory": "bird"},
        ],
        "annotations": [
            {
                "id": image_id * 10 + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [2, 2, 8, 8],
                "area": 64,
                "iscrowd": 0,
                "segmentation": [[2, 2, 10, 2, 10, 10, 2, 10]],
            },
            {
                "id": image_id * 10 + 2,
                "image_id": image_id,
                "category_id": 2,
                "bbox": [8, 3, 6, 6],
                "area": 36,
                "iscrowd": 0,
                "segmentation": [[8, 3, 14, 3, 14, 9, 8, 9]],
            },
        ],
    }
    (ann_dir / f"{split}.json").write_text(json.dumps(data), encoding="utf-8")


def test_partimagenet_nested_layout_and_legacy_train_json_hint(tmp_path):
    root = tmp_path / "PartImageNet"
    _write_split(root, "train", 1, "n00000001_1.JPEG")
    _write_split(root, "val", 2, "n00000001_2.JPEG")

    paths = resolve_partimagenet_split_paths(root, "train", "train.json")
    assert paths.annotation_json == root / "annotations" / "train" / "train.json"
    assert paths.image_root == root / "images" / "train"

    ds = RoleAwarePartImageNetDataset(paths.annotation_json, paths.image_root, img_size=16, train=False)
    assert len(ds) == 1
    item = ds[0]
    assert item["image"].shape == (3, 16, 16)
    assert item["part_masks"].shape[0] == ds.schema.num_parts
    assert item["part_masks"].sum().item() > 0
    assert item["role_masks"].sum().item() > 0


def test_image_path_resolver_drops_split_prefix(tmp_path):
    image_root = tmp_path / "PartImageNet" / "images" / "train"
    target = image_root / "n00000001" / "n00000001_3.JPEG"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(target)

    got = resolve_partimagenet_image_path(
        image_root,
        {"file_name": "train/n00000001/n00000001_3.JPEG"},
    )
    assert got == target
