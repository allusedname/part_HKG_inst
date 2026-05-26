from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

from .canonicalization import canonicalize_object_name, canonicalize_part_name, role_name
from .schema import RoleSchema
from .transforms import JointImageMaskTransform, ImageOnlyTransform


@dataclass(frozen=True)
class PartImageNetSplitPaths:
    """Resolved paths for one PartImageNet split.

    Native PartImageNet layout used by the v51 notebook:

        PartImageNet/
          annotations/train/train.json
          annotations/val/val.json
          annotations/test/test.json
          annotations/train_whole/train_whole.json
          images/train/...
          images/val/...
          images/test/...

    Stage 1 part-mask training uses annotations/<split>/<split>.json and
    images/<split>. The *_whole annotation folders are object-level annotations
    and are not used for supervised part-mask extraction.
    """

    annotation_json: Path
    image_root: Path


def _unique_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p)
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def _format_tried(paths: list[Path]) -> str:
    return "\n".join(f"  - {p}" for p in _unique_paths(paths))


def _base_split(split: str) -> str:
    split = str(split).strip().strip("/")
    return split[: -len("_whole")] if split.endswith("_whole") else split


def describe_partimagenet_layout(root: str | Path) -> str:
    """Return a readable report for the expected PartImageNet data layout."""
    root = Path(root).expanduser()
    rows = [
        ("root", root),
        ("train annotations", root / "annotations" / "train" / "train.json"),
        ("val annotations", root / "annotations" / "val" / "val.json"),
        ("test annotations", root / "annotations" / "test" / "test.json"),
        ("train images", root / "images" / "train"),
        ("val images", root / "images" / "val"),
        ("test images", root / "images" / "test"),
    ]
    lines = ["PartImageNet layout check:"]
    for label, path in rows:
        lines.append(f"  {label:18s}: {path}  [{'OK' if path.exists() else 'missing'}]")
    lines.append("")
    lines.append("Stage 1 uses annotations/<split>/<split>.json; *_whole folders are not used for part-mask training.")
    return "\n".join(lines)


def _annotation_candidates(root: Path, split: str, annotation_hint: str | Path | None) -> list[Path]:
    split = str(split)
    base = _base_split(split)
    candidates: list[Path] = []
    if annotation_hint:
        hint = Path(annotation_hint).expanduser()
        hinted = hint if hint.is_absolute() else root / hint
        candidates.append(hinted)
        if hinted.is_dir():
            candidates.extend([hinted / f"{split}.json", hinted / f"{base}.json", hinted / "annotations.json"])
            candidates.extend(sorted(hinted.glob("*.json")))
        # Legacy flat config values, e.g. train.json / val.json, should also
        # try the nested notebook/original PartImageNet location.
        if hint.name in {f"{split}.json", f"{base}.json"} and len(hint.parts) == 1:
            candidates.append(root / "annotations" / split / hint.name)
            candidates.append(root / "annotations" / base / f"{base}.json")
    candidates.extend([
        root / "annotations" / split / f"{split}.json",
        root / "annotations" / split / f"{base}.json",
        root / "annotations" / split / "annotations.json",
        root / "annotations" / base / f"{base}.json",
        root / "annotations" / f"{split}.json",
        root / "annotations" / f"{base}.json",
        root / f"{split}.json",
        root / f"{base}.json",
    ])
    return _unique_paths(candidates)


def _image_root_candidates(root: Path, split: str, image_root_hint: str | Path | None = None) -> list[Path]:
    split = str(split)
    base = _base_split(split)
    candidates: list[Path] = []
    if image_root_hint:
        hint = Path(image_root_hint).expanduser()
        candidates.append(hint if hint.is_absolute() else root / hint)
    candidates.extend([
        root / "images" / base,
        root / "images" / split,
        root / "Images" / base,
        root / "Images" / split,
        root / "JPEGImages" / base,
        root / "JPEGImages" / split,
        root / "images",
        root / "Images",
        root,
    ])
    return _unique_paths(candidates)


def resolve_partimagenet_annotation_path(
    partimagenet_root: str | Path,
    split: str = "train",
    annotation_hint: str | Path | None = None,
) -> Path:
    """Resolve the COCO-style annotation JSON for a split.

    Examples
    --------
    resolve_partimagenet_annotation_path(root, "train")
    resolve_partimagenet_annotation_path(root, "train", "annotations/train/train.json")
    """
    root = Path(partimagenet_root).expanduser()
    candidates = _annotation_candidates(root, split, annotation_hint)
    found = next((p for p in candidates if p.exists() and p.is_file()), None)
    if found is not None:
        return found
    raise FileNotFoundError(
        f"Could not find PartImageNet {split!r} annotation JSON under root {root}.\n"
        "Expected the notebook/original PartImageNet layout, for example:\n"
        f"  {root / 'annotations' / str(split) / (str(split) + '.json')}\n"
        f"Tried:\n{_format_tried(candidates)}\n\n"
        + describe_partimagenet_layout(root)
    )


def resolve_partimagenet_image_root(
    partimagenet_root: str | Path,
    split: str = "train",
    image_root_hint: str | Path | None = None,
) -> Path:
    """Resolve the image directory for a PartImageNet split."""
    root = Path(partimagenet_root).expanduser()
    candidates = _image_root_candidates(root, split, image_root_hint)
    found = next((p for p in candidates if p.exists() and p.is_dir()), None)
    if found is not None:
        return found
    raise FileNotFoundError(
        f"Could not find PartImageNet {split!r} image directory under root {root}.\n"
        "Expected the notebook/original PartImageNet layout, for example:\n"
        f"  {root / 'images' / _base_split(str(split))}\n"
        f"Tried:\n{_format_tried(candidates)}\n\n"
        + describe_partimagenet_layout(root)
    )


def resolve_partimagenet_split_paths(
    partimagenet_root: str | Path,
    split: str,
    annotation_hint: str | Path | None = None,
    image_root_hint: str | Path | None = None,
) -> PartImageNetSplitPaths:
    """Resolve the JSON and image root for one PartImageNet split."""
    root = Path(partimagenet_root).expanduser()
    annotation_json = resolve_partimagenet_annotation_path(root, split, annotation_hint)
    image_root = resolve_partimagenet_image_root(root, split, image_root_hint)
    return PartImageNetSplitPaths(annotation_json=annotation_json, image_root=image_root)


def resolve_partimagenet_image_path(image_root: str | Path, img_info: dict) -> Path:
    image_root = Path(image_root).expanduser()
    file_name = str(img_info.get("file_name", img_info.get("path", "")))
    p = Path(file_name)
    if p.is_absolute() and p.exists():
        return p
    candidates = [image_root / file_name]
    base = Path(file_name).name
    # Infer WNID prefix from filenames like n01608432_13663.JPEG.
    if "_" in base:
        candidates.append(image_root / base.split("_")[0] / base)
    # COCO-style files sometimes include train/val in file_name; when the image
    # root already points to images/train, try dropping that prefix as well.
    parts = Path(file_name).parts
    if len(parts) >= 2 and parts[0] in {"train", "val", "test"}:
        candidates.append(image_root / Path(*parts[1:]))
    # Some JSONs store paths relative to a top-level root, e.g. images/train/...
    if len(parts) >= 3 and parts[0].lower() in {"images", "jpegimages"} and parts[1] in {"train", "val", "test"}:
        candidates.append(image_root / Path(*parts[2:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(image_root.rglob(base))
    return matches[0] if matches else candidates[0]


def _safe_polygon_points(poly, h: int, w: int):
    if poly is None or not isinstance(poly, (list, tuple)):
        return []
    if len(poly) > 0 and isinstance(poly[0], (list, tuple)):
        flat = []
        for item in poly:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
        poly = flat
    coords = []
    for x in poly:
        try:
            fx = float(x)
            if np.isfinite(fx):
                coords.append(fx)
        except Exception:
            continue
    if len(coords) % 2:
        coords = coords[:-1]
    if len(coords) < 6:
        return []
    pts = []
    for i in range(0, len(coords), 2):
        x = min(max(coords[i], 0.0), float(max(w - 1, 0)))
        y = min(max(coords[i + 1], 0.0), float(max(h - 1, 0)))
        pts.append((x, y))
    if len(set((round(x, 3), round(y, 3)) for x, y in pts)) < 3:
        return []
    return pts


def _draw_bbox_fallback(mask: Image.Image, bbox, h: int, w: int) -> bool:
    if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    try:
        x, y, bw, bh = [float(v) for v in bbox[:4]]
    except Exception:
        return False
    if not all(np.isfinite([x, y, bw, bh])) or bw <= 1 or bh <= 1:
        return False
    x0 = int(max(0, min(w - 1, round(x))))
    y0 = int(max(0, min(h - 1, round(y))))
    x1 = int(max(0, min(w - 1, round(x + bw))))
    y1 = int(max(0, min(h - 1, round(y + bh))))
    if x1 <= x0 or y1 <= y0:
        return False
    ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], outline=1, fill=1)
    return True


def rasterize_segmentation(segmentation, h: int, w: int, bbox=None) -> np.ndarray:
    mask = Image.new("L", (w, h), 0)
    drew_any = False
    if isinstance(segmentation, list):
        draw = ImageDraw.Draw(mask)
        if len(segmentation) > 0 and isinstance(segmentation[0], (int, float)):
            pts = _safe_polygon_points(segmentation, h, w)
            if pts:
                draw.polygon(pts, outline=1, fill=1)
                drew_any = True
        else:
            for poly in segmentation:
                pts = _safe_polygon_points(poly, h, w)
                if pts:
                    draw.polygon(pts, outline=1, fill=1)
                    drew_any = True
        if not drew_any:
            _draw_bbox_fallback(mask, bbox, h, w)
        return np.array(mask, dtype=np.uint8)
    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
            decoded = mask_utils.decode(segmentation).astype(np.uint8)
            if decoded.ndim == 3:
                decoded = decoded.max(axis=2)
            return decoded
        except Exception:
            _draw_bbox_fallback(mask, bbox, h, w)
            return np.array(mask, dtype=np.uint8)
    _draw_bbox_fallback(mask, bbox, h, w)
    return np.array(mask, dtype=np.uint8)


class RoleAwarePartImageNetDataset(Dataset):
    """COCO-style PartImageNet dataset with functional and role masks."""

    def __init__(
        self,
        annotation_json: str | Path,
        image_root: str | Path,
        *,
        img_size: int = 384,
        train: bool = True,
        schema: RoleSchema | None = None,
        max_samples: int | None = None,
    ):
        self.annotation_json = str(annotation_json)
        self.image_root = str(image_root)
        self.transform = JointImageMaskTransform(img_size, train=train)
        data = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
        self.images = {int(im["id"]): im for im in data["images"]}
        self.categories = {int(cat["id"]): cat for cat in data["categories"]}
        self.schema = schema or RoleSchema.from_coco_categories(self.categories.values())
        self.obj_to_idx = self.schema.obj_to_idx
        self.part_to_idx = self.schema.part_to_idx
        self.role_to_idx = self.schema.role_to_idx
        anns_by_image = defaultdict(list)
        for ann in data["annotations"]:
            anns_by_image[int(ann["image_id"])].append(ann)

        self.samples = []
        for image_id, img_info in self.images.items():
            anns = anns_by_image.get(image_id, [])
            if not anns:
                continue
            obj_counter = Counter()
            for ann in anns:
                cat = self.categories.get(int(ann["category_id"]))
                if cat is not None:
                    obj_counter[canonicalize_object_name(cat.get("supercategory", "unknown"))] += 1
            if not obj_counter:
                continue
            obj = obj_counter.most_common(1)[0][0]
            if obj not in self.obj_to_idx:
                continue
            func_present, role_present = set(), set()
            img_area = max(float(img_info.get("width", 1)) * float(img_info.get("height", 1)), 1.0)
            func_area = Counter()
            for ann in anns:
                cat = self.categories.get(int(ann["category_id"]))
                if cat is None or canonicalize_object_name(cat.get("supercategory", "unknown")) != obj:
                    continue
                fn = canonicalize_part_name(cat["name"])
                if fn not in self.part_to_idx:
                    continue
                rn = role_name(obj, fn)
                func_present.add(fn)
                role_present.add(rn)
                bbox = ann.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    func_area[fn] += min(max(float(bbox[2]), 0) * max(float(bbox[3]), 0) / img_area, 1.0)
            self.samples.append({
                "image_id": image_id,
                "img_path": str(resolve_partimagenet_image_path(self.image_root, img_info)),
                "width": int(img_info["width"]),
                "height": int(img_info["height"]),
                "obj_name": obj,
                "obj_label": self.obj_to_idx[obj],
                "annotations": anns,
                "functional_present": sorted(func_present),
                "role_present": sorted(role_present),
                "functional_bbox_area": dict(func_area),
            })
        if max_samples:
            self.samples = self.samples[: int(max_samples)]
        self._compute_balance_stats()
        print(
            f"[dataset] {Path(self.annotation_json).name}: {len(self.samples)} samples | "
            f"C={self.schema.num_classes} F={self.schema.num_parts} R={self.schema.num_roles}"
        )

    def _compute_balance_stats(self) -> None:
        fnum, rnum = self.schema.num_parts, self.schema.num_roles
        pc = np.zeros(fnum)
        area = np.zeros(fnum)
        rc = np.zeros(rnum)
        for sample in self.samples:
            for fn in sample.get("functional_present", []):
                if fn in self.part_to_idx:
                    k = self.part_to_idx[fn]
                    pc[k] += 1
                    area[k] += float(sample.get("functional_bbox_area", {}).get(fn, 0))
            for rn in sample.get("role_present", []):
                if rn in self.role_to_idx:
                    rc[self.role_to_idx[rn]] += 1
        freq = pc / max(len(self.samples), 1)
        avg_area = area / np.maximum(pc, 1)
        valid = pc > 0
        fb = np.median(freq[valid]) if valid.any() else 1
        ab = np.median(avg_area[valid]) if valid.any() else 1
        w = (fb / np.maximum(freq, 1e-6)) ** 0.35 * (ab / np.maximum(avg_area, 1e-6)) ** 0.5
        w[~valid] = 1
        w = np.clip(w / (np.mean(w[valid]) if valid.any() else 1), 0.3, 8.0)
        self.part_loss_weight = w.astype(np.float32)
        pos = np.clip((ab / np.maximum(avg_area, 1e-6)) ** 0.5, 1.0, 12.0)
        pos[~valid] = 1
        self.part_pos_weight = pos.astype(np.float32)
        rvalid = rc > 0
        rw = (np.median(rc[rvalid]) / np.maximum(rc, 1)) ** 0.35 if rvalid.any() else np.ones_like(rc)
        rw[~rvalid] = 1
        self.role_loss_weight = np.clip(rw / (np.mean(rw[rvalid]) if rvalid.any() else 1), 0.3, 8).astype(np.float32)
        sw = []
        for sample in self.samples:
            inds = [self.part_to_idx[fn] for fn in sample.get("functional_present", []) if fn in self.part_to_idx]
            sw.append(float(np.clip(np.mean(w[inds]) if inds else 1.0, 0.5, 4.0)))
        sw = np.asarray(sw)
        self.sample_weight = (sw / max(sw.mean(), 1e-6)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _build_masks(self, rec: dict) -> dict[str, np.ndarray]:
        h, w = int(rec["height"]), int(rec["width"])
        func_masks = {p: np.zeros((h, w), dtype=np.uint8) for p in self.schema.part_names}
        role_masks = {r: np.zeros((h, w), dtype=np.uint8) for r in self.schema.role_names}
        union = np.zeros((h, w), dtype=np.uint8)
        for ann in rec["annotations"]:
            cat = self.categories.get(int(ann["category_id"]))
            if cat is None or canonicalize_object_name(cat.get("supercategory", "unknown")) != rec["obj_name"]:
                continue
            fn = canonicalize_part_name(cat["name"])
            rn = role_name(rec["obj_name"], fn)
            if fn not in self.part_to_idx:
                continue
            mask = rasterize_segmentation(ann.get("segmentation"), h, w, bbox=ann.get("bbox"))
            func_masks[fn] = np.maximum(func_masks[fn], mask)
            union = np.maximum(union, mask)
            if rn in role_masks:
                role_masks[rn] = np.maximum(role_masks[rn], mask)
        all_masks = {f"func::{k}": v for k, v in func_masks.items()}
        all_masks.update({f"role::{k}": v for k, v in role_masks.items()})
        all_masks["__union__"] = union
        return all_masks

    def __getitem__(self, idx: int) -> dict:
        rec = self.samples[idx]
        img_path = Path(rec["img_path"])
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image file referenced by annotation was not found: {img_path}\n"
                f"annotation_json={self.annotation_json}\nimage_root={self.image_root}"
            )
        img = Image.open(img_path).convert("RGB")
        masks = self._build_masks(rec)
        img_t, img_raw, masks_t = self.transform(img, masks)
        union = masks_t.pop("__union__")
        fnum, rnum = self.schema.num_parts, self.schema.num_roles
        part_masks = torch.zeros(fnum, union.shape[-2], union.shape[-1], dtype=torch.float32)
        role_masks = torch.zeros(rnum, union.shape[-2], union.shape[-1], dtype=torch.float32)
        for k, part in enumerate(self.schema.part_names):
            part_masks[k] = masks_t[f"func::{part}"].float().squeeze(0)
        for r, role in enumerate(self.schema.role_names):
            role_masks[r] = masks_t[f"role::{role}"].float().squeeze(0)
        presence = (part_masks.flatten(1).amax(dim=-1) > 0).float()
        role_presence = (role_masks.flatten(1).amax(dim=-1) > 0).float()
        return {
            "image": img_t,
            "image_raw": img_raw,
            "obj_label": torch.tensor(rec["obj_label"], dtype=torch.long),
            "part_masks": part_masks,
            "role_masks": role_masks,
            "union_mask": union.float(),
            "presence": presence,
            "role_presence": role_presence,
            "meta": {"image_id": rec["image_id"], "path": rec["img_path"], "obj_name": rec["obj_name"]},
        }


class Stage2ImageOnlyDataset(Dataset):
    """Fast Stage-2 loader that avoids mask rasterization."""

    def __init__(self, full_dataset: RoleAwarePartImageNetDataset, train: bool = True):
        self.full = full_dataset
        self.transform = ImageOnlyTransform(full_dataset.transform.img_size, train=train)

    def __len__(self) -> int:
        return len(self.full.samples)

    def __getitem__(self, idx: int) -> dict:
        rec = self.full.samples[idx]
        img_path = Path(rec["img_path"])
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image file referenced by annotation was not found: {img_path}\n"
                f"annotation_json={self.full.annotation_json}\nimage_root={self.full.image_root}"
            )
        img = Image.open(img_path).convert("RGB")
        image, image_raw = self.transform(img)
        return {
            "image": image,
            "image_raw": image_raw,
            "obj_label": torch.tensor(rec["obj_label"], dtype=torch.long),
            "meta": {"image_id": rec["image_id"], "path": rec["img_path"], "obj_name": rec["obj_name"]},
        }
