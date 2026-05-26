import torch


def collate_part_batch(batch: list[dict]) -> dict:
    out = {}
    tensor_keys = ["image", "image_raw", "obj_label", "part_masks", "role_masks", "union_mask", "presence", "role_presence"]
    for key in tensor_keys:
        if key in batch[0]:
            out[key] = torch.stack([item[key] for item in batch])
    out["meta"] = [item.get("meta", {}) for item in batch]
    return out


def collate_stage2_image_only(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "image_raw": torch.stack([item["image_raw"] for item in batch]),
        "obj_label": torch.stack([item["obj_label"] for item in batch]),
        "meta": [item.get("meta", {}) for item in batch],
    }
