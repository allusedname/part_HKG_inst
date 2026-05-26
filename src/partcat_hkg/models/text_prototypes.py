from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from partcat_hkg.data.canonicalization import prompt_part_text, role_prompt_text


def _fallback_bank(n: int, dim: int = 512, seed: int = 13) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed + 997 * n)
    return F.normalize(torch.randn(n, dim, generator=gen), dim=-1)


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in items:
        item = " ".join(str(item).split())
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


class TextPrototypeBank(nn.Module):
    """CLIP prompt prototype bank with deterministic random fallback.

    The prototype definition follows the proposal: multiple prompt templates are
    encoded, normalized independently, averaged, and normalized again.  If
    ``open_clip`` or its weights are unavailable, deterministic random unit
    vectors are used so the Stage-1 model remains runnable offline.
    """

    def __init__(
        self,
        obj_names: list[str],
        part_names: list[str],
        role_names: list[str],
        *,
        enabled: bool = True,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self._status = "fallback"
        obj_text = _fallback_bank(len(obj_names), seed=1)
        func_text = _fallback_bank(len(part_names), seed=2)
        role_text = _fallback_bank(len(role_names), seed=3)
        if enabled:
            try:
                import open_clip

                model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
                tokenizer = open_clip.get_tokenizer(model_name)
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(False)

                def encode_groups(groups: list[list[str]]) -> torch.Tensor:
                    outs = []
                    with torch.no_grad():
                        for prompts in groups:
                            tok = tokenizer(prompts)
                            emb = model.encode_text(tok)
                            emb = F.normalize(emb.float(), dim=-1)
                            emb = F.normalize(emb.mean(0), dim=0)
                            outs.append(emb.cpu())
                    return torch.stack(outs, dim=0)

                obj_groups = [
                    _dedupe([
                        f"a photo of a {o}",
                        f"the {o} in the image",
                        f"a close-up photo of a {o}",
                        f"an object of class {o}",
                    ])
                    for o in obj_names
                ]
                func_groups = []
                for p in part_names:
                    pp = prompt_part_text(p)
                    singular = pp[:-1] if pp.endswith("s") and len(pp) > 3 else pp
                    func_groups.append(_dedupe([
                        f"a photo of a {pp}",
                        f"a photo of the {pp}",
                        f"the {pp} of an object",
                        f"a close-up of the {pp}",
                        f"a close-up of {singular}",
                    ]))
                role_groups = []
                for role in role_names:
                    obj, part = role_prompt_text(role)
                    role_groups.append(_dedupe([
                        f"a photo of a {obj}'s {part}",
                        f"the {part} of a {obj}",
                        f"a close-up of the {part} of a {obj}",
                        f"{obj} {part}",
                    ]))
                obj_text = encode_groups(obj_groups)
                func_text = encode_groups(func_groups)
                role_text = encode_groups(role_groups)
                self._status = f"open_clip:{model_name}/{pretrained}; prompt_ensemble"
            except Exception as exc:  # pragma: no cover - optional dependency/cache
                self._status = f"fallback:{exc}"
        self.dim = int(obj_text.shape[-1])
        self.register_buffer("obj_text", obj_text)
        self.register_buffer("func_text", func_text)
        self.register_buffer("role_text", role_text)
