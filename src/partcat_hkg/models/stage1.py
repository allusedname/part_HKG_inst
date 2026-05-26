from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from partcat_hkg.config import Stage1Config
from partcat_hkg.data.schema import RoleSchema
from .backbones import ConvGNAct, OptionalDINOFeatureMap, ResNetFeatureBackbone
from .pooling import gated_masked_pool, topk_presence, topmean_presence
from .text_prototypes import TextPrototypeBank


def _valid_num_heads(dim: int, requested: int) -> int:
    dim = int(dim)
    requested = max(1, int(requested))
    for heads in range(min(dim, requested), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class GroupedCostEmbedding(nn.Module):
    """Independent per-part pointwise embedding of a scalar cost volume.

    Input is [B, K, H, W].  Output is [B, K, D, H, W].  Grouped 1x1 convolutions
    keep channels independent before the spatial/class aggregation steps.
    """

    def __init__(self, num_parts: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_parts = int(num_parts)
        self.embed_dim = int(embed_dim)
        self.net = nn.Sequential(
            nn.Conv2d(self.num_parts, self.num_parts * self.embed_dim, 1, groups=self.num_parts, bias=True),
            nn.SiLU(inplace=True),
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Conv2d(self.num_parts * self.embed_dim, self.num_parts * self.embed_dim, 1, groups=self.num_parts, bias=True),
            nn.SiLU(inplace=True),
        )

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        bsz, num_parts, height, width = cost.shape
        if num_parts != self.num_parts:
            raise ValueError(f"Expected {self.num_parts} part cost channels, got {num_parts}")
        x = self.net(cost)
        return x.view(bsz, self.num_parts, self.embed_dim, height, width)


class DINOGuidedSpatialAttentionBlock(nn.Module):
    """Spatial self-attention per part with Q/K augmented by structural features.

    This manual single-head implementation avoids the CPU/runtime fragility of
    ``nn.MultiheadAttention`` while preserving the intended PartCAT-style
    operation. The ``num_heads`` argument is accepted for config compatibility.
    """

    def __init__(self, part_dim: int, guide_dim: int, num_heads: int, dropout: float = 0.0, max_tokens: int = 512):
        super().__init__()
        self.part_dim = int(part_dim)
        self.guide_dim = int(guide_dim)
        self.max_tokens = int(max_tokens)
        self.qk_norm = nn.LayerNorm(self.part_dim + self.guide_dim)
        self.v_norm = nn.LayerNorm(self.part_dim)
        self.q_proj = nn.Linear(self.part_dim + self.guide_dim, self.part_dim)
        self.k_proj = nn.Linear(self.part_dim + self.guide_dim, self.part_dim)
        self.v_proj = nn.Linear(self.part_dim, self.part_dim)
        self.out_proj = nn.Linear(self.part_dim, self.part_dim)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.out_norm = nn.LayerNorm(self.part_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.part_dim, self.part_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(self.part_dim * 2, self.part_dim),
        )

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        # x: [B, K, D, H, W], guide: [B, G, H, W]
        bsz, num_parts, dim, height, width = x.shape
        n_tokens = height * width
        if n_tokens > self.max_tokens:
            scale = math.sqrt(float(n_tokens) / float(max(self.max_tokens, 1)))
            new_h = max(1, int(round(height / scale)))
            new_w = max(1, int(round(width / scale)))
            x_small = F.adaptive_avg_pool2d(x.reshape(bsz * num_parts, dim, height, width), (new_h, new_w))
            x_small = x_small.view(bsz, num_parts, dim, new_h, new_w)
            g_small = F.adaptive_avg_pool2d(guide, (new_h, new_w))
            y_small = self.forward(x_small, g_small)
            y = F.interpolate(
                y_small.reshape(bsz * num_parts, dim, new_h, new_w),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            return y.view(bsz, num_parts, dim, height, width)

        tokens = x.permute(0, 1, 3, 4, 2).reshape(bsz * num_parts, n_tokens, dim)
        g = guide.permute(0, 2, 3, 1).unsqueeze(1).expand(bsz, num_parts, height, width, self.guide_dim)
        g = g.reshape(bsz * num_parts, n_tokens, self.guide_dim)
        qk = self.qk_norm(torch.cat([tokens, g], dim=-1))
        q = self.q_proj(qk)
        k = self.k_proj(qk)
        v = self.v_proj(self.v_norm(tokens))
        attn = torch.softmax(torch.bmm(q.float(), k.float().transpose(1, 2)) / math.sqrt(float(dim)), dim=-1).to(v.dtype)
        tokens = tokens + self.dropout(self.out_proj(torch.bmm(attn, v)))
        tokens = tokens + self.mlp(self.out_norm(tokens))
        return tokens.view(bsz, num_parts, height, width, dim).permute(0, 1, 4, 2, 3).contiguous()


class PartSetAttentionBlock(nn.Module):
    """Part aggregation at each spatial location using manual self-attention."""

    def __init__(self, part_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.part_dim = int(part_dim)
        self.norm1 = nn.LayerNorm(self.part_dim)
        self.qkv = nn.Linear(self.part_dim, self.part_dim * 3)
        self.out_proj = nn.Linear(self.part_dim, self.part_dim)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(self.part_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.part_dim, self.part_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(self.part_dim * 2, self.part_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_parts, dim, height, width = x.shape
        tokens = x.permute(0, 3, 4, 1, 2).reshape(bsz * height * width, num_parts, dim)
        z = self.norm1(tokens)
        q, k, v = self.qkv(z).chunk(3, dim=-1)
        attn = torch.softmax(torch.bmm(q.float(), k.float().transpose(1, 2)) / math.sqrt(float(dim)), dim=-1).to(v.dtype)
        tokens = tokens + self.dropout(self.out_proj(torch.bmm(attn, v)))
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.view(bsz, height, width, num_parts, dim).permute(0, 3, 4, 1, 2).contiguous()


class PartCostAggregation(nn.Module):
    """PartCAT-style grouped cost embedding + spatial/part aggregation."""

    def __init__(self, num_parts: int, cfg: Stage1Config):
        super().__init__()
        self.num_parts = int(num_parts)
        self.embed_dim = int(cfg.cost_embed_dim)
        self.enabled = bool(cfg.use_cost_aggregation)
        self.use_spatial = bool(cfg.use_spatial_aggregation)
        self.use_part = bool(cfg.use_part_aggregation)
        self.embed = GroupedCostEmbedding(num_parts, self.embed_dim, dropout=cfg.cost_dropout)
        blocks: list[nn.ModuleDict] = []
        for _ in range(max(1, int(cfg.cost_agg_blocks))):
            blocks.append(nn.ModuleDict({
                "spatial": DINOGuidedSpatialAttentionBlock(
                    self.embed_dim,
                    int(cfg.dino_guidance_dim),
                    int(cfg.cost_agg_heads),
                    dropout=float(cfg.cost_dropout),
                    max_tokens=int(cfg.spatial_attention_max_tokens),
                ) if self.use_spatial else nn.Identity(),
                "part": PartSetAttentionBlock(
                    self.embed_dim,
                    int(cfg.cost_agg_heads),
                    dropout=float(cfg.cost_dropout),
                ) if self.use_part else nn.Identity(),
            }))
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.GroupNorm(num_groups=1, num_channels=self.num_parts * self.embed_dim)

    def forward(self, cost: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        # cost: [B,K,H,W], guide: [B,G,H,W]
        x = self.embed(cost)
        if self.enabled:
            for block in self.blocks:
                spatial = block["spatial"]
                part = block["part"]
                x = spatial(x, guide) if not isinstance(spatial, nn.Identity) else x
                x = part(x) if not isinstance(part, nn.Identity) else x
        bsz, num_parts, dim, height, width = x.shape
        return self.out_norm(x.reshape(bsz, num_parts * dim, height, width))


class PartCATHKGStage1(nn.Module):
    """PartCAT-inspired object-aware functional/role parser.

    Outputs include the compatibility tensors used by the previous notebook
    (`part_logits`, `role_logits`, token maps) plus explicit Stage-1 products
    from the proposal: functional part presence and mask-pooled part tokens.
    """

    def __init__(self, schema: RoleSchema, cfg: Stage1Config):
        super().__init__()
        self.schema = schema
        self.cfg = cfg
        self.num_classes = schema.num_classes
        self.num_parts = schema.num_parts
        self.num_roles = schema.num_roles
        self.register_buffer("role_to_obj", schema.role_to_obj.clone().long())
        self.register_buffer("role_to_part", schema.role_to_part.clone().long())

        self.backbone = ResNetFeatureBackbone(
            name=cfg.backbone_name,
            pretrained=cfg.use_imagenet_backbone_pretrain,
            freeze=cfg.freeze_backbone,
        )
        self.dino = OptionalDINOFeatureMap(cfg.use_dino, cfg.dino_model_name, cfg.dino_weights, input_size=cfg.dino_input_size, freeze=cfg.freeze_dino)
        dino_ch = max(1, int(self.dino.out_ch or 1))
        self.text_bank = TextPrototypeBank(
            schema.obj_names,
            schema.part_names,
            schema.role_names,
            enabled=cfg.use_clip_text,
            model_name=cfg.clip_model_name,
            pretrained=cfg.clip_pretrain_tag,
        )
        text_dim = int(self.text_bank.obj_text.shape[-1])
        model_dim, fuse_dim = int(cfg.model_dim), int(cfg.fuse_dim)

        self.skip_proj = ConvGNAct(self.backbone.skip_ch, model_dim, k=1, p=0)
        self.low_proj = ConvGNAct(self.backbone.low_ch, model_dim, k=1, p=0)
        self.high_proj = ConvGNAct(self.backbone.high_ch, model_dim, k=1, p=0)
        self.dino_proj = ConvGNAct(dino_ch, model_dim, k=1, p=0)
        self.dino_guide_proj = nn.Sequential(
            nn.Conv2d(model_dim, int(cfg.dino_guidance_dim), 1, bias=False),
            nn.GroupNorm(1, int(cfg.dino_guidance_dim)),
            nn.SiLU(inplace=True),
        )

        self.cost_vis_proj = nn.Conv2d(model_dim, text_dim, 1)
        self.func_cost_agg = PartCostAggregation(self.num_parts, cfg)
        self.func_agg_context = ConvGNAct(self.num_parts * int(cfg.cost_embed_dim), 64, k=1, p=0)
        self.part_agg_head = nn.Conv2d(
            self.num_parts * int(cfg.cost_embed_dim),
            self.num_parts,
            1,
            groups=self.num_parts,
            bias=True,
        )

        self.obj_cost_proj = ConvGNAct(self.num_classes, 64, k=1, p=0)
        self.func_cost_proj = ConvGNAct(self.num_parts, 64, k=1, p=0)
        self.role_cost_proj = ConvGNAct(self.num_roles, 96, k=1, p=0)

        self.fuse_high = nn.Sequential(
            ConvGNAct(model_dim * 2 + 64 + 64 + 96 + 64, fuse_dim),
            ConvGNAct(fuse_dim, fuse_dim),
        )
        self.obj_head_low = nn.Conv2d(fuse_dim, self.num_classes, 1)
        self.support_head_low = nn.Conv2d(fuse_dim, 1, 1)
        self.part_head_low = nn.Conv2d(fuse_dim, self.num_parts, 1)
        self.role_head_low = nn.Conv2d(fuse_dim, self.num_roles, 1)
        self.refine = nn.Sequential(
            ConvGNAct(model_dim + fuse_dim + 1 + self.num_parts, fuse_dim),
            ConvGNAct(fuse_dim, fuse_dim),
        )
        self.part_residual = nn.Conv2d(fuse_dim, self.num_parts, 1)
        self.role_residual = nn.Conv2d(fuse_dim, self.num_roles, 1)
        self.support_residual = nn.Conv2d(fuse_dim, 1, 1)
        self.obj_residual = nn.Conv2d(fuse_dim, self.num_classes, 1)

        # Optional small-part refinement at the higher-resolution skip map.
        # This branch is intentionally lightweight: it receives the high-res
        # skip feature plus the coarse part/support logits, then predicts a
        # residual correction before the final full-resolution upsample.
        self.use_highres_refine = bool(getattr(cfg, "use_highres_refine", False))
        hr_dim = int(getattr(cfg, "highres_refine_dim", 64))
        if self.use_highres_refine:
            self.highres_refine = nn.Sequential(
                ConvGNAct(model_dim + self.num_parts + 1, hr_dim),
                ConvGNAct(hr_dim, hr_dim),
            )
            self.part_highres_residual = nn.Conv2d(hr_dim, self.num_parts, 1)
            self.support_highres_residual = nn.Conv2d(hr_dim, 1, 1)
        else:
            self.highres_refine = None
            self.part_highres_residual = None
            self.support_highres_residual = None

        self.token_res_map = nn.Conv2d(fuse_dim, cfg.token_dim, 1)
        self.token_dino_map = nn.Conv2d(model_dim, cfg.token_dim, 1)


    @property
    def status(self) -> dict[str, object]:
        """Human-readable implementation status for logs and notebooks."""
        return {
            "backbone": getattr(self.backbone, "name", type(self.backbone).__name__),
            "backbone_fallback": bool(getattr(self.backbone, "using_fallback", False)),
            "dino": getattr(self.dino, "_status", "unknown"),
            "text": getattr(self.text_bank, "_status", "unknown"),
            "cost_aggregation": bool(self.cfg.use_cost_aggregation),
            "spatial_aggregation": bool(self.cfg.use_spatial_aggregation),
            "part_aggregation": bool(self.cfg.use_part_aggregation),
            "token_pooling": {
                "temperature": float(self.cfg.token_mask_temperature),
                "alpha": float(self.cfg.token_mask_alpha),
                "presence_gated": bool(self.cfg.presence_gate_tokens),
            },
            "highres_refine": bool(getattr(self.cfg, "use_highres_refine", False)),
        }


    def set_stage1_trainable(self) -> None:
        """Train Stage-1 heads while respecting frozen backbone/DINO settings."""
        for p in self.parameters():
            p.requires_grad_(True)
        if bool(getattr(self.cfg, "freeze_backbone", False)):
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.dino.eval()

    @torch.no_grad()
    def smoke_forward(self, batch_size: int = 2, image_size: int = 96, device: str | torch.device = "cpu") -> dict[str, tuple[int, ...]]:
        """Run a synthetic forward pass and return output shapes.

        This is intentionally dataset-free so the command notebook can validate
        Stage 1 installation, optional dependency fallbacks, and tensor contracts.
        """
        was_training = self.training
        self.eval()
        device = torch.device(device)
        x = torch.randn(int(batch_size), 3, int(image_size), int(image_size), device=device)
        out = self.forward(x)
        if was_training:
            self.train()
        keys = [
            "support_logits",
            "part_logits",
            "role_logits",
            "obj_logits",
            "part_presence",
            "role_presence",
            "part_tokens",
            "role_tokens",
            "func_agg",
        ]
        return {k: tuple(out[k].shape) for k in keys if k in out}

    def _costs(self, high_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = max(float(self.cfg.cost_temperature), 1e-6)
        v = F.normalize(self.cost_vis_proj(high_feat).float(), dim=1)
        obj = F.normalize(self.text_bank.obj_text.to(v.device).float(), dim=-1)
        func = F.normalize(self.text_bank.func_text.to(v.device).float(), dim=-1)
        role = F.normalize(self.text_bank.role_text.to(v.device).float(), dim=-1)
        obj_cost = torch.einsum("bdhw,cd->bchw", v, obj) / scale
        func_cost = torch.einsum("bdhw,kd->bkhw", v, func) / scale
        role_cost = torch.einsum("bdhw,rd->brhw", v, role) / scale
        return obj_cost, func_cost, role_cost

    def _pool_tokens(self, token_map: torch.Tensor, prob: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        return gated_masked_pool(
            token_map,
            prob,
            presence,
            temperature=float(self.cfg.token_mask_temperature),
            power=float(self.cfg.token_mask_alpha),
            gate=bool(self.cfg.presence_gate_tokens),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        skip = self.skip_proj(feats["skip"])
        low = self.low_proj(feats["low"])
        low = low + F.interpolate(skip, size=low.shape[-2:], mode="bilinear", align_corners=False)
        high = self.high_proj(feats["high"])
        dino = self.dino(x, target_hw=high.shape[-2:])
        dino_p = self.dino_proj(dino)
        dino_guide = self.dino_guide_proj(dino_p)
        obj_cost, func_cost, role_cost = self._costs(high)

        func_agg = self.func_cost_agg(func_cost, dino_guide)
        func_agg_context = self.func_agg_context(func_agg)
        part_agg_logits = self.part_agg_head(func_agg)

        fused = self.fuse_high(torch.cat([
            high,
            dino_p,
            self.obj_cost_proj(obj_cost),
            self.func_cost_proj(func_cost),
            self.role_cost_proj(role_cost),
            func_agg_context,
        ], dim=1))
        obj_low = self.obj_head_low(fused) + obj_cost
        support_low = self.support_head_low(fused) + obj_low.amax(dim=1, keepdim=True)
        part_low = self.part_head_low(fused) + func_cost + part_agg_logits

        role_part_idx = self.role_to_part.clamp_min(0)
        role_obj_idx = self.role_to_obj.clamp_min(0)
        role_prior = role_cost + part_low[:, role_part_idx] + obj_low[:, role_obj_idx]
        role_low = self.role_head_low(fused) + role_prior

        fused_up = F.interpolate(fused, size=low.shape[-2:], mode="bilinear", align_corners=False)
        part_up = F.interpolate(part_low, size=low.shape[-2:], mode="bilinear", align_corners=False)
        supp_up = F.interpolate(support_low, size=low.shape[-2:], mode="bilinear", align_corners=False)
        ref = self.refine(torch.cat([low, fused_up, torch.sigmoid(supp_up), part_up], dim=1))

        part_logits_low = part_up + self.part_residual(ref)
        role_logits_low = F.interpolate(role_low, size=low.shape[-2:], mode="bilinear", align_corners=False) + self.role_residual(ref)
        support_logits_low = supp_up + self.support_residual(ref)
        obj_logits_low = F.interpolate(obj_low, size=low.shape[-2:], mode="bilinear", align_corners=False) + self.obj_residual(ref)

        # Refine part/support logits at the skip-feature resolution when enabled.
        # This keeps the original low-resolution token map unchanged for backward
        # compatibility while producing sharper masks for small structures.
        if self.use_highres_refine and self.highres_refine is not None:
            part_skip = F.interpolate(part_logits_low, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            supp_skip = F.interpolate(support_logits_low, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            hr = self.highres_refine(torch.cat([skip, torch.sigmoid(supp_skip), part_skip], dim=1))
            part_logits_mid = part_skip + self.part_highres_residual(hr)
            support_logits_mid = supp_skip + self.support_highres_residual(hr)
        else:
            part_logits_mid = part_logits_low
            support_logits_mid = support_logits_low

        part_logits = F.interpolate(part_logits_mid, size=x.shape[-2:], mode="bilinear", align_corners=False)
        role_logits = F.interpolate(role_logits_low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        support_logits = F.interpolate(support_logits_mid, size=x.shape[-2:], mode="bilinear", align_corners=False)
        obj_logits = F.interpolate(obj_logits_low, size=x.shape[-2:], mode="bilinear", align_corners=False)

        token_res = self.token_res_map(ref)
        token_dino = self.token_dino_map(F.interpolate(dino_p, size=ref.shape[-2:], mode="bilinear", align_corners=False))

        part_prob = torch.sigmoid(part_logits)
        role_prob = torch.sigmoid(role_logits)
        obj_prob = torch.sigmoid(obj_logits)
        support_prob = torch.sigmoid(support_logits)
        if float(self.cfg.presence_topq) > 0:
            part_presence = topmean_presence(part_prob, q=float(self.cfg.presence_topq))
            role_presence = topmean_presence(role_prob, q=float(self.cfg.presence_topq))
        else:
            part_presence = topk_presence(part_prob, k=self.cfg.topk_presence_k)
            role_presence = topk_presence(role_prob, k=self.cfg.topk_presence_k)

        part_tokens_res = self._pool_tokens(token_res, part_prob, part_presence)
        part_tokens_dino = self._pool_tokens(token_dino, part_prob, part_presence)
        role_tokens_res = self._pool_tokens(token_res, role_prob, role_presence)
        role_tokens_dino = self._pool_tokens(token_dino, role_prob, role_presence)
        part_tokens = 0.5 * (part_tokens_res + part_tokens_dino)
        role_tokens = 0.5 * (role_tokens_res + role_tokens_dino)

        return {
            "support_logits": support_logits,
            "part_logits": part_logits,
            "role_logits": role_logits,
            "role_logits_low": role_logits_low,
            "obj_logits": obj_logits,
            "support_prob": support_prob,
            "part_prob": part_prob,
            "role_prob": role_prob,
            "obj_prob": obj_prob,
            "part_presence": part_presence,
            "role_presence": role_presence,
            "part_tokens": part_tokens,
            "part_tokens_res": part_tokens_res,
            "part_tokens_dino": part_tokens_dino,
            "role_tokens": role_tokens,
            "role_tokens_res": role_tokens_res,
            "role_tokens_dino": role_tokens_dino,
            "token_res_map": token_res,
            "token_dino_map": token_dino,
            "func_cost": func_cost,
            "obj_cost": obj_cost,
            "role_cost": role_cost,
            "func_agg": func_agg,
        }
