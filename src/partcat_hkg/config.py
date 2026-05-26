from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
from typing import Any
import copy
import yaml


@dataclass
class PathsConfig:
    partimagenet_root: str = "./PartImageNet"
    unseen_root: str = "./auto_wnid_split_output/unseen_target_only"
    save_dir: str = "./runs/default"

    # PartImageNet follows the notebook/COCO-style layout:
    #   <root>/annotations/train/train.json
    #   <root>/annotations/val/val.json
    #   <root>/images/train/...
    #   <root>/images/val/...
    # Keep these configurable because some local splits flatten the JSONs.
    train_annotations: str = "annotations/train/train.json"
    val_annotations: str = "annotations/val/val.json"
    train_image_root: str = "images/train"
    val_image_root: str = "images/val"


@dataclass
class DataConfig:
    img_size: int = 320
    num_workers: int = 8
    persistent_workers: bool = True
    prefetch_factor: int = 4
    max_train_samples: int = 0
    max_val_samples: int = 0
    use_stage2_image_only_loader: bool = True


@dataclass
class Stage1Config:
    # Backbone and token-map sizes.
    backbone_name: str = "resnet18"
    model_dim: int = 192
    fuse_dim: int = 128
    token_dim: int = 128
    use_imagenet_backbone_pretrain: bool = False
    freeze_backbone: bool = False

    # Text prototypes for object / functional-part / object-aware-role costs.
    use_clip_text: bool = True
    clip_model_name: str = "ViT-B-16"
    clip_pretrain_tag: str = "laion2b_s34b_b88k"

    # Frozen structural guidance branch. The DINO input can be resized even when
    # the main image is larger, which avoids fixed-positional-embedding failures.
    use_dino: bool = True
    dino_model_name: str = "vit_small_patch16_224.dino"
    dino_input_size: int = 224
    dino_weights: str = ""
    freeze_dino: bool = True

    # PartCAT-style cost aggregation. The functional part cost volume is first
    # embedded with grouped pointwise convolutions, then optionally aggregated
    # spatially with DINO-guided attention and across part channels at each pixel.
    use_cost_aggregation: bool = True
    cost_embed_dim: int = 32
    cost_agg_blocks: int = 1
    cost_agg_heads: int = 4
    cost_dropout: float = 0.0
    use_spatial_aggregation: bool = True
    use_part_aggregation: bool = True
    spatial_attention_max_tokens: int = 512
    cost_temperature: float = 1.0
    dino_guidance_dim: int = 32

    # Backward-compatible aliases used by earlier skeleton/tests/config drafts.
    cost_agg_dim: int = 0
    cost_num_heads: int = 0
    cost_agg_max_tokens: int = 0
    use_spatial_cost_attention: bool | None = None
    use_class_cost_attention: bool | None = None

    # Presence and mask-weighted token pooling. Defaults are intentionally mild:
    # fixed-k presence and unsharpened masks, while retaining proposal-style
    # sharpening as a config ablation.
    topk_presence_k: int = 64
    presence_threshold: float = 0.15
    presence_topq: float = 0.0
    token_mask_temperature: float = 1.0
    token_mask_alpha: float = 1.0
    presence_gate_tokens: bool = True
    emit_role_tokens: bool = True

    # Optional high-resolution refinement branch.  The normal decoder predicts
    # at the ResNet low-feature resolution before upsampling; this branch uses
    # the higher-resolution skip feature map to sharpen small parts such as
    # mirrors, engines, beaks, seats, and small wheels.  It is disabled by
    # default to preserve old checkpoints/configs, and enabled in the Stage-1
    # quality-upgrade config.
    use_highres_refine: bool = False
    highres_refine_dim: int = 64

    # Small-part training controls used by the quality-upgrade loss.  These are
    # read by the helper loss when present; they are kept here so the notebook,
    # scripts, and config files can share the same defaults.
    small_part_area_tau: float = 0.015
    small_part_weight_max: float = 6.0
    small_part_weight_power: float = 0.5

    # Backward-compatible aliases for proposal/notebook naming.
    presence_top_fraction: float = 0.0
    mask_pool_temperature: float = 0.0
    mask_pool_power: float = 0.0
    mask_sharpen_temperature: float = 0.0
    mask_sharpen_alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.cost_agg_dim:
            self.cost_embed_dim = int(self.cost_agg_dim)
        if self.cost_num_heads:
            self.cost_agg_heads = int(self.cost_num_heads)
        if self.cost_agg_max_tokens:
            self.spatial_attention_max_tokens = int(self.cost_agg_max_tokens)
        if self.use_spatial_cost_attention is not None:
            self.use_spatial_aggregation = bool(self.use_spatial_cost_attention)
        if self.use_class_cost_attention is not None:
            self.use_part_aggregation = bool(self.use_class_cost_attention)
        if self.presence_top_fraction:
            self.presence_topq = float(self.presence_top_fraction)
        if self.mask_pool_temperature:
            self.token_mask_temperature = float(self.mask_pool_temperature)
        if self.mask_pool_power:
            self.token_mask_alpha = float(self.mask_pool_power)
        if self.mask_sharpen_temperature:
            self.token_mask_temperature = float(self.mask_sharpen_temperature)
        if self.mask_sharpen_alpha:
            self.token_mask_alpha = float(self.mask_sharpen_alpha)


@dataclass
class HKGConfig:
    max_images_per_class: int = 0
    role_edge_min_count: int = 3
    degree_cap: int = 4
    edge_select_topm: int = 2
    edge_min_strength: float = 0.03
    edge_min_information_gain: float = 0.01
    relation_mask_size: int = 80
    relation_var_floor: float = 1e-3
    store_pmi_diagnostic: bool = True

    # AOG-inspired HKG extension.  Each class owns a small set of alternative
    # templates (Or-node branches).  Templates are learned by clustering Stage-1
    # training evidence, then each template stores role priors, role appearance
    # prototypes, sparse relation templates, and motif factors.
    num_templates_per_class: int = 3
    template_kmeans_iters: int = 12
    min_template_support: int = 2
    template_presence_smoothing: float = 1.0
    template_prior_smoothing: float = 1.0
    template_required_tau: float = 0.45
    template_edge_min_support: float = 0.12
    template_edge_degree_cap: int = 5
    template_edge_max_edges: int = 12
    template_edge_shrink_kappa: float = 8.0

    # Revised HKG-v2 settings.  The first HKG run showed that strict IG-only
    # edge pursuit made car templates collapse to a single body-wheel edge,
    # while generic motifs were either over-promoted or absent.  We therefore
    # keep information-gain pursuit, but add a conservative anchor-edge rescue:
    # if a high-prior role is reliably attached to a central body/frame role,
    # keep that edge even when its global-vs-class IG is small.
    anchor_edge_min_support: float = 0.18
    anchor_edge_required_prior: float = 0.25
    anchor_edge_max_per_template: int = 6
    motif_min_support: float = 0.30

    # Build HKG statistics from the same Stage-1 outputs used at runtime.
    # GT masks are still used to decide semantic part presence by default, but
    # appearance tokens and relation geometry should come from frozen Stage-1
    # predictions to avoid train/inference geometry mismatch.
    use_predicted_stage1_evidence: bool = True


@dataclass
class Stage2Config:
    hidden_dim: int = 256
    main_classifier: str = "parse_graph"
    use_pmi_in_main: bool = False
    use_adaptive_fusion: bool = False
    enable_legacy_v51_fusion: bool = False
    enable_relation_routing: bool = False
    role_presence_mode: str = "role"  # role, agreement, max
    partial_parse_eps: float = 1e-4
    visibility_presence_tau: float = 0.15
    visibility_quality_tau: float = 0.25
    lambda_completion_init: float = 0.35
    lambda_edge_init: float = 0.20
    lambda_contradiction_init: float = 0.10
    edge_score_mode: str = "template_fit"
    edge_positive_only: bool = True
    fusion_mode: str = "adaptive"
    fusion_output_space: str = "log_opinion_pool"
    fusion_expert_norm: str = "zscore"
    fusion_prob_tau: float = 1.0
    fusion_final_tau: float = 1.0
    relation_context_scale: float = 1.0
    relation_score_scale: float = 0.25
    relation_template_llr_weight: float = 0.15

    # AOG-HKG parse scorer.  These weights are intentionally mild by default:
    # Stage 2 starts as a calibrated explicit-HKG residual on top of a base
    # classifier, rather than a hard grammar-only classifier.
    hkg_score_clip: float = 30.0
    hkg_use_logsumexp_templates: bool = True
    hkg_template_tau: float = 1.0
    hkg_node_presence_scale: float = 0.50
    hkg_node_pmi_scale: float = 0.35
    hkg_node_app_scale: float = 1.00
    hkg_absence_penalty: float = 0.35
    hkg_conflict_penalty: float = 0.35
    hkg_spurious_template_penalty: float = 0.25
    hkg_spurious_template_tau: float = 0.08
    hkg_edge_scale: float = 0.55
    hkg_motif_scale: float = 0.12
    hkg_edge_positive_only: bool = False
    hkg_center_relation_scores: bool = True
    hkg_calibrated_fusion: bool = True
    hkg_fusion_lambda_init: float = 0.20
    hkg_use_classwise_fusion: bool = True
    hkg_normalize_scores: bool = True
    hkg_presence_floor: float = 1.0e-4


@dataclass
class ModelConfig:
    stage1: Stage1Config = field(default_factory=Stage1Config)
    hkg: HKGConfig = field(default_factory=HKGConfig)
    stage2: Stage2Config = field(default_factory=Stage2Config)


@dataclass
class CurriculumStep:
    name: str
    epochs: int
    enable_completion: bool = False
    enable_edges: bool = False


@dataclass
class TrainingConfig:
    stage1_epochs: int = 18
    stage2_epochs: int = 16
    batch_size_stage1: int = 12
    batch_size_stage2: int = 8
    lr_stage1: float = 1e-4
    lr_stage2: float = 3e-4
    weight_decay: float = 1e-4
    use_amp: bool = True
    class_balanced_stage2: bool = True
    curriculum: list[CurriculumStep] = field(default_factory=lambda: [
        CurriculumStep("node_warmup", 4, False, False),
        CurriculumStep("completion_warmup", 4, True, False),
        CurriculumStep("edge_warmup", 8, True, True),
    ])


@dataclass
class Stage1LossConfig:
    support: float = 1.0
    functional: float = 1.0
    role: float = 1.4
    invalid_role_negative: float = 0.05
    invalid_role_topk: float = 0.10
    functional_role_consistency: float = 0.25
    object_part_composition: float = 0.35
    dino_affinity: float = 0.02
    area_mass: float = 0.10
    present_coverage: float = 0.20

    # Optional Stage-1 quality upgrade, now wired directly into
    # scripts/train_stage1.py.  This replaces the earlier notebook-only
    # fine-tuning path: the only Stage-1 training entry point should be
    # `python scripts/train_stage1.py ...`.  Defaults keep the original
    # objective unless `quality_enable` is true.
    quality_enable: bool = False
    quality_presence_bce: float = 0.40
    valid_absent_topmean_fp: float = 0.08
    valid_absent_mean_fp: float = 0.02
    invalid_part_topmean: float = 0.35
    invalid_part_mean: float = 0.08
    gt_support_leak: float = 0.35
    pred_support_containment: float = 0.25
    boundary: float = 0.08
    focal_functional: float = 0.12
    tversky_functional: float = 0.12
    quality_topq: float = 0.02
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    tversky_alpha: float = 0.35
    tversky_beta: float = 0.65
    boundary_kernel: int = 3


@dataclass
class Stage2LossConfig:
    parse_ce: float = 1.0
    visible_aux: float = 0.20
    completion_aux: float = 0.10
    edge_aux: float = 0.05
    base_aux: float = 1.0
    hkg_aux: float = 0.25
    motif_aux: float = 0.02


@dataclass
class LossConfig:
    stage1: Stage1LossConfig = field(default_factory=Stage1LossConfig)
    stage2: Stage2LossConfig = field(default_factory=Stage2LossConfig)


@dataclass
class AnalysisConfig:
    vis_num_samples: int = 3
    mask_bin_thr: float = 0.40
    presence_tau: float = 0.15
    stage1_audit_max_batches: int = 0
    output_inline: bool = False


@dataclass
class ProjectConfig:
    seed: int = 123
    run_name: str = "default"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml_with_extends(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        base = _load_yaml_with_extends(parent_path)
        return deep_update(base, raw)
    return raw


def _coerce_dataclass(cls: type, data: Any):
    if not is_dataclass(cls):
        return data
    if data is None:
        return cls()
    kwargs = {}
    field_map = {f.name: f for f in fields(cls)}
    for name, f in field_map.items():
        if isinstance(data, dict) and name in data:
            value = data[name]
        else:
            continue
        origin = getattr(f.type, "__origin__", None)
        if origin is list and name == "curriculum":
            kwargs[name] = [CurriculumStep(**x) if isinstance(x, dict) else x for x in value]
        elif is_dataclass(f.type):
            kwargs[name] = _coerce_dataclass(f.type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    data = _load_yaml_with_extends(path)
    # Manual nested conversion keeps the code dependency-free and transparent.
    cfg = ProjectConfig()
    merged = deep_update(cfg.to_dict(), data)
    return ProjectConfig(
        seed=merged.get("seed", cfg.seed),
        run_name=merged.get("run_name", cfg.run_name),
        paths=_coerce_dataclass(PathsConfig, merged.get("paths")),
        data=_coerce_dataclass(DataConfig, merged.get("data")),
        model=ModelConfig(
            stage1=_coerce_dataclass(Stage1Config, merged.get("model", {}).get("stage1")),
            hkg=_coerce_dataclass(HKGConfig, merged.get("model", {}).get("hkg")),
            stage2=_coerce_dataclass(Stage2Config, merged.get("model", {}).get("stage2")),
        ),
        training=_coerce_dataclass(TrainingConfig, merged.get("training")),
        loss=LossConfig(
            stage1=_coerce_dataclass(Stage1LossConfig, merged.get("loss", {}).get("stage1")),
            stage2=_coerce_dataclass(Stage2LossConfig, merged.get("loss", {}).get("stage2")),
        ),
        analysis=_coerce_dataclass(AnalysisConfig, merged.get("analysis")),
    )
