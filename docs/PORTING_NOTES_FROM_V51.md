# Porting notes from `revised_v51_partcat_hkg_logopinion_confidence_fusion.ipynb`

| Notebook item | Skeleton location | Notes |
|---|---|---|
| Environment flags and paths | `partcat_hkg/config.py`, `configs/*.yaml` | Replace global environment reads with dataclass config. |
| Canonicalization functions | `data/canonicalization.py` | Keeps object/part synonym cleanup and `role_name`. |
| `RoleAwarePartImageNetDataset` | `data/partimagenet.py` | Keeps functional/role masks and object-gated roles. |
| `Stage2ImageOnlyDataset` | `data/partimagenet.py` | Preserved for fast Stage-2 training. |
| `PartCATHKGStage1` | `models/stage1.py` | Same idea: ResNet maps + optional DINO + CLIP cost maps + support/functional/role heads. |
| `stage1_loss` | `models/losses.py` | Support, functional, role, invalid-role top-k, consistency, composition. |
| `HKG` dataclass | `kg/datatypes.py` | Adds explicit proposal-friendly names while preserving v51 tensors. |
| `build_hierarchical_kg` | `kg/builder.py` | Builds prototypes, role edges, relation templates, and PMI diagnostics. |
| Relation feature functions | `kg/relations.py` | Keeps explicit centroid/area/contact/overlap/containment features. |
| v51 `PartCATHKGStage2` | `stage2/legacy_v51.py` and `stage2/parse_scorer.py` | Main parse scorer is simplified; v51 fusion/routing remains optional. |
| Stage-2 loss/eval | `stage2/losses.py`, `training/stage2_trainer.py`, `evaluation/metrics.py` | Adds curriculum-aware parse loss. |
| Visualization | `analysis/visualize_stage1.py`, `analysis/visualize_parse.py` | Keeps the idea of object-gated role panels and edge contribution overlays. |

## Migration checklist

- Replace notebook globals (`OBJ_NAMES`, `ROLE_INDEX_TABLE`, etc.) with a `RoleSchema` instance.
- Replace direct environment variables with YAML config.
- Keep Stage 1 checkpoint loading stable before rebuilding the HKG.
- Rebuild the HKG after each meaningful Stage-1 checkpoint.
- Report parse-score accuracy separately from any optional readout/fusion accuracy.
- Use PMI only in diagnostics unless running an explicit ablation.
- Treat latent/discrete relation-code experiments as a separate branch, not the default classifier.
