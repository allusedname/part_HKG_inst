# PartCAT-HKG project skeleton

This repository is a modular rewrite target for the uploaded LaTeX proposal
(`partcat_hkg_simplified_final_proposal.tex`) and the previous v51 notebook
(`revised_v51_partcat_hkg_logopinion_confidence_fusion.ipynb`).

The default code path follows the simplified proposal:

```text
image
  -> Stage 1 DINO-guided, text-conditioned functional / role parser
  -> functional / role masks, presence scores, mask-sharpened tokens
  -> HKG builder with functional nodes, role nodes, typed role-edge templates
  -> Stage 2 visibility-aware parse graph classifier
  -> S_parse(c | x) = S_visible + lambda_comp S_completion + lambda_edge S_edge - lambda_contr S_contradiction
```

PMI, adaptive fusion, probability mixture, log-opinion-pool fusion, and learned
relation routing are kept as optional legacy/ablation modules rather than the
main classifier. This mirrors the proposal's convergence-aware simplification
while preserving useful implementation details from the notebook.

## Repository layout

```text
partcat_hkg_project_skeleton/
  configs/                       YAML configs for final and debug modes
  docs/                          architecture and notebook migration notes
  notebooks/run_stage1.ipynb     notebook wrapper with exact Stage-1 commands
  scripts/                       command-line entry points
  src/partcat_hkg/
    data/                        PartImageNet parsing, canonicalization, schema
    models/                      Stage 1 parser, backbones, text prototypes
    kg/                          HKG dataclasses, builder, relations, diagnostics
    stage2/                      parse scoring, completion, edge factors, fusion ablations
    training/                    training loops, curriculum, checkpointing
    evaluation/                  metrics, occlusion/counterfactual evaluation
    analysis/                    visualization and reports
    utils/                       AMP, seed, IO, numerical helpers
  tests/                         smoke tests for schema, relations, visibility, Stage 1
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# optional, for CLIP/DINO text and feature backbones
pip install -e '.[vision]'
```

## Stage 1 command

Debug/smoke run, with random text prototypes and no DINO dependency:

```bash
python scripts/train_stage1.py --config configs/minimal_debug.yaml --synthetic-smoke --device cpu --torch-threads 1
```

Smoke-test Stage 1 with the default config and your real PartImageNet-style data. The loader now follows the notebook layout, not a flat `<root>/train.json` layout:

```text
PARTIMAGENET_ROOT/
  annotations/train/train.json
  annotations/val/val.json
  images/train/...
  images/val/...
```

Check the resolved data tree first:

```bash
PYTHONPATH=src python scripts/train_stage1.py \
  --config configs/default.yaml \
  --partimagenet-root ../full_hyco/PartImageNet \
  --print-data-layout
```

Then run a real-data smoke test:

```bash
PYTHONPATH=src python scripts/train_stage1.py \
  --config configs/default.yaml \
  --device auto \
  --partimagenet-root ../full_hyco/PartImageNet \
  --smoke-only \
  --num-workers 0 \
  --batch-size 2 \
  --torch-threads 1
```

Train Stage 1 with useful overrides:

```bash
python scripts/train_stage1.py \
  --config configs/default.yaml \
  --device cuda \
  --epochs 18 \
  --save-dir runs/stage1_default \
  --partimagenet-root /path/to/PartImageNet
```

If your local data uses nonstandard names, override the split paths explicitly:

```bash
PYTHONPATH=src python scripts/train_stage1.py \
  --config configs/default.yaml \
  --partimagenet-root /path/to/PartImageNet \
  --train-annotations annotations/train/train.json \
  --val-annotations annotations/val/val.json \
  --train-image-root images/train \
  --val-image-root images/val
```

The best checkpoint is saved to:

```text
runs/<name>/checkpoints/stage1_best.pt
```

and the training audit history is saved to:

```text
runs/<name>/stage1_history.csv
runs/<name>/stage1_history.json
```

## Full pipeline sequence

```bash
python scripts/train_stage1.py --config configs/default.yaml
python scripts/build_hkg.py --config configs/default.yaml --stage1-ckpt runs/default/checkpoints/stage1_best.pt
python scripts/train_stage2.py --config configs/default.yaml --stage1-ckpt runs/default/checkpoints/stage1_best.pt --hkg runs/default/checkpoints/hkg.pt
python scripts/evaluate.py --config configs/default.yaml --stage1-ckpt runs/default/checkpoints/stage1_best.pt --hkg runs/default/checkpoints/hkg.pt --stage2-ckpt runs/default/checkpoints/stage2_best.pt
```

## Stage 1 implementation status

Stage 1 is now implemented as a runnable DINO-guided cost-aggregation parser:

- CLIP/open_clip prompt-ensemble prototypes with deterministic fallback;
- ResNet spatial features plus optional frozen DINO/timm structural features;
- grouped functional-part cost embedding;
- optional DINO-guided spatial self-attention and across-part aggregation;
- functional, object-support, object-aware role, and object-logit heads;
- top-k presence scores and mask-sharpened part/role token pooling;
- Stage-1 loss with support, functional BCE+Dice, role BCE+Dice, invalid-role regularization, functional-role consistency, object-part composition, area/coverage terms, and DINO-affinity smoothing;
- training loop with validation audit, best/last checkpoints, CSV/JSON history, and notebook command wrapper.

The remaining large-scale work is experiment-side: local DINO/CLIP weight availability, full Stage-1 pretraining, KG rebuilding, Stage-2 training, and final visual diagnostics.
