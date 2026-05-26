# AOG-HKG Stage 2 Implementation

This implementation turns Stage 2 into an AOG-inspired parser over the frozen
Stage-1 part evidence.

## What Stage 1 provides

The existing Stage-1 contract is preserved:

- `part_logits`, `part_prob`, `part_presence`
- `role_logits`, `role_prob`, `role_presence`
- `part_tokens`, `part_tokens_res`, `part_tokens_dino`
- `role_tokens`, `role_tokens_res`, `role_tokens_dino`
- token maps used for offline prototype pooling

Stage 1 is trained with `scripts/train_stage1.py` using BCE + Dice style losses,
AMP-safe probability-domain BCE helpers, fixed-k mean presence by default, and
mild mask pooling. Stage 2 freezes the full Stage-1 model and uses these outputs
without changing the segmentation head.

## New files

- `src/partcat_hkg/kg/aog_builder.py`
  - Builds `AOGHierarchicalKG` from a trained Stage-1 checkpoint and a full
    mask-supervised training loader.
  - Learns class alternatives by deterministic k-means over part presence and
    geometry.
  - Estimates template role priors, required-role masks, role prototypes,
    sparse relation templates, support/IG gates, and motif edges.

- `src/partcat_hkg/stage2/aog_hkg_classifier.py`
  - Implements `AOGHKGStage2Classifier`.
  - Scores node evidence, absence penalties, conflicts, relation likelihood
    ratios, motifs, and template priors.
  - Aggregates alternatives by log-sum-exp or max.
  - Fuses HKG logits with a base part-token classifier through classwise
    softplus calibration.
  - Can decode a compact per-image parse summary.

- `src/partcat_hkg/training/aog_stage2_trainer.py`
  - Freezes Stage 1.
  - Trains only Stage-2 projections, base head, term scales, and calibrator.
  - Saves `stage2_aog_hkg_last.pt`, `stage2_aog_hkg_best.pt`, and history files.

- `scripts/build_aog_hkg.py`
  - Builds and saves `aog_hkg.pt`.

- `scripts/train_stage2_aog_hkg.py`
  - Loads Stage 1 + `aog_hkg.pt` and trains the Stage-2 parser/classifier.

- `tests/test_aog_hkg_stage2.py`
  - Synthetic CPU test for the new classifier and loss.

## Relation change

`src/partcat_hkg/kg/relations.py` now computes relation geometry in the local
union-box frame of the two masks. The feature dimensionality is kept compatible
with existing tests: 14 continuous features and 8 diagnostic relation channels.

## Recommended run order

```bash
# 1. Train or load Stage 1.
python scripts/train_stage1.py \
  --config configs/default.yaml \
  --partimagenet-root /path/to/PartImageNet \
  --save-dir runs/stage1

# 2. Build the AOG-HKG from the frozen Stage-1 checkpoint.
python scripts/build_aog_hkg.py \
  --config configs/default.yaml \
  --stage1-ckpt runs/stage1/checkpoints/stage1_best.pt \
  --out runs/stage1/checkpoints/aog_hkg.pt \
  --device auto \
  --max-images-per-class 0 \
  --num-templates-per-class 3

# 3. Train Stage 2 with Stage 1 frozen.
python scripts/train_stage2_aog_hkg.py \
  --config configs/default.yaml \
  --stage1-ckpt runs/stage1/checkpoints/stage1_best.pt \
  --hkg runs/stage1/checkpoints/aog_hkg.pt \
  --save-dir runs/stage2_aog_hkg \
  --device auto
```

## Notes

- The previous `VisibilityAwareParseGraphClassifier` remains available as a
  backward-compatible baseline.
- The latent codebook branch is still not used by default. The new AOG-HKG path
  follows the explicit continuous-relation recommendation first.
- The HKG builder uses a deterministic non-sampling DataLoader rather than the
  Stage-1 weighted sampler, so KG statistics are not accidentally biased by
  replacement sampling.
