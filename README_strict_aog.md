# Strict Neural Spatial AOG overlay

This overlay replaces the earlier GPU IS-AOG prototype with a stricter Spatial And-Or Graph path.
It keeps Stage 1 as the neural visual-vocabulary/terminal-proposal module and makes Stage 2 an AOG parser, not a fused KG residual classifier.

## AOG mapping

- Start / class Or-node: object class roots.
- Or switch: class-specific template/view/subtype branch.
- And-node: a selected template production decomposed into slots.
- Terminal nodes: Stage-1 part/component proposals with attributes: part type, confidence, geometry, token, optional mask.
- Relations: horizontal slot-slot edges with Gaussian relation potentials.
- Probability model: class/template rule probabilities plus singleton and pairwise energies.
- Parse graph: selected class, selected template, terminal-slot address variables, and active relation edges.

## Why this fixes the earlier failure mode

The earlier prototype could return uniform logits if cached terminal features were zero, grammar coverage was empty, or a black-box base branch dominated training. This version is intentionally narrower:

1. Stage 2 trains only AOG calibration/projection parameters.
2. The parser fails fast if logits become effectively uniform.
3. The AOG logits are the primary logits; `base_logits` is only a compatibility alias.
4. There is no latent relation codebook and no motif branch.
5. Slot assignment is Sinkhorn or max address-variable inference, not a fixed global wheel_1 label.

## Commands

Cache terminals from Stage 1:

```bash
PYTHONPATH=src python scripts/cache_strict_aog_terminals.py \
  --config configs/stage1_quality_upgrade.yaml \
  --stage1-ckpt runs/stage1_quality_upgrade/checkpoints/stage1_best.pt \
  --out-dir runs/strict_aog_cache \
  --device auto \
  --splits train,val \
  --batch-size 16 \
  --threshold 0.40 \
  --max-components-per-part 4 \
  --max-terminals 32
```

Build the grammar:

```bash
PYTHONPATH=src python scripts/build_strict_aog.py \
  --config configs/stage1_quality_upgrade.yaml \
  --cache runs/strict_aog_cache/train_strict_aog_terminals.pt \
  --out runs/strict_aog_cache/strict_aog.pt \
  --num-templates-per-class 3 \
  --max-slots-per-template 12 \
  --max-slots-per-part 4
```

Train/evaluate AOG calibration:

```bash
PYTHONPATH=src python scripts/train_strict_aog.py \
  --grammar runs/strict_aog_cache/strict_aog.pt \
  --train-cache runs/strict_aog_cache/train_strict_aog_terminals.pt \
  --val-cache runs/strict_aog_cache/val_strict_aog_terminals.pt \
  --save-dir runs/strict_aog \
  --device auto \
  --batch-size 64 \
  --assignment sinkhorn
```

Fast diagnostic mode:

```bash
PYTHONPATH=src python scripts/train_strict_aog.py \
  --grammar runs/strict_aog_cache/strict_aog.pt \
  --train-cache runs/strict_aog_cache/train_strict_aog_terminals.pt \
  --val-cache runs/strict_aog_cache/val_strict_aog_terminals.pt \
  --save-dir runs/strict_aog_max \
  --device auto \
  --batch-size 128 \
  --assignment max
```

## Tests

```bash
PYTHONPATH=src pytest -q tests/test_strict_aog_core.py
```

The included tests compile the new modules, verify geometry relation tensors, build a toy repeated-part grammar, run a forward parse, compute a loss, decode a parse graph, and verify grammar save/load.

## Important limits

This code has passed syntax and unit tests in this environment. It still needs your real PartImageNet cache and Stage-1 checkpoint run. I will not call any research code literally “fully bug-free” without running it on the exact dataset/checkpoints/GPU environment, but this version removes the known design causes of the constant-uniform-logit failure.
