# Stage 1 implementation notes

This update fills in the Stage 1 implementation from the proposal and the previous notebook export.

## Implemented path

Stage 1 is now a runnable `PartCATHKGStage1` module that produces:

- functional part mask logits/probabilities;
- object-aware role mask logits/probabilities;
- object support logits;
- top-k or top-q part/role presence scores;
- mask-sharpened, presence-gated part and role tokens;
- residual token maps used by Stage 2 and KG construction.

The functional part cost volume follows the proposal design: visual features are projected into the CLIP text space, dot-product costs are built against functional part prototypes, then the costs are passed through grouped per-part embeddings, optional DINO-guided spatial aggregation, and optional across-part aggregation.


## PartImageNet layout fix

The Stage-1 data path now mirrors the v51 notebook exactly. Given:

```python
PARTIMAGENET_ROOT = "../full_hyco/PartImageNet"
```

the loader resolves:

```text
../full_hyco/PartImageNet/annotations/train/train.json
../full_hyco/PartImageNet/annotations/val/val.json
../full_hyco/PartImageNet/images/train
../full_hyco/PartImageNet/images/val
```

It also keeps legacy/override support for flat `train.json` experiments and explicit `--train-annotations`, `--val-annotations`, `--train-image-root`, and `--val-image-root` flags.

## Runtime notes

Use the synthetic smoke command first when checking a fresh environment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/train_stage1.py \
  --config configs/minimal_debug.yaml \
  --synthetic-smoke \
  --device cpu \
  --torch-threads 1
```

Use real data smoke before training. The dataset loader expects the v51 notebook/original PartImageNet tree, not a flat `train.json` file at the root:

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

This means a root such as `../full_hyco/PartImageNet` should be passed directly. The loader resolves:

- `../full_hyco/PartImageNet/annotations/train/train.json`
- `../full_hyco/PartImageNet/annotations/val/val.json`
- `../full_hyco/PartImageNet/images/train`
- `../full_hyco/PartImageNet/images/val`

The sibling `annotations/train_whole` and `annotations/val_whole` directories are object-level/whole-object annotations and are not used for supervised functional part-mask extraction in Stage 1.

You can inspect the resolved paths without constructing the dataset:

```bash
PYTHONPATH=src python scripts/train_stage1.py \
  --config configs/default.yaml \
  --partimagenet-root ../full_hyco/PartImageNet \
  --print-data-layout
```

Train Stage 1:

```bash
PYTHONPATH=src python scripts/train_stage1.py \
  --config configs/default.yaml \
  --device cuda \
  --partimagenet-root ../full_hyco/PartImageNet \
  --save-dir runs/stage1_default \
  --epochs 18 \
  --torch-threads 1
```

## Edited files in this update

- `Makefile`
- `README.md`
- `STAGE1_EDITED_FILES.txt`
- `configs/default.yaml`
- `configs/minimal_debug.yaml`
- `docs/ARCHITECTURE.md`
- `docs/STAGE1_IMPLEMENTATION.md`
- `notebooks/README.md`
- `notebooks/run_stage1.ipynb`
- `scripts/smoke_stage1_synthetic.py`
- `scripts/train_stage1.py`
- `src/partcat_hkg/analysis/visualize_stage1.py`
- `src/partcat_hkg/config.py`
- `src/partcat_hkg/data/transforms.py`
- `src/partcat_hkg/data/partimagenet.py`
- `src/partcat_hkg/data/loaders.py`
- `src/partcat_hkg/evaluation/__init__.py`
- `src/partcat_hkg/evaluation/metrics.py`
- `src/partcat_hkg/evaluation/stage1.py`
- `src/partcat_hkg/models/backbones.py`
- `src/partcat_hkg/models/losses.py`
- `src/partcat_hkg/models/pooling.py`
- `src/partcat_hkg/models/stage1.py`
- `src/partcat_hkg/models/text_prototypes.py`
- `src/partcat_hkg/training/stage1_trainer.py`
- `tests/test_stage1.py`
- `tests/test_stage1_smoke.py`
- `tests/test_partimagenet_layout.py`
- `tests/test_partimagenet_paths.py`

## Validation performed

- Synthetic Stage-1 forward/loss smoke passed on CPU.
- Unit tests passed with `PYTHONPATH=src pytest -q`.
- Source compilation passed with `python -m compileall`.

The implementation is a research-ready Stage-1 code path, not a trained checkpoint. Full metrics still require running on the actual PartImageNet-style train/validation splits.
