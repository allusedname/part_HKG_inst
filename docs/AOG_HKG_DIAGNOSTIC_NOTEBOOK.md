# AOG-HKG diagnostic notebook

Notebook: `notebooks/aog_hkg_diagnostics.ipynb`

This notebook diagnoses the intermediate states of the two-stage AOG-HKG pipeline.

## What it visualizes

### Stage 1

- raw image and object label
- ground-truth object/part union mask
- predicted support mask
- top functional part masks
- top functional part presence scores

### Stage 2

- selected class-template parse
- active functional part masks
- selected template edges drawn over the image
- final/base/HKG/node/edge/motif branch logits
- per-part node evidence, missing penalty, conflict penalty
- per-edge relation likelihood ratio and residuals
- motif contributions

## Statistical analysis included

- Stage-1 per-part IoU, Dice, presence precision/recall/F1, hallucination rate, miss rate
- Stage-2 accuracy for final/base/HKG/node/edge/motif branches
- macro accuracy, confidence, expected calibration error, true-class probability and rank
- base-to-final rescue and damage rates
- per-class accuracy and active-part statistics
- confusion matrix and calibration curve
- HKG template usage by predicted and ground-truth class
- selected-template edge residuals against relation means

## How to run

From the project root:

```bash
jupyter notebook notebooks/aog_hkg_diagnostics.ipynb
```

Then edit these variables in the first configuration cell:

```python
PARTIMAGENET_ROOT = "/path/to/PartImageNet"
STAGE1_CKPT = PROJECT_ROOT / "runs" / "stage1" / "checkpoints" / "stage1_best.pt"
HKG_PATH = PROJECT_ROOT / "runs" / "stage1" / "checkpoints" / "aog_hkg.pt"
STAGE2_CKPT = PROJECT_ROOT / "runs" / "stage2_aog_hkg" / "checkpoints" / "stage2_aog_hkg_best.pt"
MAX_BATCHES = 8
```

Start with a small `MAX_BATCHES` value to verify the setup, then increase it for stable statistics.

## Output tables

The final cell writes CSVs under:

```text
runs/diagnostics/aog_hkg/
```

These CSV files are useful for paper plots, checkpoint comparison, and targeted error analysis.
