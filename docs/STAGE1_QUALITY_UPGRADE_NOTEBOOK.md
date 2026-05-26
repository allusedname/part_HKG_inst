# Stage 1 Segmentation Quality Upgrade Notebook

This notebook adds a runnable Stage-1 fine-tuning and audit path focused on the failure modes seen in the latest HKG diagnostics: hallucinated functional parts, noisy presence scores, support leakage, and weak small-part boundaries. It does not replace the default pipeline; it provides a controlled upgraded Stage-1 run that can be compared against the existing checkpoint.

Main additions:

- explicit functional-part presence BCE;
- absent-part top-mean false-positive penalty;
- class-aware invalid functional-part penalty;
- part-inside-support containment losses;
- boundary Dice loss;
- focal BCE and Tversky terms for small/rare parts;
- detailed per-part IoU, hallucination, miss, area, and presence diagnostics;
- commands for rebuilding AOG-HKG from the upgraded checkpoint with predicted Stage-1 evidence.

Recommended run:

```bash
jupyter notebook notebooks/stage1_segmentation_quality_upgrade.ipynb
```

Then set `PARTIMAGENET_ROOT`, optionally `WARM_START_CKPT`, and run the cells in order.
