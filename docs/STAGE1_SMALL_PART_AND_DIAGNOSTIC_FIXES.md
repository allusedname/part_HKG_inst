# Stage 1 small-part and diagnostic fixes

This update addresses the failure mode observed in the Stage-1 quality notebook:
large parts such as body and wheel improved, but small real parts such as mirror
were still weak, and the earlier diagnostic plots overestimated improvement by
counting empty-empty channels as IoU = 1.

## Implementation changes

### 1. High-resolution refinement branch

`Stage1Config` now supports:

```yaml
model:
  stage1:
    use_highres_refine: true
    highres_refine_dim: 64
```

When enabled, `PartCATHKGStage1` refines part/support logits using the higher
resolution ResNet skip feature map. This is intended for small parts whose masks
are lost by the coarse low-resolution decoder.

### 2. Small-present-part adaptive loss weighting

The quality-upgrade loss now computes a `[B,K]` adaptive weight from GT mask
area. Only GT-present small parts are boosted. Absent parts are not boosted.

This prevents the previous tradeoff where stronger false-positive suppression
made rare valid parts harder to learn.

### 3. Valid-absent vs invalid-absent suppression

The old absent-part loss suppressed every absent channel equally. The new loss
separates:

- valid but absent for this image: weak suppression;
- invalid for this object class: strong suppression.

This is important for parts such as `mirror`, which are valid for cars but small
and intermittently visible.

### 4. Corrected IoU/Dice diagnostics

`evaluate_stage1_quality_detailed` no longer rewards empty-empty masks. If a part
is not GT-present in the diagnostic slice, its `iou_present` and `dice_present`
are `NaN` rather than 1.0.

The evaluator now reports:

- `iou_present` and `dice_present` for GT-present examples only;
- `iou_global_nonempty` and `dice_global_nonempty` when prediction or GT is nonempty;
- presence precision/recall/F1;
- hallucination and miss rates;
- per-class/per-part rows.

### 5. Corrected notebook visualization

The notebook now:

- always includes GT-present parts in the mask panel before top predicted parts;
- displays predicted area, top-mean probability, max probability, presence, and IoU;
- fixes the overlap colors: red = pred only, green = GT only, blue = overlap;
- loads old Stage-1 checkpoints with `strict=False` because the high-res branch adds new parameters.

### 6. Quality-weighted HKG prototype construction

`build_aog_hkg` now computes a prototype-quality weight from frozen Stage-1
predicted masks and presence scores.  Class/template prototypes are averaged
with these weights instead of treating weak leaked masks and high-quality masks
equally.  Part presence priors still use GT semantic presence, but appearance
prototypes are less contaminated by weak small-part masks or background leakage.

## Recommended run

```bash
jupyter notebook notebooks/stage1_segmentation_quality_upgrade.ipynb
```

Set:

```python
RUN_TRAIN = True
```

After training, rebuild HKG from the upgraded checkpoint:

```bash
python scripts/build_aog_hkg.py \
  --config configs/stage1_quality_upgrade.yaml \
  --stage1-ckpt runs/stage1_quality_upgrade/checkpoints/stage1_quality_best.pt \
  --out runs/stage1_quality_upgrade/checkpoints/aog_hkg_from_quality_stage1.pt \
  --device auto
```

## What to look for

A useful improvement should show:

- `body` and `wheel` remain strong;
- hallucination on invalid car parts remains low;
- small valid parts such as `mirror` improve in `iou_present`, not only in presence F1;
- per-class/per-part diagnostics cover the classes of interest, not only one class slice.
