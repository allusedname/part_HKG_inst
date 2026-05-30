# Instance-Slot AOG implementation patch

This patch adds a parallel Stage-2 path for the proposed Instance-Slot And-Or Graph (IS-AOG) design.
It reuses the existing Stage-1 `PartCATHKGStage1` outputs and adds:

- connected-component terminal extraction for repeated functional parts,
- a compact `InstanceAOG` grammar dataclass,
- an offline `build_instance_aog` grammar compiler,
- a `InstanceAOGStage2Classifier` parser/scorer,
- a Stage-2 trainer and command-line scripts,
- smoke tests for component splitting, serialization, builder, and forward pass.

## Build grammar

```bash
PYTHONPATH=src python scripts/build_instance_aog.py \
  --config configs/stage1_quality_upgrade.yaml \
  --stage1-ckpt runs/stage1_quality_upgrade/checkpoints/stage1_best.pt \
  --out runs/stage1_quality_upgrade/checkpoints/instance_aog.pt \
  --device auto \
  --num-templates-per-class 3 \
  --component-threshold 0.40 \
  --max-components-per-part 4
```

## Train Stage 2

```bash
PYTHONPATH=src python scripts/train_stage2_instance_aog.py \
  --config configs/stage1_quality_upgrade.yaml \
  --stage1-ckpt runs/stage1_quality_upgrade/checkpoints/stage1_best.pt \
  --grammar runs/stage1_quality_upgrade/checkpoints/instance_aog.pt \
  --save-dir runs/stage2_instance_aog \
  --device auto
```

## Tests

```bash
PYTHONPATH=src pytest -q tests/test_instance_aog.py
```

## Notes

This implementation intentionally does not impose global names like `wheel_1` or `wheel_2`.
Components are unordered terminals. A class/template owns local slots, and each forward pass solves a
latent component-to-slot assignment under that parse hypothesis. `slot_family` is diagnostic sharing across
templates, not a hard global ID.
