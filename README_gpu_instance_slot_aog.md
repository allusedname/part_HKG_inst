# GPU Instance-Slot AOG overlay

This overlay replaces the slow reference Instance-Slot AOG implementation with a cached, batched PyTorch version.

## What changed

The slow path did three expensive things during Stage-2 training:

1. ran frozen Stage 1 inside every Stage-2 forward pass;
2. extracted connected components with CPU/Python traversal inside the forward path;
3. looped over batch, class, template, slot, component, and relation edges in Python.

The GPU path fixes those points:

1. `scripts/cache_gpu_instance_components.py` runs frozen Stage 1 once and stores fixed-size component tensors;
2. component terminals are extracted with batched GPU local-max NMS and soft gaussian supports, not CPU connected components;
3. Stage 2 consumes cached tensors and computes slot-component compatibility as broadcasted tensors `[B,C,A,S,N]`;
4. relation scoring is vectorized from component geometry and scatter-adds edge evidence to class/template scores.

## New files

```text
src/partcat_hkg/kg/gpu_instance_components.py
src/partcat_hkg/kg/gpu_instance_aog.py
src/partcat_hkg/kg/gpu_instance_aog_builder.py
src/partcat_hkg/data/gpu_component_cache.py
src/partcat_hkg/stage2/gpu_instance_aog_classifier.py
src/partcat_hkg/training/gpu_instance_aog_trainer.py
scripts/cache_gpu_instance_components.py
scripts/build_gpu_instance_aog.py
scripts/train_stage2_gpu_instance_aog.py
tests/test_gpu_instance_aog.py
```

## Install overlay

Copy the overlay into the repo root:

```bash
cp -R gpu_instance_slot_aog_impl/* /path/to/part_HKG_inst/
cd /path/to/part_HKG_inst
```

## 1. Cache Stage-1 evidence and GPU components

```bash
PYTHONPATH=src python scripts/cache_gpu_instance_components.py \
  --config configs/stage1_quality_upgrade.yaml \
  --stage1-ckpt runs/stage1_quality_upgrade/checkpoints/stage1_best.pt \
  --out-dir runs/stage1_quality_upgrade/gpu_component_cache \
  --device auto \
  --splits train,val \
  --batch-size 32 \
  --mask-size 64 \
  --component-threshold 0.20 \
  --max-components-per-part 2 \
  --max-total-components 32
```

The cache stores CPU tensors on disk, but extraction itself runs on GPU.

## 2. Build GPU Instance-Slot AOG grammar from cache

```bash
PYTHONPATH=src python scripts/build_gpu_instance_aog.py \
  --config configs/stage1_quality_upgrade.yaml \
  --cache-dir runs/stage1_quality_upgrade/gpu_component_cache \
  --out runs/stage1_quality_upgrade/checkpoints/gpu_instance_aog.pt \
  --device auto \
  --num-templates-per-class 2 \
  --max-components-per-part 2 \
  --template-edge-max-edges 8
```

## 3. Train cached, fully GPU Stage 2

```bash
PYTHONPATH=src python scripts/train_stage2_gpu_instance_aog.py \
  --config configs/stage1_quality_upgrade.yaml \
  --cache-dir runs/stage1_quality_upgrade/gpu_component_cache \
  --grammar runs/stage1_quality_upgrade/checkpoints/gpu_instance_aog.pt \
  --save-dir runs/stage2_gpu_instance_aog \
  --device auto \
  --batch-size 64 \
  --assignment softmax
```

Fastest debugging mode:

```bash
PYTHONPATH=src python scripts/train_stage2_gpu_instance_aog.py \
  --config configs/stage1_quality_upgrade.yaml \
  --cache-dir runs/stage1_quality_upgrade/gpu_component_cache \
  --grammar runs/stage1_quality_upgrade/checkpoints/gpu_instance_aog.pt \
  --save-dir runs/stage2_gpu_instance_aog_fastmax \
  --device auto \
  --batch-size 128 \
  --assignment max
```

More parse-like but slower:

```bash
--assignment sinkhorn --class-chunk 4
```

Use `--class-chunk` to lower memory if `[B,C,A,S,N]` is large.

## Suggested starting settings

```yaml
model:
  hkg:
    num_templates_per_class: 2
    max_components_per_part: 2
    template_edge_max_edges: 8
    template_edge_min_support: 0.12
  stage2:
    isaog_assignment: softmax
    isaog_assignment_tau: 0.35
    isaog_class_chunk: 9999
    isaog_edge_scale: 0.35
    hkg_fusion_lambda_init: 0.20
training:
  batch_size_stage2: 64
```

If memory is tight, reduce `batch_size_stage2`, set `isaog_class_chunk: 4`, or use `--assignment max`.

## Smoke test

```bash
PYTHONPATH=src pytest -q tests/test_gpu_instance_aog.py
```

## Notes

This is not exact connected-component parsing. It is a fully GPU soft-terminal approximation designed for speed. Exact CPU connected components can still be used for visualization or offline analysis, but they should not run inside Stage-2 training.
