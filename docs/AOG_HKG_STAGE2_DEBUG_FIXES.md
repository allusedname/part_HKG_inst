# AOG-HKG Stage 2 debug fixes

This patch addresses the failure mode observed in `run.ipynb` where final/HKG accuracy stays high while standalone base accuracy drops, and where motif/template visualizations look overly dense.

Changes:

1. Initialize `hkg_lambda_raw` with inverse-softplus so `hkg_fusion_lambda_init` is the actual initial fusion weight. The previous raw value of 0.20 became a fusion weight of about 0.80.
2. Set `loss.stage2.base_aux=1.0` so the base branch remains a real classifier instead of becoming an unconstrained residual for the fused classifier.
3. Honor the configured Stage-2 curriculum: edge/motif factors are disabled during the early warmup epochs.
4. Build the AOG-HKG from the deterministic training split by disabling random train-time flips/jitter during grammar construction.
5. Filter template edges by information gain, not only support. This prevents high-frequency but generic co-occurrence edges from dominating templates.
6. Promote only structural edges to motif factors and reduce default motif scale. Generic high-support pairs remain edges but no longer become motifs.
7. Visualization now draws only positive, contributing selected edges instead of drawing every selected template edge.

Recommended rerun:

```bash
python scripts/build_aog_hkg.py --config configs/default.yaml --stage1-ckpt runs/stage1/checkpoints/stage1_best.pt --out runs/stage1/checkpoints/aog_hkg.pt --device auto --num-templates-per-class 3
python scripts/train_stage2_aog_hkg.py --config configs/default.yaml --stage1-ckpt runs/stage1/checkpoints/stage1_best.pt --hkg runs/stage1/checkpoints/aog_hkg.pt --save-dir runs/stage2_aog_hkg_debugfix --device auto
```
