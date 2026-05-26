# AOG-HKG diagnostic interpretation and direct visualization

This note explains the diagnostic notebook figures and the added direct configuration/motif views.

## What the existing figures do

1. **HKG structure tables and histograms** show the class-level grammar before any image is parsed.
   - Template prior: how often each Or-node template branch is selected during HKG construction.
   - Edge support: how frequently a relation edge is observed inside a class-template branch.
   - Motif support: how frequently a structural motif is observed.

2. **Stage-1 sample visualization** shows the evidence available to Stage 2.
   - Raw image and GT union mask.
   - Predicted support mask.
   - Union of top predicted functional masks.
   - Top part presence scores.

3. **Stage-1 statistics** aggregate mask and presence quality.
   - mIoU/Dice measure segmentation quality.
   - Presence precision/recall/F1 measure whether the correct functional parts are active.
   - Hallucination/miss rates expose false active parts or missed parts.

4. **Stage-2 branch audit** compares final/base/HKG/node/edge/motif branches.
   - If final is high but base drops, base is being used as a residual unless base_aux is enabled.
   - If HKG is strong but edge/motif contributions are noisy, inspect relation residuals and direct configuration plots.

5. **Parse overlay** combines Stage-1 masks, selected HKG edges, and branch logits in one figure.
   - It is useful for quick inspection but can look like spaghetti if template edges are dense.
   - The helper now draws only positive contributing edges by default.

6. **Template usage** checks Or-node collapse.
   - If one template dominates every image in a class, alternatives are not meaningful.
   - If every validation example is from one class because MAX_BATCHES is too small or loader ordering is not shuffled, template usage can look artificially collapsed.

7. **Relation residual analysis** compares observed role-mask relation features with selected template means.
   - Large residuals mean the selected relation template does not match the image evidence.
   - If many residuals are large, HKG construction or template clustering is too noisy.

## What the uploaded run currently suggests

The logged Stage-1 run is usable: mIoU rises to roughly 0.55 and presence F1 reaches roughly 0.81.
The Stage-2 run shows final/HKG accuracy staying high, but the standalone base branch collapses from about 0.95 to about 0.58 by epoch 16. This is a training design issue, not a Stage-1 issue: the base head was being optimized only through fused logits and can become a residual correction. The debug-fix config sets `loss.stage2.base_aux=1.0` and initializes the HKG fusion weight with inverse-softplus so the requested initial weight is the actual weight.

The HKG build log reports `templates/class=3 edges=100 motifs=96`. For 11 classes and 3 templates per class, 96 motifs is dense. After the motif-filter fix, motifs should be much more structural. The direct motif-only plot is intended to make this immediately visible.

## New direct visualizations

The notebook now includes:

- `plot_hkg_class_templates_grid(kg, class_name)`: shows all valid template alternatives for one class.
- `plot_hkg_template_configuration(kg, class_name, template=None)`: shows the selected class-template grammar branch and motif-only view.
- `plot_parse_configuration_and_motifs(model, batch, ...)`: shows four separated panels: Stage-1 masks, observed configuration, active motifs, and selected template prior.
- `summarize_template_structure_quality(kg)`: reports edge/motif density and template prior per class-template.

Read the direct plots as follows:

- If **Stage-1 masks** are wrong, fix Stage 1 or presence thresholds first.
- If **observed configuration** is sparse and reasonable but **template prior** is dense, fix HKG building: edge filtering, motif filtering, template clustering.
- If **motif-only panel** is crowded, motif promotion is too permissive.
- If **template grid** shows almost identical alternatives, the Or-node clustering is not meaningful.
- If **template usage** always picks template 0, either the class really has one dominant view or the HKG template branches collapsed.
