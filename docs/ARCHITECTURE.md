# Architecture

## Main proposal path

The main model is a visibility-aware hierarchical parse graph classifier.

1. **Stage 1 DINO-guided cost-aggregation parser**
   - builds object, functional-part, and object-aware-role cost volumes from CLIP/open_clip text prototypes;
   - embeds the functional cost volume with grouped pointwise convolutions, then aggregates spatially with optional DINO-guided attention and across functional part channels;
   - predicts object support, functional part masks, and object-specific role masks;
   - computes top-k functional/role presence and pools residual/DINO token maps through sharpened masks.

2. **HKG builder**
   - creates functional nodes and class-specific role nodes;
   - stores global functional prototypes and class-role prototypes;
   - stores PMI only as a diagnostic statistic;
   - stores typed role-edge templates using explicit geometry/contact features.

3. **Stage 2 parse scorer**
   - estimates role visibility states: visible, unknown, contradictory, absent;
   - scores visible role evidence by role-prototype similarity;
   - optionally scores top-down completion for unknown roles using functional tokens;
   - selects reliable typed relation edges and scores class-vs-global relation likelihood ratios;
   - predicts directly from the parse score.

```text
S_parse(c|x) = S_vis(c) + lambda_comp S_comp(c) + lambda_edge S_edge(c) - lambda_contr S_contr(c)
```

## Notebook features preserved as optional modules

The v51 notebook contains extra components that are useful for experiments but
not central to the simplified proposal:

- base part-token classifier;
- PMI branch;
- adaptive fusion gate;
- probability-mixture fusion;
- log-opinion-pool fusion;
- learned relation routing;
- rescue losses for base-wrong/HKG-correct examples.

These are represented in `stage2/calibration.py` and `stage2/legacy_v51.py`.
The default config disables them.

## Main design decision

Stage 1 now keeps the notebook-compatible object-aware role outputs while adding the proposal-style grouped cost aggregation and DINO-guided attention path. Stage 2's default output remains the proposal's graph parse score.
This prevents the final result from becoming a mixture of many classifier-like
experts and keeps explanations aligned with visible nodes, completion terms, and
selected typed edges.
