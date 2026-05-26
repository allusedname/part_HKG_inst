# Source summary used for this skeleton

## LaTeX proposal

The proposal's simplified final design says the final prediction should be the
score of an explicit visibility-aware hierarchical part-role parse graph. It
keeps interpretable functional and role nodes, explicit typed relations, role
visibility states, top-down completion, selected reliable edges, and a staged
curriculum. It removes PMI and adaptive fusion from the main classifier and
keeps them for diagnostics or ablations.

## v51 notebook

The notebook implementation supplies practical details:

- robust PartImageNet parsing and canonicalization;
- object-aware role channels such as `car:wheel` and `bird:wing`;
- Stage 1 PartCAT-style cost maps from CLIP prompts plus optional DINO features;
- role-valid supervision and invalid-role top-k suppression;
- HKG construction from Stage-1/GT masks, including prototypes and role-edge templates;
- explicit relation features from masks;
- partial-parse scoring and top-down completion;
- optional learned relation routing and adaptive log-opinion-pool fusion.

## Skeleton default

The default config uses the proposal path:

```text
Stage 1 -> HKG -> visibility-aware parse score
```

Legacy v51 fusion/routing remains available under `configs/legacy_v51.yaml`.
