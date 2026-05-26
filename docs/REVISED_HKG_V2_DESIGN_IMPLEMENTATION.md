# Revised HKG-v2 design implementation

This revision is based on the latest diagnostics:

- Stage 2 base accuracy no longer collapses after the calibrated-fusion/base-aux fix.
- Final accuracy is high, but the HKG contribution is still mostly node/part evidence.
- Edge and motif branches contribute too little; car templates are often only one edge.
- Stage 1 still hallucinates some wrong functional parts, so candidate class/template scores need stronger spurious-part control.

The revised implementation keeps the stable explicit-HKG path and does **not** re-enable the unstable latent relation codebook.  Instead, it changes the explicit HKG so relation and motif evidence can actually matter.

## Main implementation changes

### 1. Anchor-edge rescue during HKG construction

Strict information-gain-only edge pursuit made some templates too thin.  HKG-v2 still filters by information gain, but rescues stable body/frame/head anchor relations when both parts have enough template prior and enough support.

New config fields:

```yaml
model:
  hkg:
    template_edge_min_support: 0.12
    template_edge_degree_cap: 5
    template_edge_max_edges: 12
    edge_min_information_gain: 0.01
    anchor_edge_min_support: 0.18
    anchor_edge_required_prior: 0.25
    anchor_edge_max_per_template: 6
```

This should prevent the car grammar from degenerating to only one body-wheel relation, while avoiding the old dense-spaghetti failure.

### 2. Body/frame-to-appendage motif type

The earlier debug fix made motifs too conservative.  HKG-v2 adds a sparse appendage motif for stable relations such as:

- body--wheel
- body--wing
- body--tail
- body--leg/foot
- body--engine/mirror

The motif vocabulary is now:

```python
["generic", "attached", "containment", "lateral", "appendage"]
```

Generic edges are still not promoted into motifs.

### 3. Signed template-fit edge scoring

The previous edge branch used global-vs-template LLR with ReLU.  Common but useful relations often got zeroed out.  HKG-v2 changes the default to direct template-fit energy:

```yaml
model:
  stage2:
    edge_score_mode: template_fit
    hkg_edge_positive_only: false
    hkg_center_relation_scores: true
```

Relation scores are centered across valid class/template hypotheses so they become relative evidence instead of a constant negative energy.

### 4. Trainable relation and motif utility weights

In the previous version, `edge_aux` and `motif_aux` could not substantially improve relation behavior because edge/motif templates were fixed buffers.  HKG-v2 adds trainable per-edge and per-motif positive utility weights:

```python
self.edge_weight_raw  # [num_template_edges]
self.motif_weight_raw # [num_motifs]
```

This makes relation/motif auxiliary CE meaningful.

### 5. Stronger spurious-template part penalty

Diagnostics showed wrong part activations such as `foot`, `head`, `tail`, or `seat` on car images.  HKG-v2 adds a template-level spurious-part penalty: if a part is active but not expected in the candidate template, it subtracts from that candidate.

New config fields:

```yaml
model:
  stage2:
    hkg_spurious_template_penalty: 0.25
    hkg_spurious_template_tau: 0.08
```

### 6. Moderate relation auxiliaries

Because edges/motifs now have trainable utility weights, the defaults include small auxiliary terms:

```yaml
loss:
  stage2:
    edge_aux: 0.05
    motif_aux: 0.02
```

The base auxiliary loss remains enabled to prevent the old base-collapse failure.

## Recommended rerun

Rebuild the HKG and retrain Stage 2.  The old HKG checkpoint cannot show the new behavior.

```bash
python scripts/build_aog_hkg.py \
  --config configs/default.yaml \
  --stage1-ckpt runs/stage1/checkpoints/stage1_best.pt \
  --out runs/stage1/checkpoints/aog_hkg_v2.pt \
  --device auto \
  --num-templates-per-class 3
```

```bash
python scripts/train_stage2_aog_hkg.py \
  --config configs/default.yaml \
  --stage1-ckpt runs/stage1/checkpoints/stage1_best.pt \
  --hkg runs/stage1/checkpoints/aog_hkg_v2.pt \
  --save-dir runs/stage2_aog_hkg_v2 \
  --device auto
```

Update the diagnostic notebook paths accordingly.

## Expected diagnostics after rerun

Healthy behavior should look like:

- `base_acc` stays stable and competitive.
- `final_acc` improves over base or at least does not damage base.
- Template graphs are not dense, but car templates should usually have more than one relation.
- Motifs should be sparse but not always empty.
- `edge_logits` and `motif_logits` should no longer be numerically flat/zero.
- Selected parse overlays should show a few meaningful relation factors, not only node evidence.

If edge/motif evidence is still weak after this rerun, the next necessary step is true instance-role expansion, especially splitting repeated parts such as car wheels into components and assigning them to front/rear slots.
