# Plan 2 — EfficientNetB0 Architecture Modification (Generalization-Motivated)

**Owner:** Sia Jia Le (22062566)
**Depends on:** Plan 1 outcome. Build these modifications **on top of the best augmentation config from Plan 1** (if background-randomization helped, that is the new baseline; if not, baseline is `efficientnetb0_stack_on`). Each modification is one new row against that fixed baseline.
**Goal:** Test whether architecture-level changes *with a domain-shift rationale* further reduce the real-world gap — and be able to justify each to an examiner as targeting unseen-field robustness, not lab accuracy.

> **The governing principle.** Your gap is domain shift, not lab difficulty. So every candidate here is chosen because its *mechanism* discourages reliance on PlantVillage-specific shortcuts — NOT because it lifts benchmark accuracy. A modification that raises PlantVillage accuracy but not real-world has not helped your problem; judge every row on the **real-world macro-F1 and gap columns.**

> **Why not "just add another attention block."** EfficientNetB0 already contains squeeze-and-excitation (channel attention) in every MBConv, and your Stack-ON already added CBAM. A third attention mechanism has diminishing, hard-to-justify returns and invites the examiner question "why three?". None of the tiers below is "more attention." Each has a regularization / invariance rationale instead.

---

## 1. Hygiene (identical discipline to every other row)

1. **One variable per row.** Each modification changes exactly one architectural thing versus the Plan-1 baseline. Never stack two modifications in one run — if Tier 1 and Tier 2 both help, test Tier 1+2 as a *separate* explicit row.
2. **Frozen split, seed 42, same budget/engine.** Reuse `data/processed`, `common/seeding.py`, `common/engine.py`.
3. **Select on PlantVillage val macro-F1; read real-world once per row.** As you add candidate rows, the temptation to peek at real-world to decide "keep iterating?" multiplies — resist it. Keep/discard on val; real-world read last.
4. **Keep the original.** The unmodified baseline row stays in the table untouched. Modifications are *additional* rows, exactly as you planned.

---

## 2. The candidate menu — three tiers by complexity/risk

Each tier states the **mechanism**, the **generalization rationale** (what you tell an examiner), the **implementation**, and the **risk**. Try in order; stop when you get a real-world gain or exhaust the ones you're willing to build.

### Tier 1 — Stochastic depth + head regularization (LOW risk, do first)

**Mechanism.** Randomly drop entire MBConv residual branches during training (stochastic depth / "drop-path"), and add dropout in the classifier head.

**Generalization rationale (examiner-facing).** Stochastic depth prevents deep blocks from co-adapting to source-domain-specific feature combinations — it's a regularizer that forces redundancy, so the network can't rely on one fragile PlantVillage-specific pathway. This is a *documented* out-of-distribution regularizer, not an accuracy trick. Reduced co-adaptation → less overfitting to the lab domain → smaller field gap. You can cite the stochastic-depth literature (Huang et al., 2016) and the general principle that regularization improves domain transfer.

**Implementation.**
- EfficientNet in `timm` exposes `drop_path_rate` directly — set it (e.g. 0.2–0.3) at model construction. If using torchvision, insert drop-path in the MBConv residual add.
- Add `Dropout(p=0.3–0.5)` in your two-layer head before the final linear.
- Both are single-line/config changes. Lowest-effort, most-defensible starting point.

**Risk.** Minimal. Worst case it slightly lowers lab accuracy with no real-world gain → report as bounded negative.

**Rows:** `..._droppath` (and optionally a small sweep of `drop_path_rate ∈ {0.2, 0.3}`, selected on val).

---

### Tier 2 — Input-resolution / receptive-field adjustment (LOW–MEDIUM risk)

**Mechanism.** EfficientNetB0's native resolution is 224×224 (your current input). Field images are noisier and symptoms appear at different scales; a modest resolution increase (e.g. 240–260) gives the network more spatial detail to distinguish lesions from clutter.

**Generalization rationale.** Domain shift includes scale/detail differences between clean close-ups (PlantVillage) and variable-distance field shots. More input resolution can help the network attend to lesion *texture* (which transfers) rather than global shape/background (which doesn't). This is a principled, cheap knob — and EfficientNet's compound-scaling design explicitly couples resolution to the architecture, so it's an architecturally-motivated change, not an arbitrary one.

**Implementation.**
- Change the resize target and adapt the first layers' expected input; retrain. Keep it modest (memory cost grows with resolution²).
- **Caveat:** this changes preprocessing, so ensure val/test use the *same* new resolution — consistency preserved, just at a new size.

**Risk.** Medium — larger memory/time; too-large a jump can hurt if pretrained weights were tuned for 224. Keep the increment small.

**Rows:** `..._res240`.

---

### Tier 3 — Domain-generalization objective (HIGHER risk, strongest contribution if it works)

**Mechanism.** Instead of adding a *module*, add a *loss/training-objective* term that explicitly encourages domain-invariant features. Two condition-1-legal options (neither uses real-world data in training):

- **3a. Heavy feature-space augmentation consistency (e.g. MixStyle).** MixStyle mixes feature-map statistics (mean/variance) across training samples inside the network, synthesizing novel "styles" so the model can't rely on PlantVillage's specific style statistics. It's *designed* for domain generalization and inserts after early MBConv stages. This is the most on-target candidate for your exact problem — its entire purpose is closing lab→field style gaps without seeing the target domain.
- **3b. Self-supervised auxiliary head** (e.g. predict rotation / solve a jigsaw on the leaf) trained jointly, forcing features that capture leaf structure rather than background cues.

**Generalization rationale.** These directly optimize for domain-invariance rather than source accuracy — the most principled possible answer to "why would this help unseen fields?" MixStyle in particular has published evidence on exactly the lab→field style-shift problem you have. This is the modification most defensible as a *generalization* contribution rather than a benchmark tweak.

**Implementation.**
- MixStyle: insert MixStyle layers after the first 1–2 MBConv stages (train-mode only; disabled at eval). Reference implementation is public and small.
- Auxiliary SSL head: add a small head + auxiliary loss term; weight it modestly.

**Risk.** Higher — more moving parts, an extra loss weight to tune (on val), and integration effort. But if any single change closes the gap, this class is the likeliest, and it's the strongest story: "I added a domain-generalization objective and the field gap narrowed."

**Rows:** `..._mixstyle` (and, if pursued, `..._ssl_aux`).

---

## 3. Config pattern (same for every tier)

```yaml
run_name: efficientnetb0_bestaug_droppath      # one modification named per run
seed: 42
backbone: efficientnetb0
split_file: data/processed
realworld_dir: data/processed/real_environment_test
solution_stack:                                 # inherited from Plan-1 best baseline
  advanced_augmentation: true
  label_smoothing: 0.1
  strong_head_bn: true
  cbam: true
  stage_b_unfreeze: [last_two_mbconv]
  background_randomization: {enabled: true, prob: 0.5}   # if Plan 1 won; else absent
architecture_mod:                               # the ONE new thing this row tests
  drop_path_rate: 0.2                           # Tier 1
  head_dropout: 0.3                             # Tier 1
  # input_resolution: 240                       # Tier 2 (separate row)
  # mixstyle: {enabled: true, layers: [1,2]}    # Tier 3 (separate row)
training:
  stage_a_epochs: 15
  stage_b_epochs: 25
  patience: 7
  stage_a_lr: 1.0e-3
  stage_b_lr: 1.0e-4
  batch_size: 32
eval:
  plantvillage_test: true
  realworld_test: true                          # read ONCE per row
```

---

## 4. Execution order

1. [ ] Confirm the Plan-1 baseline (best augmentation config) is locked and its metrics recorded.
2. [ ] **Tier 1** (`droppath` + head dropout) — cheapest, most defensible. Select rate on val; read real-world once.
3. [ ] **Tier 2** (`res240`) only if Tier 1 plateaus and you want another cheap lever.
4. [ ] **Tier 3** (`mixstyle`) if you want the strongest generalization contribution and can take the integration effort. This is the one most aligned with your exact gap.
5. [ ] If any tier helps, test the **combination** as an explicit separate row (e.g. `droppath + mixstyle`) — never assume additivity.
6. [ ] Emit the standard metric set per row; compare each against the fixed baseline.

---

## 5. Interpreting results (same metric discipline)

- Judge every row on **real-world macro-F1 and gap.** Lab accuracy is a secondary sanity check, not the target.
- A modification that helps real-world → keep as a new best; discuss its mechanism honestly.
- A modification that doesn't → **keep it in the report as a tried-and-bounded negative.** "We tested stochastic depth / resolution / MixStyle; none further closed the gap beyond augmentation" is a legitimate, informative finding that strengthens your honesty and directly supports the thesis that domain shift is hard to close with model-side changes alone.
- **The honest ceiling:** none of these is guaranteed to close a 57-point gap. Realistically they move real-world by single-digit points if they work. The likely overall conclusion — "augmentation and architecture tweaks narrow the gap modestly but a substantial field gap persists, because full closure needs real field data (condition 2)" — is your genuine contribution, and every negative row here reinforces it rather than weakening it.

---

## 6. Report placement
- Method → new subsection under **3.2.2** (architectural modifications), each with its generalization rationale + citation (stochastic depth: Huang et al. 2016; MixStyle: Zhou et al. 2021).
- Result → Results chapter, EfficientNetB0 modification rows against the fixed baseline.
- Interpretation → Discussion: which mechanism (if any) helped, why, and the persistent-gap conclusion tied to Mohanty (2016), Fenu & Malloci (2022), Tang (2025).
- Frame the *selection of EfficientNetB0* as: "largest generalization improvement under the solution stack **and** most parameter-efficient backbone (≈5.3M params)," not "highest accuracy" — the efficiency-plus-responsiveness justification, not a leaderboard one.
