# CP2 Agent Implementation Plan — Solution-Stack Generalization Ablation

**Companion to:** `CP2_implementation_plan.md` (your spec). This document is *my* execution plan for delivering that spec: what I will build, which files change, in what order, and the decisions I need from you before I start.

**Golden rule inherited from the spec:** this is a measurement, not a contest. I will not tune, select, or reframe anything to make ResNet34 win. The real-world test set is touched exactly once per final model.

---

## 1. Current-state audit (code as it actually is today)

I read the real code, not just the plan. Findings that change the work:

| Spec assumption | Reality in code | Consequence |
|---|---|---|
| Dead `WeightedRandomSampler` to delete (3.1) | **Not present in any `train.py`/`dataset.py`.** All loaders use `shuffle=True`. Sampler text lives only in `real_world_generalization_plan.md`, `baseline_models_plan.md`, READMEs. | 3.1 becomes a **docs/report** cleanup, not a code deletion. |
| Checkpoint on macro-F1 (5) | Every `train.py` saves best on **`val_loss`**. | Must switch checkpoint criterion to **val macro-F1** for all runs. |
| Frozen split file needed (3.3, 6, 7) | Split already frozen physically: all 6 models point at the same `resnet34_model/data/processed/{train,val,test}`. | On-disk ImageFolder split already guarantees identical files. A split JSON is optional; I'll emit a **hash manifest** of the split for the audit trail instead of re-splitting. |
| Global seed per run (3.3, 7) | **No seed anywhere.** | Add a single seeding utility used by every run. Strictly, no existing run is seed-controlled → all 12 arms should be (re-)run under seeds to be a valid ablation. |
| OFF and ON differ only by the stack (2.5, 5) | ON (resnet34) and OFF (baselines) live in **separate folders with duplicated, subtly-divergent code** (e.g. ON crop `scale=0.7–1.0` + vertical flip; OFF crop `scale=0.8–1.0`, no vertical flip). Not cleanly nested. | Need a **single code path** where ON = OFF's four basic augs + advanced layered on top, so the only difference is a config flag. |
| Classical KNN/SVM/RF in comparison | `compare_models.py` hardcodes all 9 incl. classical. | Drop classical from the ablation; cite Khan/Tan instead. |

**What exists vs. what's missing in the 12-run matrix:**
- Exists: ResNet34 **ON** (`resnet34_model/`), and OFF for VGG16/ResNet50/AlexNet/MobileNetV2/EfficientNetB0.
- Missing to build: ResNet34 **OFF** (primary control) + **ON** for the five baselines.
- Caveat: because none of the existing runs were seeded or macro-F1-checkpointed, a rigorous ablation re-runs **all 12** under the unified engine, not just the 6 missing arms.

---

## 2. Proposed architecture — one config-driven runner

Replace the six duplicated folders (for *this experiment*) with a single package so OFF/ON differ by config only. The existing folders stay on disk as legacy but are no longer the source of truth for CP2 numbers.

```
experiments/
  configs/                     # one YAML per run = 12 files
    resnet34_off.yaml   resnet34_on.yaml
    resnet50_off.yaml   resnet50_on.yaml
    vgg16_off.yaml      vgg16_on.yaml
    alexnet_off.yaml    alexnet_on.yaml
    mobilenetv2_off.yaml mobilenetv2_on.yaml
    efficientnetb0_off.yaml efficientnetb0_on.yaml
  common/
    seeding.py       # seed_everything(seed): python/numpy/torch/cuda + deterministic flags
    data.py          # build_loaders(cfg): basic-4 always; advanced layered iff cfg.advanced_augmentation. Val/test = resize+normalize ONLY, with a hard assertion.
    heads.py         # plain head (Dropout->Linear) vs strong head (BN1d->Dropout->Linear->ReLU->Dropout->Linear), by feature dim per backbone
    cbam.py          # CBAM module + per-architecture insertion registry
    backbones.py     # build_backbone(name, cfg): loads pretrained, swaps head, optional CBAM, exposes stage_b param groups per arch
    engine.py        # two-stage loop, macro-F1 checkpointing, early stop, seed, config snapshot, metrics.json
    evaluate.py      # both test sets: acc/macroP/R/F1, gap, per-class, confusion matrix (raw + row-normalized PNG)
  run.py             # python -m experiments.run --config configs/resnet34_on.yaml
  compare.py         # master table + 6-pair ablation table + gap-narrowing grouped bar chart
  results/           # <run_name>/ : resolved_config.yaml, best_model.pth, metrics.json, plots, confusion matrices
```

**Config schema** (mirrors spec 6.1):
```yaml
run_name: resnet34_on
seed: 42
backbone: resnet34            # resnet34|resnet50|vgg16|alexnet|mobilenetv2|efficientnetb0
data_dir: resnet34_model/data/processed
real_world_dir: data/processed/real_environment_test
stack:
  advanced_augmentation: true   # false => OFF
  label_smoothing: 0.1          # 0.0 => OFF
  strong_head: true             # false => plain head
  cbam: true                    # false => no attention
  stage_b: two_group            # two_group (ON) | one_group (OFF)
training: { stage_a_epochs: 15, stage_b_epochs: 25, patience: 7,
            stage_a_lr: 1.0e-3, stage_b_lr: 1.0e-4, batch_size: 32 }
```
OFF preset = all `stack.*` false/0.0/one_group. ON preset = all true/0.1/two_group. Nothing else differs between a pair.

---

## 3. Component design decisions (how each stack piece is defined)

- **Basic-4 augmentation (both OFF and ON):** `RandomResizedCrop(224, scale=0.8–1.0)` + `HorizontalFlip(0.5)` + `Rotate(±20)` + `ColorJitter(brightness/contrast)`. Identical strengths in every run. Val/test: `Resize(256)->CenterCrop(224)->Normalize` only.
- **Advanced pipeline (ON only, layered on top of basic-4):** perspective, elastic, shadow, blur, gauss-noise, image-compression, coarse-dropout — ported from the current resnet34 `dataset.py`, moderate strengths, train-only.
- **Label smoothing:** `CrossEntropyLoss(label_smoothing=0.1)` ON, `0.0` OFF. Same factor all ON runs.
- **Strong head vs plain head** by backbone feature dim: ResNet34=512, ResNet50=2048, VGG16=4096, AlexNet=4096, MobileNetV2=1280, EfficientNetB0=1280.
- **CBAM insertion + Stage-B unfreeze**, per spec §5 mapping table:
  - ResNet34/50: CBAM after `layer3`,`layer4`; unfreeze `layer3`+`layer4` (ON) vs `layer4` only (OFF).
  - VGG16: CBAM after conv `block4`,`block5`; unfreeze those two (ON) vs last block (OFF).
  - AlexNet: CBAM after last two conv layers; unfreeze those (ON) vs last conv (OFF).
  - MobileNetV2: CBAM on **output of last two inverted-residual stages** (respect residual add); map to `features[...]` indices explicitly — not ResNet logic.
  - EfficientNetB0: CBAM after last two MBConv stages (on top of built-in SE — flagged for the write-up; its OFF→ON delta is the evidence).
- **Checkpoint:** save best on **val macro-F1** (compute per epoch via sklearn), replacing `val_loss`.
- **Determinism:** `seed_everything` + `torch.use_deterministic_algorithms(True)` where feasible; log seed into `resolved_config.yaml`.

---

## 4. File-by-file change list

**New (the runner):** everything under `experiments/` in §2 (≈9 modules + 12 configs + 2 SLURM wrappers).

**Modified:**
- `compare_models.py` → replaced by `experiments/compare.py`: 12 CNN rows, drop KNN/SVM/RF, add OFF/ON ablation-pair table + gap-narrowing figure.
- Docs reconciliation (spec 3.2): edit `real_world_generalization_plan.md`, `baseline_models_plan.md` to mark weighted-sampler / PlantDoc as **"considered, not applied."**

**New SLURM wrappers:** `experiments/run_all_off_slurm.sh`, `experiments/run_all_on_slurm.sh` (partition `gpu-24c-l4-4g`, `gres=gpu:l4:1`, conda `tomato-ml`), each looping its 6 configs.

**Untouched:** all `backend/`, `frontend/` work; the legacy per-model folders (left as-is for provenance).

---

## 5. Execution matrix & HPC flow

12 runs. On an L4, each CNN run is ~1–2 h; batch of 12 ≈ 12–24 h → fits one 24 h SLURM job, or split OFF/ON into two jobs.

```
git pull
sbatch experiments/run_all_off_slurm.sh   # 6 OFF runs
sbatch experiments/run_all_on_slurm.sh    # 6 ON runs
# then, once complete:
python -m experiments.evaluate --all      # both test sets, per-class, confusion matrices
python -m experiments.compare             # master + ablation tables + gap figure
```

---

## 6. Deliverables produced (maps to spec §8)

1. Master comparison table — 12 runs, both datasets, gap column.
2. Ablation table — 6 OFF/ON pairs, Δ(real-world acc), Δ(real-world macro-F1) per architecture.
3. Same-treatment architecture ranking (fair ResNet34-vs-ResNet50, both ON).
4. Per-class real-world tables (each ON model).
5. Confusion-matrix heatmaps (real-world, OFF vs ON) for at least ResNet34 + best two backbones.
6. Gap-narrowing grouped bar chart (real-world macro-F1, OFF vs ON, all six).
7. Written honest findings + Khan/Tan one-liner for the dropped classical baselines.

---

## 7. Decisions — LOCKED (approved by user 2026-07-08)

1. **Runner: unified `experiments/` config runner.** ✅ One YAML = one run; OFF/ON differ by flags only. Legacy folders kept for provenance.
2. **Scope: re-run all 12 arms** under one seed/engine. ✅ No mixing of seeded and unseeded runs.
3. **Checkpoint: save BOTH metrics.** ✅ Canonical `best_model.pth` selected on **val macro-F1** (spec §5 selection rule); additionally save `best_by_loss.pth` and log per-epoch histories of *both* val loss and val macro-F1 to `metrics.json`. Nothing discarded.
4. **Seed: 42, single seed** (spec default). Multi-seed mean±std deferred to optional/stretch.

---

## 8. Suggested sequence (after approval)

1. Docs reconciliation (sampler/PlantDoc → "considered, not applied").
2. Build `experiments/common/*` (seeding, data, heads, cbam, backbones, engine, evaluate).
3. Wire `resnet34_off.yaml` + `resnet34_on.yaml`; get the 1-vs-2 primary pair running end-to-end first (highest-value, validates the whole harness).
4. Add the other 10 configs in porting-risk order: ResNet50 → VGG16 → AlexNet → MobileNetV2 → EfficientNetB0.
5. SLURM wrappers; run all; evaluate; compare; figures.
6. Write findings honestly, including the persistent gap and its mechanism.
```

_Nothing is implemented yet — this is the plan for approval per the project's Rule 1._
