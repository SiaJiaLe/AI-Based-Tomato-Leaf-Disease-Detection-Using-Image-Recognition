# Plan 7 — Visual PlantVillage vs real-world for the two structural-zero classes

## Why
Target_Spot (40 real images) and Tomato_mosaic_virus (32) score EXACTLY 0 correct in
every one of the 19 evaluated models — below the ~4/40 that random guessing over 10
classes would give. That is not "hard"; it means the models have a confident, wrong,
consistent answer. Two hypotheses, and a side-by-side image grid separates them:
  (a) DOMAIN SHIFT — the real images don't resemble PlantVillage's version of the
      class (so the learned appearance simply doesn't transfer), or
  (b) LABEL/DEFINITION MISMATCH — the real images genuinely look like the disease the
      model predicts (Target_Spot -> Septoria/Late_blight; mosaic -> Early/Late_blight),
      i.e. a data problem, not a modelling one.

## Deliverable
A new, additive, standalone script `experiments/domain_gap_samples.py` that builds a
comparison figure. Touches no isolation-contract file; needs only the images on the HPC.

For each of the two classes it renders a labelled block:
- Row A: N PlantVillage samples of the TRUE class (what the model learned)
- Row B: N real-world samples of the TRUE class (what it is tested on, all 0/N correct)
- Each block's caption states the counts and where the misses actually go, from the
  confusion analysis already generated:
    Target_Spot (real): 40 imgs, 0 correct; -> Septoria ~50%, Late_blight ~20-55%
    mosaic_virus (real): 32 imgs, 0 correct; -> Early_blight / Late_blight

Optional Row C (flag `--with-predicted`, default off, slightly more effort): N
PlantVillage samples of the DOMINANT predicted class, so the reader can judge "do the
real Target_Spot leaves look more like PlantVillage Septoria than PlantVillage
Target_Spot?" — the direct test of hypothesis (b). v1 ships A+B; C is a quick follow-up.

## Interface
    python -m experiments.domain_gap_samples
    python -m experiments.domain_gap_samples --n 8 --pv-split train --seed 0
    python -m experiments.domain_gap_samples --classes Tomato___Target_Spot --with-predicted

- `--classes` default = the two structural-zero classes.
- `--pv-split` default = `test` (the controlled set that scored ~0.95), so the contrast
  is "the distribution it aced" vs "the field". Override to `train`/`val`.
- `--n` samples per row (default 8), random but `--seed`-fixed for reproducibility.
- Writes `experiments/results/confusion/domain_gap_<class>.png` per class plus a combined
  `domain_gap_targetspot_mosaic.png`. PNG so it drops next to the other confusion figures.
- Reads real images from `data/real_environment_dataset/<class>` (the cleaned 333 set);
  PlantVillage from `data/processed/<split>/<class>`. Fails clearly if a folder is
  missing/empty (same disease as the eval traps: a silent empty folder is worse than a
  loud error). ASCII-only console output.

## Implementation notes
- PIL to load, thumbnail to a fixed size; matplotlib grid, `imshow`, axis off, per-image
  filename as a tiny caption so a striking example can be found again.
- No model, no inference in v1 -> runs on a login node, no GPU.
- Deterministic sampling (seeded) so the figure is reproducible for the write-up.

## Test
Offline `tmp/test_domain_gap.py`: build a fake data tree of coloured dummy JPEGs for
both sources/classes, run the builder, assert the PNGs exist and are non-trivial in
size, and that a missing/empty class folder raises rather than silently render blanks.
Skip cleanly if PIL/matplotlib are absent locally.

## After approval
Build script + test -> run test -> show diff -> commit/push. Then on HPC:
`git pull` -> `python -m experiments.domain_gap_samples` -> scp the PNG to view.
