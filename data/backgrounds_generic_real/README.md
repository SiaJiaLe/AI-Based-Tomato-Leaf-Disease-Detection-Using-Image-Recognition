# Real CC0 background photos — `efficientnetb0_on_bgrand_real`

Drop 30–100 **real, generic, CC0-licensed** background photos here — soil,
grass, wood, foliage, greenhouse, sky, hands, gravel. These are the backdrops
composited behind segmented leaves for the real-background variant of Plan 1.

This is the "real" half of the synthetic-vs-real comparison; its sibling
`data/backgrounds_generic_synthetic/` holds the procedural textures generated
by `make_synthetic_backgrounds.py`.

**Rules (these protect the result — do not bend them)**
- **NOT PlantVillage images.** PlantVillage backdrops are the uniform ones the
  method is trying to escape; using them defeats the purpose.
- **NOT the real-world test images** (`data/processed/real_environment_test`).
  `run_bgrand.py` asserts this folder is disjoint from the real-world test dir
  — using field test images here would leak the test domain into training and
  invalidate the measurement.

These are **backdrops, not training data**. The training source stays
`data/processed/train`. With `prob: 0.5`, half of each epoch's images pass
through untouched, so the model still learns the originals.

Any image files (`.jpg/.png/...`) here are used. Images are git-ignored.

When populated, run:
```bash
sbatch experiments/plan1_bgrand/run_bgrand_real_slurm.sh
```
