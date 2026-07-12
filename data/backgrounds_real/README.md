# Real CC0 background photos (Plan 1 — real-background run)

Drop 30–100 **real, generic, CC0-licensed** background photos here — soil,
grass, wood, foliage, greenhouse, sky, hands, gravel — for the real-background
variant of Plan 1 (`efficientnetb0_on_bgrand_real`).

**Rules (same as the synthetic folder)**
- NOT PlantVillage images.
- NOT the real-world test images (`data/processed/real_environment_test`). The
  runner asserts this folder is disjoint from the real-world test dir; using
  field test images here would leak the test domain into training.

Any image files (`.jpg/.png/...`) present are used. Images here are git-ignored.

When populated, run:
```bash
sbatch experiments/plan1_bgrand/run_bgrand_real_slurm.sh
```
