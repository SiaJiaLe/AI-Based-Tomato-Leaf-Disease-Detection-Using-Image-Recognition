# Generic background textures (Plan 1)

Backdrops composited behind segmented leaves during training, to break the
"clean uniform background" shortcut. **These must be generic, domain-neutral
textures** — soil, grass, wood, sky, gravel, foliage.

**Rules**
- NOT PlantVillage images (their backdrops are the uniform ones we're escaping).
- NOT the real-world test images (`data/processed/real_environment_test`) —
  using them here would leak the test domain into training. The runner asserts
  this folder is disjoint from the real-world test dir.

**How it gets populated**
- Synthetic (default): `python -m experiments.plan1_bgrand.make_synthetic_backgrounds`
- Real CC0 photos (later): just drop 30–100 image files in here; the run reads
  whatever image files are present.

Images here are git-ignored (generated/local only).
