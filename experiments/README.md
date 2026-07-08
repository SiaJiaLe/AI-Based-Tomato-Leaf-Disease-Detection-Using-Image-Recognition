# CP2 Solution-Stack Ablation Runner

Config-driven runner for the 12-run OFF/ON ablation (6 CNN backbones × Stack
OFF/ON). One YAML in `configs/` = one run. OFF and ON share a single code
path, so the only difference between a pair is the solution stack. See
`../CP2_implementation_plan.md` (the spec) and `../CP2_agent_plan.md` (design).

## Layout
- `configs/` — 12 run configs (`<backbone>_<off|on>.yaml`).
- `common/` — shared modules: `seeding`, `data`, `heads`, `cbam`, `backbones`,
  `engine` (two-stage training, macro-F1 checkpoint), `evaluate` (both test
  sets, gap, per-class, confusion matrices).
- `run.py` — train + evaluate a run (or `--all`).
- `compare.py` — master table, OFF/ON ablation table, gap-narrowing figure.
- `smoke_test.py` — fast structural check of all 6 backbones, no data needed.
- `results/<run_name>/` — per-run outputs: `best_model.pth` (best val macro-F1),
  `best_by_loss.pth`, `metrics.json`, `eval_results.json`,
  `eval_results_real_world.json`, confusion-matrix PNGs.

## The solution stack (Stack ON vs OFF)
| Component | OFF | ON |
|---|---|---|
| Augmentation | basic 4 (crop, hflip, rotate, brightness/contrast) | basic 4 + advanced field pipeline |
| Label smoothing | 0.0 | 0.1 |
| Head | Dropout→Linear | BatchNorm1d→Dropout→Linear→ReLU→Dropout→Linear |
| CBAM attention | none | after last two stages |
| Stage-B unfreeze | deepest stage only | deepest two stages |

Val/test always get resize+normalize only (a hard assertion enforces this).

## Run it (HPC)
```bash
git pull
pip install -r experiments/requirements.txt      # once, into tomato-ml env
python -m experiments.smoke_test                  # validate all 6 backbones first
sbatch experiments/run_all_slurm.sh               # trains+evals all 12, then compares
```
Single run for debugging:
```bash
python -m experiments.run --config experiments/configs/resnet34_on.yaml
```
Re-evaluate without retraining:
```bash
python -m experiments.run --config experiments/configs/resnet34_on.yaml --eval-only
python -m experiments.compare
```
