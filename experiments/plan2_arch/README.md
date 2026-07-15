# Plan 2 — EfficientNetB0 architecture modifications (isolated)

Each tier is a **standalone row** against the fixed `efficientnetb0_on`
baseline, changing exactly one architectural thing. Tiers are never bundled
with each other, and never with Plan 1's background randomization (which was a
bounded negative, so the baseline stays `efficientnetb0_on`).

The results table therefore reads:

| row | what it is |
|---|---|
| `efficientnetb0_off` / `efficientnetb0_on` | the OFF/ON ablation baselines |
| `efficientnetb0_on_bgrand` | Plan 1 — background randomization (negative) |
| `efficientnetb0_on_droppath02/03` | **Plan 2 Tier 1 — stochastic depth** |
| `..._res240` | Plan 2 Tier 2 — resolution bump (not built yet) |
| `..._mixstyle` | Plan 2 Tier 3 — MixStyle (not built yet) |

## Tier 1 — stochastic depth (drop-path)

**The one variable:** `drop_path_rate` passed to `timm.create_model`. timm
distributes it linearly across MBConv blocks, so the value is the *maximum*
rate, applied at the deepest block.

**Rationale (examiner-facing):** stochastic depth stops deep blocks from
co-adapting into one fragile PlantVillage-specific pathway, forcing redundant
features. It is a documented out-of-distribution regularizer (Huang et al.,
2016), not an accuracy trick — so the row is judged on real-world macro-F1 and
the gap, not lab accuracy.

**Note for the report:** the plan's Tier 1 also suggests adding head dropout,
but `common/heads.py:strong_head` already has `Dropout(0.4)` + `Dropout(0.3)`
and the baseline uses it. Head dropout is therefore already in the baseline;
adding more would be a *second* variable and break attribution. Tier 1 is
drop-path only.

## Running it

```bash
sbatch experiments/plan2_arch/run_droppath_slurm.sh
```

That does the whole hygienic sequence:

1. Train `drop_path_rate=0.2` and `=0.3` with `--train-only` (no test set read).
2. `select_on_val.py` — pick the winner on **PlantVillage val macro-F1**.
3. `run_arch.py --eval-only` on the winner — real-world read **once**.
4. `compare_arch.py` — deltas vs `efficientnetb0_on`.

Manual equivalent:

```bash
python -m experiments.plan2_arch.run_arch --config experiments/plan2_arch/configs/efficientnetb0_on_droppath02.yaml --train-only
python -m experiments.plan2_arch.run_arch --config experiments/plan2_arch/configs/efficientnetb0_on_droppath03.yaml --train-only
python -m experiments.plan2_arch.select_on_val efficientnetb0_on_droppath02 efficientnetb0_on_droppath03
python -m experiments.plan2_arch.run_arch --config experiments/plan2_arch/configs/<winner>.yaml --eval-only
python -m experiments.plan2_arch.compare_arch --run <winner>
```

## Isolation contract

Modifies **no existing file**. Imports from `experiments/common/*` only, and
reuses `_run_epoch`, `build_loaders`, `seed_everything`, `build_head`,
`Sequential_CBAM`, `BuiltModel`, and `evaluate_run` unchanged. `run.py`,
`compare.py`, the 12 ablation configs/results, and `plan1_bgrand/` are
untouched. Comparison goes through `compare_arch.py`, not `compare.py`.

`evaluate_run` is reused **verbatim** because drop-path is parameter-free and
is identity in `eval()` — the state_dict loads into the plain builder, so the
network measured is exactly the network trained.

## Guards

`run_arch.py` refuses to run a config that:

- contains a `background_randomization` block (would bundle Plan 1 into a Plan 2 row),
- has a `stack` differing from the `efficientnetb0_on` baseline,
- sets more than one `architecture_mod` key.
