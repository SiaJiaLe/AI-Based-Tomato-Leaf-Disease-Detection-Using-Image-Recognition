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
| `efficientnetb0_on_droppath02/03` | Plan 2 Tier 1 — stochastic depth (negative) |
| `..._res240` | Plan 2 Tier 2 — resolution bump (negative) |
| `..._mixstyle_l12` / `_l123` | **Plan 2 Tier 3 — MixStyle** |
| `..._mixstyle_l1*_bgrand` | **Combination row — Tier 3 + Plan 1** (see below) |

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

## Tier 3 — MixStyle

**The one variable:** `mixstyle: {layers, p, alpha}`. A feature map's
channel-wise mean/std *are* its style; MixStyle normalizes each sample by its own
stats then re-applies stats interpolated with a **shuffled** sample's stats
(`lambda ~ Beta(0.1, 0.1)`). The label follows the content sample, so a class is
seen under a continuum of styles and style stops being a usable shortcut for it.
Random-shuffle variant — the paper's domain-label variant needs multiple source
domains and we have one (PlantVillage).

**Rationale (examiner-facing):** Tiers 1 and 2 changed capacity and detail; both
failed. MixStyle is the only tier that targets the lab->field *style* shift
directly, and it is designed for exactly this problem (Zhou et al., 2021).

**Honest prediction, recorded before the run:** single-source MixStyle can only
mix styles that exist *inside* PlantVillage, whose style variance is small
(uniform lighting, uniform backdrop). The synthesized styles are interpolations
within lab style and may never reach field style. A third bounded negative is a
live possibility — and would be evidence for the "the gap is not model-side"
conclusion, not a surprise.

**Why hooks, not wrapping:** wrapping each stage in an `nn.Sequential` (as
`Sequential_CBAM` does) would renumber state_dict keys (`blocks.1.x` ->
`blocks.1.0.x`) and force a duplicated eval path, as Tier 2's resolution change
did. Forward hooks leave the key layout byte-identical, so Tier 3 is scored by
the shared, unmodified `common.evaluate`. The mixers are *also* registered as
child modules so `train()`/`eval()` propagates to them — a mixer outside the
module tree would keep `training=True` forever and silently corrupt evaluation.
Being parameter-free, they add no state_dict entries; `assert_eval_compatible`
proves this at startup rather than trusting it.

## The combination row (Tier 3 + Plan 1)

plan2 §4 step 5: "test the combination as an explicit separate row — never assume
additivity." `..._mixstyle_l1*_bgrand` completes a **2x2 factorial**:

| row | mixstyle | bgrand |
|---|---|---|
| `efficientnetb0_on` | - | - |
| `efficientnetb0_on_bgrand` | - | yes |
| `..._mixstyle_l12` | yes | - |
| `..._mixstyle_l12_bgrand` | yes | yes |

With all four cells you can still attribute: combo vs mixstyle-alone isolates
bgrand's marginal effect *in MixStyle's presence*, and vice versa. The combo row
alone is **not** attributable to either factor — `compare_arch.py` stamps
`[COMBINATION — not attributable to either factor alone]` on its title so the
number is never quoted as Tier 3's effect.

These two factors are the one pairing with a mechanistic reason to compose:
bgrand perturbs style in **input space** (it swaps real pixels behind the leaf),
MixStyle perturbs it in **feature space** (it mixes internal statistics).

The combo's `background_randomization` block is **identical** to the
`efficientnetb0_on_bgrand` row's (same synthetic pool, prob, segmentation). If it
differed, the factorial would break and bgrand's marginal effect would be
uncomputable.

`combination: true` is a **required opt-in**: `run_arch.py` rejects any config
carrying a `background_randomization` block without it, so Plan 1 can never be
bundled into a Plan 2 row by accident — only deliberately, with the row named and
reported as a combination.

## Running it

```bash
sbatch experiments/plan2_arch/run_droppath_slurm.sh   # Tier 1
sbatch experiments/plan2_arch/run_res240_slurm.sh     # Tier 2
sbatch experiments/plan2_arch/run_mixstyle_slurm.sh   # Tier 3 + combination
```

`run_mixstyle_slurm.sh` runs both parts in one job: it sweeps the MixStyle depth
`{[1,2], [1,2,3]}` blind, picks the winner on **val**, reads the real-world set
once for it, then trains the combination row using that same val-chosen depth
plus bgrand. No selection ever touches the real-world set; it is read once per
row (twice total, for two distinct rows).

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

The combination row imports Plan 1's `BackgroundRandomize` transform and reuses
it verbatim (`data_res.py`), so the combo's input pipeline is the same code the
bgrand-alone row ran. `plan1_bgrand/` itself is read-only here.

`evaluate_run` is reused **verbatim** for Tier 1 and Tier 3 because drop-path and
MixStyle are both parameter-free and identity in `eval()` — the state_dict loads
into the plain builder, so the network measured is exactly the network trained.
Only Tier 2 needed its own eval path, because its resolution change is a
*preprocessing* change, not a parameter one; even there the scoring helpers
(`_load_model`, `_predict`, `_metrics`, `_plot_confusion`) are imported unchanged,
so every row is measured by one ruler.

## Guards

`run_arch.py` refuses to run a config that:

- carries a `background_randomization` block without `combination: true`
  (would bundle Plan 1 into a Plan 2 row silently),
- sets `combination: true` but combines nothing,
- has a `stack` differing from the `efficientnetb0_on` baseline,
- sets more than one `architecture_mod` key, or an unknown one.

`engine_arch.py` additionally proves at startup, before the optimizer is built,
that the row's modification left the state_dict untouched
(`assert_eval_compatible`) — so a wrong parameter-free assumption dies in seconds
rather than after an hour of training.
