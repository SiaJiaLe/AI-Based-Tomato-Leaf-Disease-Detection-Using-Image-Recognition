# Plan 3 — Real-world confusion analysis

Goal: for every evaluated row, recover the **real-world confusion matrix** as machine-readable
numbers, so we can say *which diseases each solution confuses with which* — not just that its
macro-F1 was low.

Status: awaiting approval. Nothing has been run.

---

## 1. What already exists (and why it is not enough)

`experiments/common/evaluate.py:122` already writes `cm_real_world_test.png` into every
evaluated run directory. Those pictures are on the HPC right now.

But `_plot_confusion` (evaluate.py:67-85) computes `cm`, `return cm.tolist()` — and both
callers discard the return value. **The counts were never saved.** So today:

* one matrix at a time, by eye, as a PNG — fine;
* eighteen matrices compared, error modes ranked, a thesis table — impossible.

The classification report in `eval_results_real_world.json` cannot substitute. It gives
per-class precision/recall/F1/support, which yields the *diagonal* and the column totals — but
**not the off-diagonal**, which is exactly the question being asked ("misclassified as *what*").

So: one re-inference pass is genuinely required. There is no offline shortcut.

## 2. Does this violate the read-once rule?

No, and here is the argument in full, because it must survive an examiner.

The read-once rule exists so the real-world set cannot influence **model or hyperparameter
selection**. This pass:

* loads **frozen** checkpoints — nothing is trained, tuned, or selected;
* is **deterministic** inference — same weights, same transform, same order;
* reports a **richer summary of the same forward pass** whose aggregate is already published.

The predictions are *the same predictions*. No decision is conditioned on them.

**Enforced, not asserted:** for every row the script recomputes real-world macro-F1 and
**hard-fails** unless it matches the published `eval_results_real_world.json` to 1e-6. If a
matrix does not reproduce its own published number, the run aborts. This turns "trust me, the
preprocessing matched" into a check. It is also a real regression test on the repaired conda
env.

**The two val-only rows are EXCLUDED** — `droppath03` and `mixstyle_l12` were rejected on
validation and their real-world set was *never* read. Reading it now, retroactively, for
configs we discarded, is the actual violation. The script refuses them by default and prints
why. This is the one place where doing less is the rigorous choice.

## 3. Rows covered (18)

| Group | Runs |
|---|---|
| Ablation (12) | `{alexnet, vgg16, resnet34, resnet50, mobilenetv2, efficientnetb0}_{off,on}` |
| Plan 1 (2) | `efficientnetb0_on_bgrand` (synthetic), `efficientnetb0_on_bgrand_real` (real CC0) |
| Plan 2 (4) | `..._droppath02` (T1), `..._res240` (T2), `..._mixstyle_l123` (T3), `..._mixstyle_l123_bgrand` (combo) |

Excluded: `droppath03`, `mixstyle_l12` — see §2.

## 4. Files

**New only. No existing file is modified** — the isolation contract holds. `common/evaluate.py`
is imported, never touched.

| File | Purpose |
|---|---|
| `experiments/confusion_matrices.py` | NEW. Re-infers real-world per row, writes counts, renders figures + report. |
| `experiments/run_confusion_slurm.sh` | NEW. Short GPU job (30 min wall): preflight, then the script. |

Correctness detail that matters: **`res240` must be evaluated at 240px**, not 224. The script
dispatches on `architecture_mod.input_resolution` exactly as `run_arch.py:121-125` does —
`evaluate_run_res` for 240, the plain 224 path otherwise. Getting this wrong would silently
mismatch preprocessing (the CLAUDE.md hazard) and the §2 macro-F1 check would catch it anyway.

Tier 1/Tier 3 rows load through the **plain** builder: drop-path and MixStyle are identity at
eval and parameter-free, which is what `assert_eval_compatible` proved on the HPC (369
state_dict entries unchanged). No special-casing needed.

## 5. Outputs

Written to `experiments/results/confusion/`:

1. **`<run>_cm_real_world.json`** — raw counts + class names + the reproduced macro-F1.
   Machine-readable, permanent: **this pass never needs repeating.**
2. **`<run>.png`** — per-run row-normalized matrix (readable, one per row).
3. **`grid_efficientnetb0.png`** — the 6 EfficientNetB0 rows (baseline, bgrand, bgrand_real,
   T1, T2, T3, combo) as one multi-panel figure. This is the thesis figure: it shows what each
   intervention *did to the errors*, side by side.
4. **`confusion_report.{txt,md}`** — aligned text for the terminal, markdown for the report
   (same convention as `compile_results`, per your last feedback):
   * per run: **top-5 confusion pairs** — `true X -> predicted Y, N times (Z% of all X)`;
   * **error-profile table**: for each true class, its dominant wrong prediction in each run —
     answers "did this intervention change *which* mistake the model makes, or just how often";
   * **`Target_Spot` focus**: it is near-zero in *every* row. The matrix says where those
     images go instead. That is currently an unexplained failure in your results; this is the
     cheapest thing that could explain it.

Reuses `compile_results.py`'s rendering conventions: ASCII-only runtime output (a `Δ` or
em-dash crashes `print()` under a C locale and would take the job down), numeric columns
right-aligned, nothing hardcoded — every figure read from JSON.

## 6. Steps

1. Write `experiments/confusion_matrices.py` + the SLURM script (diffs shown for accept/reject).
2. Commit to `main`, push.
3. You: `git pull` on the HPC, `sbatch experiments/run_confusion_slurm.sh`.
4. Paste the log; I read the report and interpret the error modes.

Cost: inference only over the real-world set x 18 rows. No training. Expect minutes, not hours.

## 7. What this can and cannot deliver

**Can:** the error structure of every row; whether the interventions changed *what* the model
confuses or merely how much; a direct answer on `Target_Spot`; a genuine thesis figure.

**Cannot:** fix the single-seed problem. Confusion matrices are a *description* of the same
n=1 runs — a shifted confusion pair between two rows may be noise, exactly as `bgrand_real`'s
+0.0052 may be. Read them as **structure and hypotheses**, not as evidence of significance.
The 3-seed run remains the highest-value experiment; this is complementary, not a substitute.
