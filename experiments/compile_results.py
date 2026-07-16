"""Compile every run's saved metrics into one report-ready table.

    python -m experiments.compile_results              # the standard report
    python -m experiments.compile_results --all        # every run found on disk
    python -m experiments.compile_results --no-write   # print only

READS ONLY. This script never computes a metric and never touches a test set. It
loads what the shared evaluator (`common.evaluate`) already wrote for each row:

    experiments/results/<run>/eval_results.json             (controlled test)
    experiments/results/<run>/eval_results_real_world.json  (real-world test)
    experiments/results/<run>/metrics.json                  (val, provenance)
    experiments/results/<run>/resolved_config.json          (what the row IS)

Every number therefore comes from the same measuring code for every row, which is
what makes the rows comparable at all. A missing file prints `n/a` — never a
guess, never a reconstruction from a remembered delta.

Outputs (unless --no-write):
    experiments/results/all_results.md    — paste into the report
    experiments/results/all_results.csv   — for Excel / plots

Isolation: additive. Does not modify compare.py, compare_arch.py,
compare_bgrand.py, or any result.
"""
import argparse
import csv
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")

BASELINE = "efficientnetb0_on"

# The 12-run OFF/ON ablation, weakest backbone first (compare.py's order).
BACKBONE_ORDER = ["alexnet", "vgg16", "resnet34", "resnet50", "mobilenetv2", "efficientnetb0"]

# The EfficientNetB0 modification story, in narrative order. Each entry is
# (run_name, label). Sweep members that were deliberately never evaluated are
# included so their absence is visible as evidence of the read-once rule.
STORY_ROWS = [
    ("efficientnetb0_on_bgrand", "Plan 1 - background randomization (synthetic)"),
    ("efficientnetb0_on_bgrand_real", "Plan 1 - background randomization (real CC0)"),
    ("efficientnetb0_on_droppath02", "Tier 1 - stochastic depth 0.2"),
    ("efficientnetb0_on_droppath03", "Tier 1 - stochastic depth 0.3"),
    ("efficientnetb0_on_res240", "Tier 2 - input resolution 240"),
    ("efficientnetb0_on_mixstyle_l12", "Tier 3 - MixStyle layers [1,2]"),
    ("efficientnetb0_on_mixstyle_l123", "Tier 3 - MixStyle layers [1,2,3]"),
    ("efficientnetb0_on_mixstyle_l123_bgrand", "COMBINATION - Tier 3 + Plan 1"),
]

METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]


def _load(run_name, fname):
    path = os.path.join(RESULTS_DIR, run_name, fname)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _fmt(x, sign=False):
    if x is None:
        return "n/a"
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def _status(run_name, controlled, metrics):
    """Say plainly what state a row is in, so a blank is never ambiguous."""
    if controlled is not None:
        return "evaluated"
    if metrics is not None:
        # Trained but never evaluated: a sweep member that lost on val. The
        # real-world set was deliberately not read for it. That is the rule
        # working, not missing data.
        return "val-only (not evaluated)"
    return "not run"


def collect(run_name):
    controlled = _load(run_name, "eval_results.json")
    real = _load(run_name, "eval_results_real_world.json")
    metrics = _load(run_name, "metrics.json")
    cfg = _load(run_name, "resolved_config.json")
    row = {
        "run": run_name,
        "status": _status(run_name, controlled, metrics),
        "val_macro_f1": metrics.get("best_val_macro_f1") if metrics else None,
        "combination": bool(cfg.get("combination")) if cfg else False,
    }
    for m in METRICS:
        row[f"controlled_{m}"] = controlled.get(m) if controlled else None
        row[f"realworld_{m}"] = real.get(m) if real else None
    row["gap_accuracy"] = real.get("generalization_gap_accuracy") if real else None
    row["gap_macro_f1"] = real.get("generalization_gap_macro_f1") if real else None
    return row


def _delta_vs_baseline(row, base):
    if row["realworld_macro_f1"] is None or base.get("realworld_macro_f1") is None:
        return None, None
    d_rw = row["realworld_macro_f1"] - base["realworld_macro_f1"]
    d_gap = (row["gap_macro_f1"] - base["gap_macro_f1"]
             if row["gap_macro_f1"] is not None and base["gap_macro_f1"] is not None else None)
    return d_rw, d_gap


def _table(rows, labels, base, title, note=None):
    out = [f"### {title}", ""]
    if note:
        out += [note, ""]
    out.append("| row | what it is | status | ctrl acc | ctrl macro-F1 | RW acc | RW macro-F1 | "
               "gap acc | gap macro-F1 | dRW macro-F1 | dgap macro-F1 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        d_rw, d_gap = _delta_vs_baseline(r, base)
        label = labels.get(r["run"], "")
        # Bold must wrap the code span, not sit inside it — `**x**` renders the
        # asterisks literally.
        name = f"**`{r['run']}`**" if r["run"] == BASELINE else f"`{r['run']}`"
        if r["run"] == BASELINE:
            label = label + " *(fixed baseline)*" if label else "*(fixed baseline)*"
        out.append(
            f"| {name} | {label} | {r['status']} | "
            f"{_fmt(r['controlled_accuracy'])} | {_fmt(r['controlled_macro_f1'])} | "
            f"{_fmt(r['realworld_accuracy'])} | {_fmt(r['realworld_macro_f1'])} | "
            f"{_fmt(r['gap_accuracy'])} | {_fmt(r['gap_macro_f1'])} | "
            f"{_fmt(d_rw, sign=True)} | {_fmt(d_gap, sign=True)} |")
    out.append("")
    return out


def _full_metrics_table(rows, labels):
    """Precision/recall/weighted-F1 too — the full metric set per row."""
    out = ["### Full metric set (controlled | real-world)", "",
           "| row | ctrl P | ctrl R | ctrl F1 | ctrl wF1 | RW P | RW R | RW F1 | RW wF1 | val F1 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] == "not run":
            continue
        out.append(
            f"| `{r['run']}` | {_fmt(r['controlled_macro_precision'])} | "
            f"{_fmt(r['controlled_macro_recall'])} | {_fmt(r['controlled_macro_f1'])} | "
            f"{_fmt(r['controlled_weighted_f1'])} | {_fmt(r['realworld_macro_precision'])} | "
            f"{_fmt(r['realworld_macro_recall'])} | {_fmt(r['realworld_macro_f1'])} | "
            f"{_fmt(r['realworld_weighted_f1'])} | {_fmt(r['val_macro_f1'])} |")
    out.append("")
    return out


def _per_class_table(run_names):
    """Per-class REAL-WORLD F1, classes x runs."""
    reports = {}
    for run in run_names:
        real = _load(run, "eval_results_real_world.json")
        if real:
            reports[run] = real["classification_report"]
    if not reports:
        return ["### Per-class real-world F1", "", "_No evaluated runs found._", ""]

    classes = [c for c in next(iter(reports.values()))
               if c not in ("accuracy", "macro avg", "weighted avg")]
    shown = list(reports)
    out = ["### Per-class real-world F1 (evaluated runs only)", "",
           "| class | " + " | ".join(r.replace("efficientnetb0", "eb0") for r in shown) + " |",
           "|---" * (len(shown) + 1) + "|"]
    for c in classes:
        cells = [_fmt(reports[r][c]["f1-score"]) if c in reports[r] else "n/a" for r in shown]
        # A class that is 0.0000 in every single column is a data problem, not a
        # modelling one — say so where it cannot be missed.
        vals = [reports[r][c]["f1-score"] for r in shown if c in reports[r]]
        flag = "  **<- zero in every run**" if vals and all(v == 0.0 for v in vals) else ""
        out.append(f"| {c}{flag} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Include every run directory found, not just the known rows.")
    parser.add_argument("--no-write", action="store_true", help="Print only; write no files.")
    args = parser.parse_args()

    ablation_names = [f"{bb}_{t}" for bb in BACKBONE_ORDER for t in ("off", "on")]
    story_names = [r for r, _ in STORY_ROWS]
    known = ablation_names + story_names

    if args.all:
        found = sorted(d for d in os.listdir(RESULTS_DIR)
                       if os.path.isdir(os.path.join(RESULTS_DIR, d))) if os.path.isdir(RESULTS_DIR) else []
        known += [d for d in found if d not in known]

    labels = {f"{bb}_off": "Stack OFF" for bb in BACKBONE_ORDER}
    labels.update({f"{bb}_on": "Stack ON" for bb in BACKBONE_ORDER})
    labels.update(dict(STORY_ROWS))

    rows = {name: collect(name) for name in known}
    base = rows.get(BASELINE) or {"realworld_macro_f1": None, "gap_macro_f1": None}

    doc = ["# All results", "",
           "Every number is read from the row's own `eval_results*.json`, written by the "
           "single shared evaluator (`experiments/common/evaluate.py`). Same measuring code "
           "for every row - that is what makes them comparable. `n/a` means the file does "
           "not exist; nothing here is reconstructed or inferred.", "",
           f"Deltas are versus the fixed baseline `{BASELINE}`. Judge rows on **real-world "
           "macro-F1**; the gap is a supporting number, not the target (plan2 sec.5).", ""]

    doc += _table([rows[n] for n in ablation_names], labels, base,
                  "1. Backbone ablation - solution stack OFF vs ON (12 rows)",
                  "Each backbone trained twice: without and with the solution stack "
                  "(advanced augmentation, label smoothing, strong head, CBAM, two-group "
                  "Stage-B unfreeze). EfficientNetB0 ON is the fixed baseline for everything below.")

    doc += _table([rows[n] for n in story_names], labels, base,
                  "2. EfficientNetB0 modifications - Plan 1 and Plan 2",
                  "Each row is the `efficientnetb0_on` baseline plus ONE change (same split, "
                  "seed 42, same budget), except the final COMBINATION row, which carries two "
                  "and is **not attributable to either factor alone**. Sweep members shown as "
                  "`val-only (not evaluated)` were trained blind and lost on validation, so the "
                  "real-world set was never read for them - that is the read-once rule working.")

    # The baseline leads both detail tables: a per-class number is only readable
    # against the row it is supposed to beat.
    detail_rows = [BASELINE] + [n for n in story_names if n != BASELINE]
    doc += _full_metrics_table([rows[n] for n in detail_rows], labels)
    doc += _per_class_table([n for n in detail_rows if rows[n]["status"] == "evaluated"])

    # Reading notes that the numbers alone do not carry.
    combo = next((rows[n] for n in story_names if rows[n]["combination"]
                  and rows[n]["status"] == "evaluated"), None)
    doc += ["### How to read this", "",
            "- **Judge on real-world macro-F1.** Controlled accuracy is a sanity check, not "
            "the target. A row that raises PlantVillage accuracy but not real-world has not "
            "helped the problem (plan2 sec.7).",
            "- **A narrowing gap is not automatically good.** The gap is a *difference*: it "
            "shrinks if the controlled side falls faster than the real-world side. Always "
            "read it next to the real-world column."]
    if combo is not None and combo["gap_macro_f1"] is not None:
        d_rw, d_gap = _delta_vs_baseline(combo, base)
        if d_gap is not None and d_gap < 0 and d_rw is not None and d_rw < 0:
            doc.append(f"- **`{combo['run']}` is exactly that case:** gap "
                       f"{_fmt(d_gap, sign=True)} (the only narrowing in the study) while "
                       f"real-world macro-F1 is {_fmt(d_rw, sign=True)}. The gap closed "
                       f"because the controlled score collapsed, not because field "
                       f"performance improved. It is not a win.")
    doc += ["- **Sweep losers are listed, not hidden.** Their missing test numbers are "
            "evidence that hyperparameters were selected on validation only.", ""]

    text = "\n".join(doc)
    print(text)

    if args.no_write:
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, "all_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    csv_path = os.path.join(RESULTS_DIR, "all_results.csv")
    fields = (["run", "label", "status", "val_macro_f1", "combination"]
              + [f"controlled_{m}" for m in METRICS]
              + [f"realworld_{m}" for m in METRICS]
              + ["gap_accuracy", "gap_macro_f1", "delta_realworld_macro_f1", "delta_gap_macro_f1"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name in known:
            r = dict(rows[name])
            d_rw, d_gap = _delta_vs_baseline(r, base)
            r["label"] = labels.get(name, "")
            r["delta_realworld_macro_f1"] = d_rw
            r["delta_gap_macro_f1"] = d_gap
            w.writerow({k: r.get(k) for k in fields})

    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
