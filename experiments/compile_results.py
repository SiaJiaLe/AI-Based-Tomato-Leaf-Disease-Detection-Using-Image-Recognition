"""Compile every run's saved metrics into one report-ready table.

    python -m experiments.compile_results              # aligned text (terminal)
    python -m experiments.compile_results --markdown   # markdown to stdout
    python -m experiments.compile_results --all        # every run found on disk
    python -m experiments.compile_results --no-write   # print only, write no files

Tables are built as data and rendered twice: aligned fixed-width text for reading
in a terminal, and markdown for pasting into the report. `all_results.md` is
always markdown; stdout defaults to text because raw markdown pipes are unreadable
on a console.

READS ONLY. This script never computes a metric and never touches a test set. It
loads what the shared evaluator (`common.evaluate`) already wrote for each row:

    experiments/results/<run>/eval_results.json             (controlled test)
    experiments/results/<run>/eval_results_real_world.json  (real-world test)
    experiments/results/<run>/metrics.json                  (val, provenance)
    experiments/results/<run>/resolved_config.json          (what the row IS)

Every number therefore comes from the same measuring code for every row, which is
what makes the rows comparable at all. A missing file prints `n/a` - never a
guess, never a reconstruction from a remembered delta.

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
    ("efficientnetb0_on_bgrand", "Plan 1 - bg randomization (synthetic)"),
    ("efficientnetb0_on_bgrand_real", "Plan 1 - bg randomization (real CC0)"),
    ("efficientnetb0_on_droppath02", "Tier 1 - stochastic depth 0.2"),
    ("efficientnetb0_on_droppath03", "Tier 1 - stochastic depth 0.3"),
    ("efficientnetb0_on_res240", "Tier 2 - input resolution 240"),
    ("efficientnetb0_on_mixstyle_l12", "Tier 3 - MixStyle layers [1,2]"),
    ("efficientnetb0_on_mixstyle_l123", "Tier 3 - MixStyle layers [1,2,3]"),
    ("efficientnetb0_on_mixstyle_l123_bgrand", "COMBINATION - Tier 3 + Plan 1"),
]

METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]


class Table:
    """A table held as data so it can be rendered as text OR markdown."""

    def __init__(self, title, headers, rows, note=None, mark=None):
        self.title = title
        self.headers = headers
        self.rows = rows            # list[list[str]]
        self.note = note
        self.mark = mark or set()   # indices of rows to emphasise (markdown only)


def _is_number(cell):
    """True for a metric cell: a float, or the 'n/a' that stands in for one."""
    s = str(cell).strip()
    if s == "n/a":
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


def _wrap(text, width):
    """Break a paragraph onto lines <= width, for the text renderer."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines


def render_text(tables, preamble):
    out = []
    for para in preamble:
        out += _wrap(para, 100) + [""]
    for t in tables:
        out += [t.title, "=" * len(t.title), ""]
        if t.note:
            out += _wrap(t.note, 100) + [""]
        cols = len(t.headers)
        widths = [max(len(str(t.headers[i])),
                      max((len(str(r[i])) for r in t.rows), default=0)) for i in range(cols)]
        # Right-align numeric columns so decimal points line up and a column can
        # be scanned; left-align text columns, which read badly ragged-left.
        numeric = [all(_is_number(r[i]) for r in t.rows) and bool(t.rows) for i in range(cols)]

        def line(cells):
            parts = [str(cells[i]).rjust(widths[i]) if numeric[i]
                     else str(cells[i]).ljust(widths[i]) for i in range(cols)]
            return "  ".join(parts).rstrip()
        out.append(line(t.headers))
        out.append("-" * len(line(t.headers)))
        out += [line(r) for r in t.rows]
        out.append("")
    return "\n".join(out)


def render_markdown(tables, preamble):
    out = ["# All results", ""]
    for para in preamble:
        out += [para, ""]
    for t in tables:
        out += [f"### {t.title}", ""]
        if t.note:
            out += [t.note, ""]
        out.append("| " + " | ".join(t.headers) + " |")
        out.append("|" + "---|" * len(t.headers))
        for i, r in enumerate(t.rows):
            cells = list(r)
            # Bold must wrap the code span, not sit inside it: `**x**` renders the
            # asterisks literally.
            cells[0] = f"**`{cells[0]}`**" if i in t.mark else f"`{cells[0]}`"
            out.append("| " + " | ".join(str(c) for c in cells) + " |")
        out.append("")
    return "\n".join(out)


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


def _status(controlled, metrics):
    """Say plainly what state a row is in, so a blank is never ambiguous."""
    if controlled is not None:
        return "evaluated"
    if metrics is not None:
        # Trained but never evaluated: a sweep member that lost on val. The
        # real-world set was deliberately not read for it. That is the rule
        # working, not missing data.
        return "val-only"
    return "not run"


def collect(run_name):
    controlled = _load(run_name, "eval_results.json")
    real = _load(run_name, "eval_results_real_world.json")
    metrics = _load(run_name, "metrics.json")
    cfg = _load(run_name, "resolved_config.json")
    row = {
        "run": run_name,
        "status": _status(controlled, metrics),
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


def main_table(rows, labels, base, title, note):
    headers = ["row", "what it is", "status", "ctrl acc", "ctrl F1",
               "RW acc", "RW F1", "gap acc", "gap F1", "dRW F1", "dgap F1"]
    body, mark = [], set()
    for i, r in enumerate(rows):
        d_rw, d_gap = _delta_vs_baseline(r, base)
        label = labels.get(r["run"], "")
        if r["run"] == BASELINE:
            mark.add(i)
            label = (label + " (fixed baseline)").strip()
        body.append([r["run"], label, r["status"],
                     _fmt(r["controlled_accuracy"]), _fmt(r["controlled_macro_f1"]),
                     _fmt(r["realworld_accuracy"]), _fmt(r["realworld_macro_f1"]),
                     _fmt(r["gap_accuracy"]), _fmt(r["gap_macro_f1"]),
                     _fmt(d_rw, sign=True), _fmt(d_gap, sign=True)])
    return Table(title, headers, body, note, mark)


def full_metrics_table(rows):
    headers = ["row", "ctrl P", "ctrl R", "ctrl F1", "ctrl wF1",
               "RW P", "RW R", "RW F1", "RW wF1", "val F1"]
    body = []
    for r in rows:
        if r["status"] == "not run":
            continue
        body.append([r["run"],
                     _fmt(r["controlled_macro_precision"]), _fmt(r["controlled_macro_recall"]),
                     _fmt(r["controlled_macro_f1"]), _fmt(r["controlled_weighted_f1"]),
                     _fmt(r["realworld_macro_precision"]), _fmt(r["realworld_macro_recall"]),
                     _fmt(r["realworld_macro_f1"]), _fmt(r["realworld_weighted_f1"]),
                     _fmt(r["val_macro_f1"])])
    return Table("Full metric set (P = macro precision, R = macro recall, wF1 = weighted F1)",
                 headers, body,
                 "Macro F1 weights every class equally, so the near-dead classes drag it down; "
                 "weighted F1 does not. The distance between RW F1 and RW wF1 is the cost of "
                 "the classes the model cannot transfer.")


def per_class_table(run_names):
    reports = {}
    for run in run_names:
        real = _load(run, "eval_results_real_world.json")
        if real:
            reports[run] = real["classification_report"]
    if not reports:
        return Table("Per-class real-world F1", ["class"], [],
                     "No evaluated runs found.")

    classes = [c for c in next(iter(reports.values()))
               if c not in ("accuracy", "macro avg", "weighted avg")]
    shown = list(reports)
    headers = ["class"] + [r.replace("efficientnetb0_on", "on").replace("efficientnetb0", "eb0")
                           for r in shown]
    body = []
    for c in classes:
        vals = [reports[r][c]["f1-score"] if c in reports[r] else None for r in shown]
        present = [v for v in vals if v is not None]
        # A class scoring ~0 in every column is a data problem, not a modelling
        # one. Flag it where it cannot be missed.
        flag = "  <- near-zero everywhere" if present and max(present) < 0.05 else ""
        body.append([c.replace("Tomato___", "") + flag] + [_fmt(v) for v in vals])
    return Table("Per-class real-world F1 (evaluated runs only)", headers, body,
                 "Baseline first. A class near zero in every column is a data failure, not a "
                 "modelling one, and it drags every macro-F1 in this report down.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Include every run directory found, not just the known rows.")
    parser.add_argument("--markdown", action="store_true",
                        help="Print markdown to stdout instead of aligned text.")
    parser.add_argument("--no-write", action="store_true", help="Print only; write no files.")
    args = parser.parse_args()

    ablation_names = [f"{bb}_{t}" for bb in BACKBONE_ORDER for t in ("off", "on")]
    story_names = [r for r, _ in STORY_ROWS]
    known = ablation_names + story_names

    if args.all and os.path.isdir(RESULTS_DIR):
        found = sorted(d for d in os.listdir(RESULTS_DIR)
                       if os.path.isdir(os.path.join(RESULTS_DIR, d)))
        known += [d for d in found if d not in known]

    labels = {f"{bb}_off": "Stack OFF" for bb in BACKBONE_ORDER}
    labels.update({f"{bb}_on": "Stack ON" for bb in BACKBONE_ORDER})
    labels.update(dict(STORY_ROWS))

    rows = {name: collect(name) for name in known}
    base = rows.get(BASELINE) or {"realworld_macro_f1": None, "gap_macro_f1": None}

    preamble = [
        "Every number is read from the row's own eval_results*.json, written by the single "
        "shared evaluator (experiments/common/evaluate.py). Same measuring code for every row "
        "- that is what makes them comparable. 'n/a' means the file does not exist; nothing "
        "here is reconstructed or inferred.",
        f"Deltas (dRW F1, dgap F1) are versus the fixed baseline {BASELINE}. Judge rows on "
        "REAL-WORLD macro-F1; the gap is a supporting number, not the target (plan2 sec.5).",
    ]

    tables = [
        main_table([rows[n] for n in ablation_names], labels, base,
                   "1. Backbone ablation - solution stack OFF vs ON (12 rows)",
                   "Each backbone trained twice: without and with the solution stack (advanced "
                   "augmentation, label smoothing, strong head, CBAM, two-group Stage-B "
                   "unfreeze). EfficientNetB0 ON is the fixed baseline for everything below."),
        main_table([rows[n] for n in story_names], labels, base,
                   "2. EfficientNetB0 modifications - Plan 1 and Plan 2",
                   "Each row is the baseline plus ONE change (same split, seed 42, same budget), "
                   "except the final COMBINATION row, which carries two and is NOT attributable "
                   "to either factor alone. Rows marked 'val-only' were trained blind and lost "
                   "on validation, so the real-world set was never read for them - that is the "
                   "read-once rule working, not missing data."),
    ]

    # The baseline leads both detail tables: a per-class number is only readable
    # against the row it is supposed to beat.
    detail = [BASELINE] + [n for n in story_names if n != BASELINE]
    tables.append(full_metrics_table([rows[n] for n in detail]))
    tables.append(per_class_table([n for n in detail if rows[n]["status"] == "evaluated"]))

    notes = ["How to read this", "=" * len("How to read this"), "",
             "- Judge on REAL-WORLD macro-F1. Controlled accuracy is a sanity check, not the",
             "  target. A row that raises PlantVillage accuracy but not real-world has not",
             "  helped the problem (plan2 sec.7).",
             "- A narrowing gap is not automatically good. The gap is a DIFFERENCE: it shrinks",
             "  if the controlled side falls faster than the real-world side. Always read it",
             "  next to the real-world column."]

    # Derived from the data, not hardcoded, so these stay honest if numbers change.
    for n in story_names:
        r = rows[n]
        if r["status"] != "evaluated":
            continue
        d_rw, d_gap = _delta_vs_baseline(r, base)
        if d_rw is None or d_gap is None:
            continue
        if d_gap < 0 and d_rw < 0:
            notes.append(f"- {n} is exactly that trap: gap {d_gap:+.4f} (a narrowing) but "
                         f"real-world macro-F1 {d_rw:+.4f}. The gap closed because the "
                         f"controlled score fell faster, not because field performance "
                         f"improved. NOT a win.")
        elif d_rw > 0:
            notes.append(f"- {n} is the only shape that counts as progress: real-world "
                         f"macro-F1 {d_rw:+.4f} AND gap {d_gap:+.4f}. Small and single-seed, "
                         f"so report it as 'no harm, possibly a small gain' - but compare it "
                         f"against its own synthetic twin, not just the baseline.")
    notes += ["- Sweep losers are listed, not hidden. Their missing test numbers are evidence",
              "  that hyperparameters were selected on validation only.", ""]

    text = render_text(tables, preamble) + "\n" + "\n".join(notes)
    md = render_markdown(tables, preamble) + "\n" + "\n".join(
        [f"### {notes[0]}", ""] + [n for n in notes[3:]])

    print(md if args.markdown else text)

    if args.no_write:
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, "all_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    txt_path = os.path.join(RESULTS_DIR, "all_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
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

    print(f"\nWrote {md_path}  (markdown - paste into the report)")
    print(f"Wrote {txt_path}  (aligned text)")
    print(f"Wrote {csv_path}  (for Excel / plots)")


if __name__ == "__main__":
    main()
