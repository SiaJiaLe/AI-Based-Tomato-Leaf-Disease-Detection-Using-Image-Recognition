"""Aggregate all runs into the CP2 deliverable tables and the gap figure.

Reads every experiments/results/<run>/eval_results.json (+ real-world) and
produces:
  1. Master comparison table — all 12 runs, both datasets, gap column.
  2. Ablation table — the six OFF/ON pairs, Δ(real-world acc) and
     Δ(real-world macro-F1) attributable to the solution stack per backbone.
  3. Gap-narrowing figure — grouped bar chart of real-world macro-F1, OFF vs
     ON, for all six architectures.

Run naming convention: <backbone>_<off|on>.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results")

BACKBONE_ORDER = ["resnet34", "resnet50", "vgg16", "alexnet", "mobilenetv2", "efficientnetb0"]
DISPLAY = {"resnet34": "ResNet34", "resnet50": "ResNet50", "vgg16": "VGG16",
           "alexnet": "AlexNet", "mobilenetv2": "MobileNetV2", "efficientnetb0": "EfficientNetB0"}


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _run(run_name):
    d = os.path.join(RESULTS_DIR, run_name)
    return _load(os.path.join(d, "eval_results.json")), _load(os.path.join(d, "eval_results_real_world.json"))


def master_table():
    header = f"{'Run':<20} {'Acc':>8} {'MacroF1':>8}  {'RW_Acc':>8} {'RW_F1':>8}  {'GapAcc':>8}"
    lines = [header, "-" * len(header)]
    rows = []
    for bb in BACKBONE_ORDER:
        for treat in ("off", "on"):
            name = f"{bb}_{treat}"
            controlled, real = _run(name)
            if controlled is None:
                lines.append(f"{name:<20} {'-':>8} {'-':>8}  {'-':>8} {'-':>8}  {'-':>8}  (missing)")
                continue
            acc, f1 = controlled["accuracy"], controlled["macro_f1"]
            rw_acc = real["accuracy"] if real else None
            rw_f1 = real["macro_f1"] if real else None
            gap = (acc - rw_acc) if rw_acc is not None else None
            lines.append(f"{name:<20} {acc:>8.4f} {f1:>8.4f}  "
                         f"{_f(rw_acc):>8} {_f(rw_f1):>8}  {_f(gap, sign=True):>8}")
            rows.append({"run": name, "backbone": bb, "treatment": treat,
                         "accuracy": acc, "macro_f1": f1,
                         "real_world_accuracy": rw_acc, "real_world_macro_f1": rw_f1,
                         "generalization_gap_accuracy": gap})
    return "\n".join(lines), rows


def _f(x, sign=False):
    if x is None:
        return "-"
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def ablation_table():
    header = f"{'Architecture':<16} {'RW_F1_OFF':>10} {'RW_F1_ON':>10} {'dF1':>9}  {'RW_Acc_OFF':>11} {'RW_Acc_ON':>10} {'dAcc':>9}"
    lines = [header, "-" * len(header)]
    pairs = []
    for bb in BACKBONE_ORDER:
        _, off = _run(f"{bb}_off")
        _, on = _run(f"{bb}_on")
        if not off or not on:
            lines.append(f"{DISPLAY[bb]:<16} {'(incomplete pair — need both OFF and ON real-world results)':<}")
            continue
        d_f1 = on["macro_f1"] - off["macro_f1"]
        d_acc = on["accuracy"] - off["accuracy"]
        lines.append(f"{DISPLAY[bb]:<16} {off['macro_f1']:>10.4f} {on['macro_f1']:>10.4f} {d_f1:>+9.4f}  "
                     f"{off['accuracy']:>11.4f} {on['accuracy']:>10.4f} {d_acc:>+9.4f}")
        pairs.append({"backbone": bb, "rw_f1_off": off["macro_f1"], "rw_f1_on": on["macro_f1"],
                      "delta_rw_f1": d_f1, "rw_acc_off": off["accuracy"],
                      "rw_acc_on": on["accuracy"], "delta_rw_acc": d_acc})
    lines.append("")
    lines.append("dF1/dAcc = ON minus OFF on the real-world test set (positive = the stack helped).")
    return "\n".join(lines), pairs


def gap_figure(pairs, out_path):
    if not pairs:
        return
    labels = [DISPLAY[p["backbone"]] for p in pairs]
    off = [p["rw_f1_off"] for p in pairs]
    on = [p["rw_f1_on"] for p in pairs]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, off, w, label="Stack OFF")
    ax.bar(x + w / 2, on, w, label="Stack ON")
    ax.set_ylabel("Real-world macro-F1")
    ax.set_title("Real-world generalization: Stack OFF vs ON per architecture")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.legend(); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    master, rows = master_table()
    ablation, pairs = ablation_table()
    print("\n=== MASTER COMPARISON (all 12 runs) ===\n" + master)
    print("\n=== ABLATION (OFF vs ON per architecture) ===\n" + ablation)

    fig_path = os.path.join(RESULTS_DIR, "gap_narrowing.png")
    gap_figure(pairs, fig_path)

    with open(os.path.join(RESULTS_DIR, "comparison.json"), "w") as f:
        json.dump({"master": rows, "ablation": pairs}, f, indent=2)
    print(f"\nSaved comparison.json and gap_narrowing.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
