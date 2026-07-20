#!/bin/bash
# SLURM-free, RESUME-SAFE runner for the full 8-class study on ONE GPU
# (Kaggle Notebook, Colab, or any rented single-GPU box). This is retrain_all.sh
# with the cluster-only parts removed:
#
#   * no `conda activate` (Kaggle uses the base env; deps are pip-installed in a
#     notebook cell first - see experiments/KAGGLE_SETUP.md),
#   * no `sbatch` / no job arrays / no `afterok` - every run goes sequentially in
#     THIS process,
#   * it does NOT archive-and-wipe experiments/results/. The opposite of
#     retrain_all.sh: on a 12-hour-capped platform we KEEP prior results so a
#     re-run SKIPS whatever already finished and continues where it stopped,
#   * it does NOT `mv` the read-only real-world eval set (Kaggle mounts data
#     read-only). The setup notebook already built an 8-class view of it.
#
# Target_Spot and Tomato_mosaic_virus are EXCLUDED from training here (same as the
# HPC study): the splitter omits them from train/val/test, and the notebook's
# 8-class real-world view omits them from evaluation.
#
# Usage (run from the repo root, after the notebook has wired the data):
#     bash experiments/run_all_local.sh              # train + evaluate + post-process
#     bash experiments/run_all_local.sh --dry-run    # print the plan + skip/keep, run nothing
#
# Re-run it as many times as it takes: finished runs are skipped, so across
# several Kaggle sessions the study fills in until every row exists.

set -uo pipefail          # NOT -e: one failing run must not kill the whole batch
cd "$(dirname "$0")/.."
REPO="$(pwd)"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown arg: $arg (accepted: --dry-run)" >&2; exit 2 ;;
  esac
done

# Classes turned OFF (not deleted): skipped in the split AND absent from the
# 8-class real-world view the notebook builds. To turn them back on, empty this
# list and rebuild the split + real-world view.
EXCLUDE=("Tomato___Target_Spot" "Tomato___Tomato_mosaic_virus")

PY="${PYTHON:-python}"
# RESULTS is where run.py/run_bgrand/run_arch write (repo-root experiments/results).
# The override exists ONLY so the offline dry-run test can point the skip-check at a
# fake tree; a real run must leave it at the default so the markers match Python.
RESULTS="${RUN_ALL_RESULTS:-experiments/results}"
CFG=experiments/configs
BGCFG=experiments/plan1_bgrand/configs
SRCFG=experiments/plan1_bgrand/configs/seedrep
P2CFG=experiments/plan2_arch/configs

FAILED=()

# --- helpers --------------------------------------------------------------

# run_name of a config, read straight from the file (filename != run_name for the
# seedrep configs). Pure grep/sed so it needs no yaml library.
cfg_run_name() {
  grep -E '^run_name:' "$1" 2>/dev/null | head -1 | sed 's/^run_name:[[:space:]]*//' | tr -d '\r'
}

# run_step <marker-or-empty> <label> <command...>
#   marker set + present  -> skip (already done)
#   dry-run               -> print intent, run nothing
#   else                  -> run; a failure is logged and the batch continues
run_step() {
  local marker="$1"; shift
  local label="$1"; shift
  if [ -n "$marker" ] && [ -f "$marker" ]; then
    echo "[skip] $label"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[WOULD RUN] $label"
    echo "            $*"
    return 0
  fi
  echo ""; echo "=== $label ==="
  if ! "$@"; then
    echo "!! $label FAILED - continuing with the rest." >&2
    FAILED+=("$label")
    return 1
  fi
  return 0
}

have_rembg() { $PY -c "import rembg" >/dev/null 2>&1; }

# --- [0] preflight (non-fatal) -------------------------------------------
echo "=== [0/9] Preflight ==="
mkdir -p "$RESULTS"
if [ "$DRY_RUN" -eq 0 ]; then
  # Import/version/CUDA check. Non-fatal on purpose: a version-drift WARNING (e.g.
  # a slightly different torch than the L4 pins) must not stop a Kaggle run, and a
  # transient hiccup shouldn't waste the session. A truly broken env will fail loudly
  # on the first real run anyway.
  $PY -m experiments.preflight_env || echo "WARNING: preflight reported issues (see above); continuing."
else
  echo "[dry-run] skipping preflight"
fi

# --- [1] split (Target_Spot + mosaic excluded) ---------------------------
echo ""; echo "=== [1/9] Split data/raw -> data/processed (8 classes) ==="
run_step "" "split (exclude: ${EXCLUDE[*]}; skip if already 8-class)" \
  $PY -m experiments.split_dataset --exclude "${EXCLUDE[@]}" --skip-if-exists

# --- [2] ablation: 12 backbones OFF/ON -----------------------------------
echo ""; echo "=== [2/9] Ablation (12 runs) ==="
ABLATION=(
  alexnet_off alexnet_on
  vgg16_off vgg16_on
  resnet34_off resnet34_on
  resnet50_off resnet50_on
  mobilenetv2_off mobilenetv2_on
  efficientnetb0_off efficientnetb0_on
)
for name in "${ABLATION[@]}"; do
  rn="$(cfg_run_name "$CFG/$name.yaml")"; rn="${rn:-$name}"
  run_step "$RESULTS/$rn/eval_results_real_world.json" "ablation: $rn" \
    $PY -m experiments.run --config "$CFG/$name.yaml"
done

# --- rembg gate for the background-randomization blocks -------------------
# bgrand / bgrand_real / seedrep / the mixstyle combo all use segmentation:
# pretrained (U^2-Net). If rembg isn't importable, SKIP those blocks with the pip
# line rather than failing the whole run - the ablation + Plan 2 T1/T2 above and
# their post-processing still complete.
REMBG_OK=0
if [ "$DRY_RUN" -eq 1 ] || have_rembg; then
  REMBG_OK=1
else
  echo ""
  echo "WARNING: rembg is not importable, so the background-randomization blocks"
  echo "         (bgrand, bgrand_real, seedrep, mixstyle combo) will be SKIPPED."
  echo "         Install it, then re-run this script to fill them in:"
  echo "             pip install -r experiments/plan1_bgrand/requirements.txt"
  echo "             $PY -c \"from rembg import new_session; new_session('u2net')\""
fi

# Synthetic backgrounds are generated on demand (idempotent) if the folder is empty.
ensure_synthetic_bg() {
  [ "$DRY_RUN" -eq 1 ] && return 0
  if [ -z "$(ls -A data/backgrounds_generic_synthetic 2>/dev/null | grep -Ei '\.(jpg|jpeg|png|bmp|tif|tiff|webp)$' || true)" ]; then
    echo "No synthetic backgrounds - generating..."
    $PY -m experiments.plan1_bgrand.make_synthetic_backgrounds || echo "WARNING: synthetic bg generation failed."
  fi
}

if [ "$REMBG_OK" -eq 1 ]; then
  ensure_synthetic_bg

  # --- [3] Plan 1: bgrand (synthetic backgrounds) ------------------------
  echo ""; echo "=== [3/9] Plan 1 bgrand (synthetic) ==="
  run_step "$RESULTS/efficientnetb0_on_bgrand/eval_results_real_world.json" "bgrand: efficientnetb0_on_bgrand" \
    $PY -m experiments.plan1_bgrand.run_bgrand --config "$BGCFG/efficientnetb0_on_bgrand.yaml"
  run_step "" "compare bgrand vs baseline" \
    $PY -m experiments.plan1_bgrand.compare_bgrand

  # --- [4] Plan 1: bgrand_real (real CC0 backgrounds) --------------------
  echo ""; echo "=== [4/9] Plan 1 bgrand_real (real backgrounds) ==="
  run_step "$RESULTS/efficientnetb0_on_bgrand_real/eval_results_real_world.json" "bgrand_real: efficientnetb0_on_bgrand_real" \
    $PY -m experiments.plan1_bgrand.run_bgrand --config "$BGCFG/efficientnetb0_on_bgrand_real.yaml"
  run_step "" "compare bgrand_real vs baseline" \
    $PY -m experiments.plan1_bgrand.compare_bgrand --bgrand efficientnetb0_on_bgrand_real

  # --- [5] seedrep: 3-seed replication (7 new runs) ----------------------
  echo ""; echo "=== [5/9] Seed replication (7 runs) ==="
  SEEDREP=(base_s42 base_s43 base_s44 bgrand_s43 bgrand_s44 bgrandreal_s43 bgrandreal_s44)
  for name in "${SEEDREP[@]}"; do
    rn="$(cfg_run_name "$SRCFG/$name.yaml")"; rn="${rn:-$name}"
    run_step "$RESULTS/$rn/eval_results_real_world.json" "seedrep: $rn" \
      $PY -m experiments.plan1_bgrand.run_bgrand --config "$SRCFG/$name.yaml"
  done
else
  echo ""; echo "=== [3-5/9] bgrand / bgrand_real / seedrep SKIPPED (no rembg) ==="
fi

# --- [6] Plan 2 Tier 1: drop-path (sweep -> val winner -> eval once) -----
echo ""; echo "=== [6/9] Plan 2 Tier 1: drop-path ==="
for rate in 02 03; do
  rn="$(cfg_run_name "$P2CFG/efficientnetb0_on_droppath${rate}.yaml")"; rn="${rn:-efficientnetb0_on_droppath${rate}}"
  run_step "$RESULTS/$rn/best_model.pth" "plan2-t1 train (train-only): $rn" \
    $PY -m experiments.plan2_arch.run_arch --config "$P2CFG/efficientnetb0_on_droppath${rate}.yaml" --train-only
done
if [ "$DRY_RUN" -eq 1 ]; then
  echo "[WOULD RUN] plan2-t1 select_on_val {droppath02,droppath03} -> eval winner (real-world read once) -> compare_arch"
else
  $PY -m experiments.plan2_arch.select_on_val efficientnetb0_on_droppath02 efficientnetb0_on_droppath03 || true
  T1_WINNER=$($PY -m experiments.plan2_arch.select_on_val efficientnetb0_on_droppath02 efficientnetb0_on_droppath03 --print-winner 2>/dev/null || true)
  if [ -n "${T1_WINNER:-}" ]; then
    echo "Tier 1 val winner: $T1_WINNER"
    run_step "$RESULTS/$T1_WINNER/eval_results_real_world.json" "plan2-t1 eval winner: $T1_WINNER" \
      $PY -m experiments.plan2_arch.run_arch --config "$P2CFG/${T1_WINNER}.yaml" --eval-only
    run_step "" "plan2-t1 compare: $T1_WINNER" \
      $PY -m experiments.plan2_arch.compare_arch --run "$T1_WINNER"
  else
    echo "!! Tier 1 winner selection produced nothing (both sweep members missing?) - skipping T1 eval." >&2
    FAILED+=("plan2-t1 select")
  fi
fi

# --- [7] Plan 2 Tier 2: res240 (single row, train + eval) ----------------
echo ""; echo "=== [7/9] Plan 2 Tier 2: res240 ==="
run_step "$RESULTS/efficientnetb0_on_res240/eval_results_real_world.json" "plan2-t2: efficientnetb0_on_res240" \
  $PY -m experiments.plan2_arch.run_arch --config "$P2CFG/efficientnetb0_on_res240.yaml"
run_step "" "plan2-t2 compare: efficientnetb0_on_res240" \
  $PY -m experiments.plan2_arch.compare_arch --run efficientnetb0_on_res240

# --- [8] Plan 2 Tier 3: MixStyle (sweep -> winner) + combination row -----
echo ""; echo "=== [8/9] Plan 2 Tier 3: MixStyle (+ combination) ==="
for depth in l12 l123; do
  rn="$(cfg_run_name "$P2CFG/efficientnetb0_on_mixstyle_${depth}.yaml")"; rn="${rn:-efficientnetb0_on_mixstyle_${depth}}"
  run_step "$RESULTS/$rn/best_model.pth" "plan2-t3 train (train-only): $rn" \
    $PY -m experiments.plan2_arch.run_arch --config "$P2CFG/efficientnetb0_on_mixstyle_${depth}.yaml" --train-only
done
if [ "$DRY_RUN" -eq 1 ]; then
  echo "[WOULD RUN] plan2-t3 select_on_val {mixstyle_l12,mixstyle_l123} -> eval winner -> compare_arch"
  echo "[WOULD RUN] plan2-t3 combination row <winner>_bgrand (train+eval, needs rembg) -> compare_arch"
else
  $PY -m experiments.plan2_arch.select_on_val efficientnetb0_on_mixstyle_l12 efficientnetb0_on_mixstyle_l123 || true
  T3_WINNER=$($PY -m experiments.plan2_arch.select_on_val efficientnetb0_on_mixstyle_l12 efficientnetb0_on_mixstyle_l123 --print-winner 2>/dev/null || true)
  if [ -n "${T3_WINNER:-}" ]; then
    echo "Tier 3 val winner: $T3_WINNER"
    run_step "$RESULTS/$T3_WINNER/eval_results_real_world.json" "plan2-t3 eval winner: $T3_WINNER" \
      $PY -m experiments.plan2_arch.run_arch --config "$P2CFG/${T3_WINNER}.yaml" --eval-only
    run_step "" "plan2-t3 compare: $T3_WINNER" \
      $PY -m experiments.plan2_arch.compare_arch --run "$T3_WINNER"

    # Combination row (MixStyle winner + bgrand). Needs rembg; inherits the VAL
    # winner so no selection ever touches the real-world set (read once per row).
    COMBO="${T3_WINNER}_bgrand"
    if [ "$REMBG_OK" -eq 1 ]; then
      ensure_synthetic_bg
      run_step "$RESULTS/$COMBO/eval_results_real_world.json" "plan2-t3 combination: $COMBO" \
        $PY -m experiments.plan2_arch.run_arch --config "$P2CFG/${COMBO}.yaml"
      run_step "" "plan2-t3 combination compare: $COMBO" \
        $PY -m experiments.plan2_arch.compare_arch --run "$COMBO"
    else
      echo "plan2-t3 combination row ($COMBO) SKIPPED - needs rembg (see the pip line above)."
    fi
  else
    echo "!! Tier 3 winner selection produced nothing - skipping T3 eval + combination." >&2
    FAILED+=("plan2-t3 select")
  fi
fi

# --- [9] post-processing (aggregate tables + confusion + seed summary) ---
echo ""; echo "=== [9/9] Post-processing ==="
run_step "" "compile_results"        $PY -m experiments.compile_results
run_step "" "confusion_matrices"     $PY -m experiments.confusion_matrices
run_step "" "compare_seeds"          $PY -m experiments.plan1_bgrand.compare_seeds

# --- summary --------------------------------------------------------------
echo ""
echo "=================================================================="
if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN complete - nothing was executed. Remove --dry-run to train."
elif [ "${#FAILED[@]}" -eq 0 ]; then
  echo "Done. No failures recorded this pass."
else
  echo "Done, but ${#FAILED[@]} step(s) FAILED this pass:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
  echo "Re-run this script to retry only what's missing (finished runs are skipped)."
fi
echo ""
echo "Results:"
echo "  $RESULTS/all_results.md                 master metric table"
echo "  $RESULTS/confusion/confusion_report.txt aggregate confusion counts"
echo "  $RESULTS/<run>/cm_real_world_test.png   per-model confusion matrix"
echo "  $RESULTS/seedrep_summary.json           3-seed mean +/- std + contrasts"
echo "=================================================================="
