# Plan 9 — Drop Target_Spot + mosaic_virus, retrain on 8 classes

## Why
On the new data Target_Spot became a SINK (baseline predicts it for ~81% of real
images) and mosaic_virus is a persistent zero. User's read: their PlantVillage training
images resemble healthy leaves, so they corrupt the other classes. Dropping both should
let the remaining 8 classes stop leaking into Target_Spot.

## The classes to exclude
    Tomato___Target_Spot
    Tomato___Tomato_mosaic_virus
Remaining 8: Bacterial_spot, Early_blight, Late_blight, Leaf_Mold, Septoria_leaf_spot,
Spider_mites Two-spotted_spider_mite, Tomato_Yellow_Leaf_Curl_Virus, healthy.

## Critical: BOTH sides must be 8 classes
- TRAINING: exclude the two class folders when splitting data/raw -> data/processed, so
  ImageFolder builds 8-class models.
- REAL-WORLD EVAL: common/evaluate.py remaps real-world folders into the training label
  space BY NAME -> a real-world folder with no matching training class is a KeyError.
  So the two folders must also leave data/real_environment_dataset. They are MOVED (not
  deleted) to data/real_environment_dataset_excluded/ so nothing is lost and it is a
  one-line undo.

## Changes (additive / to files I own; no contract file, no config edited)
1. `experiments/split_dataset.py`: add `--exclude <class> ...`. Excluded class folders are
   skipped when building processed/{train,val,test}. Their raw images stay on disk.
2. `experiments/retrain_all.sh`: define
   `EXCLUDE=("Tomato___Target_Spot" "Tomato___Tomato_mosaic_virus")`, pass it to
   split_dataset, and (idempotently) move any excluded class folder out of
   data/real_environment_dataset into data/real_environment_dataset_excluded/ so the eval
   set matches. Everything else (archive, submit all jobs, postprocess) is unchanged.
3. Update tmp/test_split_dataset.py to cover --exclude (excluded class gets no split dirs;
   the rest are unaffected).

## What automatically adapts (no change needed)
- All runners read num_classes from the dataset -> 8-class heads.
- confusion_matrices / compile_results / compare_seeds read class names from each run's
  own outputs -> they render 8x8 / 8-row tables. The 10-entry code map just uses 8 of them.
- real_world_dir in every config still points at data/real_environment_dataset (now 8
  folders) -> NO config edit, so the isolation contract is untouched.

## Caveats to state
- Label indices change (8 classes, re-sorted) -> these 8-class models are NOT comparable
  cell-for-cell to the 10-class runs; they are a separate study. The 10-class results are
  already archived in experiments/results_archive_<ts>/ from the last retrain.
- The backend's class_labels.json expects 10 classes; this is a RESEARCH retrain only and
  does not touch the backend.
- Removing Target_Spot removes the sink, so baseline macro-F1 should jump for reasons that
  are partly bookkeeping (one bad class gone) as well as real (less leakage). Report the 8
  remaining classes honestly as a scoped study, noting the two were excluded and WHY
  (persistent zero / sink), with the confusion evidence.

## Run (HPC, after git pull)
    bash experiments/retrain_all.sh          # now excludes the two classes on both sides
Then the same postprocess produces 8-class tables + confusion matrices + seed summary.

## Test
Run the updated tmp/test_split_dataset.py before committing.
