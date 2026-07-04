# SVM Baseline

Classical ML baseline: a frozen, ImageNet-pretrained ResNet34 (fc removed)
extracts 512-dim feature vectors per image; an SVM classifier is fit on
those vectors. This is the standard approach in existing literature for
combining CNN features with classical classifiers (Khan et al. 2021;
Tan et al. 2021).

Uses scikit-learn's standard defaults (`kernel='rbf'`, `C=1.0`,
`gamma='scale'`) — not tuned to match the proposed ResNet34 model in any
way.

## Usage

```powershell
cd SVM
pip install -r requirements.txt
python src/train_evaluate.py
```

Outputs (`classification_report.txt`, `eval_results.json`,
`cm_processed_test.png`) are written to `./outputs`, along with
`classifier.joblib` and `class_labels.json` so the fitted classifier can
be reused without refitting.

## Real-world evaluation

Once `resnet34_model/data/real_environment_test/` is populated with field
photos (one subfolder per disease class), run:

```powershell
python src/evaluate_real_world.py
```

Requires `train_evaluate.py` to already have been run. Writes
`eval_results_real_world.json`, `classification_report_real_world.txt`,
and `cm_real_world_test.png` to `./outputs`.
