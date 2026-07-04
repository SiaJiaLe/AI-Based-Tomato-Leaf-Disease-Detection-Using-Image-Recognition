# Random Forest Baseline

Classical ML baseline: a frozen, ImageNet-pretrained ResNet34 (fc removed)
extracts 512-dim feature vectors per image; a Random Forest classifier is
fit on those vectors. This is the standard approach in existing
literature for combining CNN features with classical classifiers (Khan
et al. 2021; Tan et al. 2021).

Uses scikit-learn's standard default (`n_estimators=100`, with a fixed
`random_state` only for reproducibility) — not tuned to match the
proposed ResNet34 model in any way.

## Usage

```powershell
cd RandomForest
pip install -r requirements.txt
python src/train_evaluate.py
```

Outputs (`classification_report.txt`, `eval_results.json`,
`cm_processed_test.png`) are written to `./outputs`.
