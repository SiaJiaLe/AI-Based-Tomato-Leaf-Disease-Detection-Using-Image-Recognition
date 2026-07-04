# AlexNet Baseline

Standard-configuration AlexNet baseline for comparison against the proposed
ResNet34 solution in `resnet34_model/`. Trains and evaluates on the same
`resnet34_model/data/processed` split, but uses:

- ImageNet-pretrained weights with a standard `Dropout(0.2) -> Linear` head
  (not the proposed model's BatchNorm/Dropout/2-layer head)
- Basic augmentation only: random-resized crop, horizontal flip, small
  rotation (not the proposed model's advanced field-condition pipeline)
- No attention modules, no weighted sampler
- Its own `src/config.py` (not imported from `resnet34_model/src/config.py`),
  with epoch/patience values set to match ResNet34's budget (15/25/7)
  so architecture is the only variable in the comparison
- Standard cross-entropy loss (no label smoothing)

## Usage

```powershell
cd AlexNet
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

Outputs (`best_model.pth`, `class_labels.json`, `training_history.png`,
`classification_report.txt`, `eval_results.json`, `cm_processed_test.png`)
are written to `./outputs`.
