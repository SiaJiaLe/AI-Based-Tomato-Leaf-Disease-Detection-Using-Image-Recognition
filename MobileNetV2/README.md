# MobileNetV2 Baseline

Standard-configuration MobileNetV2 baseline for comparison against the
proposed ResNet34 solution in `resnet34_model/`. Trains and evaluates on
the same `resnet34_model/data/processed` split, but uses:

- ImageNet-pretrained weights with a standard `Dropout(0.2) -> Linear` head
- Basic augmentation only (random-resized crop, horizontal flip, small rotation)
- No attention modules, no weighted sampler
- Its own `src/config.py`, with epoch/patience values set to match
  ResNet34's budget (15/25/7) so architecture is the only variable
- Standard cross-entropy loss (no label smoothing)
- Standard fine-tune of the last 5 inverted-residual blocks (own
  architecture-native unfreeze ratio, not copied from ResNet34)

## Usage

```powershell
cd MobileNetV2
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

Outputs are written to `./outputs`.

## Real-world evaluation

Once `resnet34_model/data/real_environment_test/` is populated with field
photos (one subfolder per disease class), run:

```powershell
python src/evaluate_real_world.py
```

Requires training to already be complete. Writes
`eval_results_real_world.json`, `classification_report_real_world.txt`,
and `cm_real_world_test.png` to `./outputs`, alongside the controlled-set
results.
