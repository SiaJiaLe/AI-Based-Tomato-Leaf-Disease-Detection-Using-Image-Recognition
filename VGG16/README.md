# VGG16 Baseline

Standard-configuration VGG16 baseline for comparison against the proposed
ResNet34 solution in `resnet34_model/`. Trains and evaluates on the same
`resnet34_model/data/processed` split, but uses:

- ImageNet-pretrained weights with a standard `Dropout(0.2) -> Linear` head
- Basic augmentation only (random-resized crop, horizontal flip, small rotation)
- No attention modules, no weighted sampler
- Its own independent training budget in `src/config.py`
- Standard cross-entropy loss (no label smoothing)
- Standard fine-tune of block4/block5 (own architecture-native unfreeze
  targets, not copied from the proposed ResNet34's layer3/layer4 scheme)

## Usage

```powershell
cd VGG16
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

Outputs are written to `./outputs`.
