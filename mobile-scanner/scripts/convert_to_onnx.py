"""Convert best_model.pth (EfficientNet-B0 + CBAM + strong head, 8-class) to ONNX
for in-browser inference with onnxruntime-web.

Reuses the project's own architecture builder (experiments/common) READ-ONLY so
the exported graph is byte-identical in structure to the trained model, then:
  1. loads the checkpoint state_dict,
  2. writes class_labels.json straight from checkpoint['class_to_idx'] (no hand-typing),
  3. exports FP32 ONNX (opset 17, fixed batch 1, input 1x3x224x224),
  4. exports an int8 dynamic-quantized variant for the size/latency comparison.

Run from the repo root:  python mobile-scanner/scripts/convert_to_onnx.py
"""
import sys
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]          # repo root
sys.path.insert(0, str(REPO))                        # import experiments.* read-only
from experiments.common.backbones import build_backbone  # noqa: E402

CKPT = REPO / "best_model.pth"
OUT_DIR = REPO / "mobile-scanner" / "public" / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading checkpoint: {CKPT}")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
class_to_idx = ck["class_to_idx"]
cfg = ck["config"]
num_classes = len(class_to_idx)
backbone = cfg["backbone"]
strong = cfg["stack"]["strong_head"]
cbam = cfg["stack"]["cbam"]
print(f"  backbone={backbone}  num_classes={num_classes}  strong_head={strong}  cbam={cbam}")
print(f"  val_macro_f1={ck.get('val_macro_f1')}")

# --- rebuild exact architecture and load weights -----------------------------
built = build_backbone(backbone, num_classes, strong, cbam)
built.warm_up(torch.device("cpu"))          # materialize lazy CBAM channel branches
missing, unexpected = built.module.load_state_dict(ck["state_dict"], strict=False)
if missing or unexpected:
    print(f"  WARNING missing={missing}\n          unexpected={unexpected}")
else:
    print("  state_dict loaded (strict match)")
built.module.eval()

# --- class_labels.json (idx -> class name, ordered 0..N-1) -------------------
idx_to_class = {v: k for k, v in class_to_idx.items()}
labels = {str(i): idx_to_class[i] for i in range(num_classes)}
labels_path = OUT_DIR / "class_labels.json"
labels_path.write_text(json.dumps(labels, indent=2))
print(f"  wrote {labels_path} ({num_classes} classes)")

# --- FP32 ONNX export --------------------------------------------------------
dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
fp32_path = OUT_DIR / "best_model.onnx"
torch.onnx.export(
    built.module, dummy, str(fp32_path),
    input_names=["input"], output_names=["logits"],
    opset_version=17,
)
size_fp32 = fp32_path.stat().st_size
print(f"  FP32 ONNX  -> {fp32_path}  ({size_fp32/1e6:.2f} MB)")

# --- int8 dynamic quantization ----------------------------------------------
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8_path = OUT_DIR / "best_model.int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    size_int8 = int8_path.stat().st_size
    print(f"  INT8 ONNX  -> {int8_path}  ({size_int8/1e6:.2f} MB, "
          f"{100*size_int8/size_fp32:.0f}% of FP32)")
except Exception as e:  # noqa: BLE001
    print(f"  INT8 quantization skipped: {e}")

# --- parity self-check: ONNX vs PyTorch on the same input --------------------
try:
    import numpy as np
    import onnxruntime as ort
    with torch.no_grad():
        torch_out = built.module(dummy).numpy()
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"  parity check: max|torch-onnx| = {max_diff:.2e} "
          f"({'OK' if max_diff < 1e-3 else 'CHECK'})")
except Exception as e:  # noqa: BLE001
    print(f"  parity check skipped: {e}")

print("Done.")
