"""Post-processor — converts raw ONNX logits to label + confidence."""
import numpy as np


class Postprocessor:
    def to_label_and_confidence(
        self, logits: np.ndarray, labels: dict
    ) -> tuple[str, float]:
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        top_idx = int(np.argmax(probs))
        return labels.get(str(top_idx), f"class_{top_idx}"), float(probs[top_idx] * 100)

    def top_k(
        self, logits: np.ndarray, labels: dict, k: int = 3
    ) -> list[dict]:
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        top_indices = np.argsort(probs)[-k:][::-1]
        return [
            {
                "label": labels.get(str(int(i)), f"class_{i}"),
                "confidence": round(float(probs[i] * 100), 2),
            }
            for i in top_indices
        ]
