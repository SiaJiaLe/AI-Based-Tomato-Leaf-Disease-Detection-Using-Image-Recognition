"""ResNet34 ONNX inferencer with 6-variant Test-Time Augmentation."""
import io
import json

import numpy as np
from PIL import Image, ImageEnhance

from infrastructure.ml.preprocessor import Preprocessor
from infrastructure.ml.postprocessor import Postprocessor
from infrastructure.onnx_inference_engine import ONNXEngine


class ResNet34Inferencer:
    def __init__(self, engine: ONNXEngine, labels: dict):
        self.engine = engine
        self.labels = labels
        self._preprocessor = Preprocessor()
        self._postprocessor = Postprocessor()

    @classmethod
    def from_paths(cls, model_path: str, labels_path: str) -> "ResNet34Inferencer":
        engine = ONNXEngine()
        engine.load_from_path(model_path)
        with open(labels_path, "r") as f:
            labels = json.load(f)
        return cls(engine, labels)

    def _tta_variants(self, image: Image.Image) -> list[Image.Image]:
        enhancer = ImageEnhance.Brightness(image)
        return [
            image,
            image.transpose(Image.FLIP_LEFT_RIGHT),
            image.rotate(10, resample=Image.BILINEAR),
            image.rotate(-10, resample=Image.BILINEAR),
            enhancer.enhance(1.2),
            enhancer.enhance(0.8),
        ]

    def _avg_probs(self, image: Image.Image) -> np.ndarray:
        all_probs = []
        for variant in self._tta_variants(image):
            tensor = self._preprocessor.transform(variant)
            logits = self.engine.run(tensor)[0].squeeze()
            exp = np.exp(logits - np.max(logits))
            all_probs.append(exp / exp.sum())
        return np.mean(all_probs, axis=0)

    def predict(self, image_bytes: bytes) -> tuple[str, float, list[dict]]:
        """Returns (top_label, confidence_pct, top3_alternatives)."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        avg = self._avg_probs(image)

        top_idx = int(np.argmax(avg))
        confidence = round(float(avg[top_idx] * 100), 2)
        label = self.labels.get(str(top_idx), f"class_{top_idx}")

        top3_indices = np.argsort(avg)[-3:][::-1]
        alternatives = [
            {
                "label": self.labels.get(str(int(i)), f"class_{i}"),
                "confidence": round(float(avg[i] * 100), 2),
            }
            for i in top3_indices
            if int(i) != top_idx
        ]

        return label, confidence, alternatives

    def predict_image(self, image: Image.Image) -> tuple[str, float, list[dict]]:
        """Predict from an already-opened PIL image (used by session endpoint)."""
        avg = self._avg_probs(image)
        top_idx = int(np.argmax(avg))
        confidence = round(float(avg[top_idx] * 100), 2)
        label = self.labels.get(str(top_idx), f"class_{top_idx}")

        top3_indices = np.argsort(avg)[-3:][::-1]
        alternatives = [
            {
                "label": self.labels.get(str(int(i)), f"class_{i}"),
                "confidence": round(float(avg[i] * 100), 2),
            }
            for i in top3_indices
            if int(i) != top_idx
        ]
        return label, confidence, alternatives
