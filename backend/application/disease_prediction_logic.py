import json
import numpy as np
from PIL import Image
import io
from infrastructure.environment_variables import settings
from infrastructure.onnx_inference_engine import onnx_engine
from domain.prediction_data_models import PredictionResult

class PredictionService:
    def __init__(self):
        self.labels = {}
        self.image_size = 224
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load_labels(self):
        if not self.labels:
            with open(settings.LABELS_PATH, "r") as f:
                self.labels = json.load(f)

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ratio = 256.0 / min(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.BILINEAR)
        left = (image.width - self.image_size) / 2
        top = (image.height - self.image_size) / 2
        right = (image.width + self.image_size) / 2
        bottom = (image.height + self.image_size) / 2
        image = image.crop((left, top, right, bottom))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_bytes: bytes) -> PredictionResult:
        self.load_labels()
        input_tensor = self.preprocess_image(image_bytes)
        outputs = onnx_engine.run(input_tensor)
        exp_preds = np.exp(outputs[0] - np.max(outputs[0]))
        probs = exp_preds / exp_preds.sum()
        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx] * 100)
        label = self.labels.get(str(class_idx), f"Unknown class {class_idx}")
        return PredictionResult(disease=label, confidence=round(confidence, 2))

prediction_service = PredictionService()
