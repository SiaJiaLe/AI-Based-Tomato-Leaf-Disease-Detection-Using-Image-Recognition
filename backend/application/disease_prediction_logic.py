import json
import numpy as np
from PIL import Image, ImageEnhance
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

    def get_tta_variants(self, image: Image.Image) -> list[Image.Image]:
        """Generates variants for Test-Time Augmentation (TTA)"""
        variants = [
            image,                                        # 1. Original
            image.transpose(Image.FLIP_LEFT_RIGHT),       # 2. Horizontal Flip
            image.rotate(10, resample=Image.BILINEAR),    # 3. Slight Rotation Right
            image.rotate(-10, resample=Image.BILINEAR),   # 4. Slight Rotation Left
        ]
        enhancer = ImageEnhance.Brightness(image)
        variants.append(enhancer.enhance(1.2))            # 5. Brighter
        variants.append(enhancer.enhance(0.8))            # 6. Darker
        return variants

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
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
        
        # Open base image
        base_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Generate TTA variants
        variants = self.get_tta_variants(base_image)
        
        all_probs = []
        for variant in variants:
            input_tensor = self.preprocess_image(variant)
            outputs = onnx_engine.run(input_tensor)
            # Softmax calculation
            exp_preds = np.exp(outputs[0] - np.max(outputs[0]))
            probs = exp_preds / exp_preds.sum()
            all_probs.append(probs)
            
        # Average the probabilities across all TTA variants
        avg_probs = np.mean(all_probs, axis=0)
        
        class_idx = np.argmax(avg_probs)
        confidence = float(avg_probs[0][class_idx] * 100) if avg_probs.ndim > 1 else float(avg_probs[class_idx] * 100)
        label = self.labels.get(str(class_idx), f"Unknown class {class_idx}")
        
        return PredictionResult(disease=label, confidence=round(confidence, 2))

prediction_service = PredictionService()
