"""Image preprocessor — transforms must exactly match training val/test pipeline."""
import numpy as np
from PIL import Image


class Preprocessor:
    IMAGE_SIZE = 224
    RESIZE_TO = 256
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def transform(self, image: Image.Image) -> np.ndarray:
        """Resize(256) → CenterCrop(224) → normalize → NCHW float32 batch tensor."""
        # Resize shortest side to 256
        ratio = self.RESIZE_TO / min(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.BILINEAR)

        # Center crop to 224x224
        left = (image.width - self.IMAGE_SIZE) / 2
        top = (image.height - self.IMAGE_SIZE) / 2
        image = image.crop((
            int(left), int(top),
            int(left + self.IMAGE_SIZE), int(top + self.IMAGE_SIZE),
        ))

        # HWC float32, normalize with ImageNet stats
        arr = np.array(image, dtype=np.float32) / 255.0
        arr = (arr - self.MEAN) / self.STD

        # CHW → NCHW batch dim
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)
