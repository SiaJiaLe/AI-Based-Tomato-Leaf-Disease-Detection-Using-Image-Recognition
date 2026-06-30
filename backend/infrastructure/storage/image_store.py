"""Uploaded image storage — saves images to the uploads directory."""
import os
import uuid

from shared.config import settings


class ImageStore:
    def save(self, file_bytes: bytes, original_filename: str) -> str:
        """Saves bytes to UPLOAD_DIR/<uuid><ext> and returns the relative path."""
        os.makedirs(settings.upload_dir, exist_ok=True)
        ext = os.path.splitext(original_filename)[1] or ".jpg"
        filename = f"{uuid.uuid4().hex}{ext}"
        full_path = os.path.join(settings.upload_dir, filename)
        with open(full_path, "wb") as f:
            f.write(file_bytes)
        return filename
