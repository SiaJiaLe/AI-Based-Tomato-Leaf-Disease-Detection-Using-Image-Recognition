"""Uploaded image storage (scaffold)."""


class ImageStore:
  def save(self, file_bytes: bytes, filename: str) -> str:
    raise NotImplementedError
