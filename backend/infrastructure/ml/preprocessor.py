"""Image preprocessor — must match training transforms (scaffold)."""


class Preprocessor:
  def transform(self, image):
    raise NotImplementedError
