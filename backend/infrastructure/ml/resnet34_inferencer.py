"""ResNet34 inferencer (scaffold)."""


class ResNet34Inferencer:
  def __init__(self, model_path: str, labels_path: str):
    self.model_path = model_path
    self.labels_path = labels_path

  def predict(self, image):
    raise NotImplementedError
