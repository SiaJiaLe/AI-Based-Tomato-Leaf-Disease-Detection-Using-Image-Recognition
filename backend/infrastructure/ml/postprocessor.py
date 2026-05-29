"""Post-process model logits (scaffold)."""


class Postprocessor:
  def to_label_and_confidence(self, logits):
    raise NotImplementedError
