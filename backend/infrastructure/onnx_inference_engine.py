"""ONNX inference engine singleton — loaded once at startup."""
import onnxruntime as ort
from shared.config import settings


class ONNXEngine:
    def __init__(self):
        self.session = None
        self.input_name = ""
        self.output_name = ""

    def load(self):
        self.load_from_path(settings.model_path)

    def load_from_path(self, path: str):
        try:
            self.session = ort.InferenceSession(path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"ONNX model loaded from {path}")
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def run(self, input_tensor):
        if self.session is None:
            raise RuntimeError("ONNX engine not loaded. Call load() first.")
        return self.session.run([self.output_name], {self.input_name: input_tensor})


onnx_engine = ONNXEngine()
