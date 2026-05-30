import onnxruntime as ort
from infrastructure.environment_variables import settings

class ONNXEngine:
    def __init__(self):
        self.session = None
        self.input_name = ""
        self.output_name = ""

    def load(self):
        try:
            self.session = ort.InferenceSession(settings.MODEL_PATH)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"ONNX Engine loaded successfully from {settings.MODEL_PATH}")
        except Exception as e:
            print(f"Error loading ONNX model: {str(e)}")

    def run(self, input_tensor):
        if self.session is None:
            self.load()
        return self.session.run([self.output_name], {self.input_name: input_tensor})[0]

onnx_engine = ONNXEngine()
