import os
import torch

from config import OUTPUT_DIR, IMAGE_SIZE
from dataset import get_dataloaders
from model import TomatoResNet34

def export_to_onnx():
    device = torch.device("cpu") # ONNX export is usually done on CPU
    print("Preparing to export model to ONNX format...")
    
    # 1. Detect classes
    _, _, _, class_to_idx = get_dataloaders()
    num_classes = len(class_to_idx)
    
    # 2. Load the best model weights
    model = TomatoResNet34(num_classes).to(device)
    weights_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found. Train the model first!")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 3. Create a dummy input tensor of the correct shape [Batch, Channels, Height, Width]
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    
    # 4. Define the output path
    onnx_path = os.path.join(OUTPUT_DIR, "best_model.onnx")
    
    # 5. Export!
    print("Exporting...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=11,          # Standard ONNX opset version
        do_constant_folding=True,  # Optimize constant operations
        input_names=['input'],     # Name the input tensor
        output_names=['output'],   # Name the output tensor
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Allow variable batch sizes for the backend
    )
    
    print(f"Successfully exported model to {onnx_path}!")
    print("This lightweight .onnx file is completely decoupled from PyTorch and ready to be deployed to your Backend!")

if __name__ == "__main__":
    export_to_onnx()
