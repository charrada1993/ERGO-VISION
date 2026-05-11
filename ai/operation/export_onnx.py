# ai/operation/export_onnx.py
import torch
import os
import sys

# Add project root to path to ensure ai.model is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai.model import ErgoNet

def export_to_onnx():
    print("[AI] Initializing ErgoNet for ONNX export...")
    
    # 1. Instantiate model
    model = ErgoNet(input_dim=99, hidden_dim=512)
    model.eval() # Set to evaluation mode (disables dropout/batchnorm training)

    # 2. Load weights if available (Optional)
    # weights_path = "ai/models/ergo_net.pth"
    # if os.path.exists(weights_path):
    #     model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    #     print(f"[AI] Loaded weights from {weights_path}")
    # else:
    #     print("[AI] WARNING: No weights found. Exporting model with random initialization.")

    # 3. Create dummy input (Batch size 1, 99 landmark coordinates)
    dummy_input = torch.randn(1, 99)

    # 4. Define output directory
    os.makedirs("ai/models", exist_ok=True)
    onnx_path = "ai/models/ergo_net.onnx"

    # 5. Export
    print(f"[AI] Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,        # Store trained parameter weights inside the model file
        opset_version=17,          # Use modern ONNX opset
        do_constant_folding=True,  # Optimization: fold constant nodes
        input_names=['landmarks'], # Name of the input layer
        output_names=['angles', 'scores', 'anomalies'], # Names of multi-head outputs
        dynamic_axes={
            'landmarks': {0: 'batch_size'}, # Allow variable batch sizes
            'angles': {0: 'batch_size'},
            'scores': {0: 'batch_size'},
            'anomalies': {0: 'batch_size'}
        }
    )
    
    print("[AI] ONNX Export Complete.")
    
    # Optional: Verify with ONNX library if installed
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[AI] ONNX Model verified successfully.")
    except ImportError:
        print("[AI] Note: 'onnx' library not installed. Skipping verification.")

if __name__ == "__main__":
    export_to_onnx()
