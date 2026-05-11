# ai/operation/trt_engine.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os

class ErgoTRTEngine:
    """
    TensorRT Inference Engine for ErgoNet.
    Handles memory allocation and GPU execution.
    """
    def __init__(self, engine_path="ai/models/ergo_net.engine"):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT Engine not found at {engine_path}")
            
        print(f"[TRT] Loading Engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.output_names = ['angles', 'scores', 'anomalies']
        
        for i in range(self.engine.num_bindings):
            is_input = self.engine.binding_is_input(i)
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            
            # For dynamic batching, we assume batch size 1 for simplicity in this wrapper
            if shape[0] == -1:
                shape[0] = 1
            
            size = np.prod(shape)
            nbytes = size * np.dtype(dtype).itemsize
            
            # CUDA Memory allocation
            device_allocation = cuda.mem_alloc(nbytes)
            host_allocation = np.empty(shape, dtype=dtype)
            
            self.allocations.append(int(device_allocation))
            
            if is_input:
                self.inputs.append({"name": name, "host": host_allocation, "device": device_allocation, "shape": shape})
            else:
                self.outputs.append({"name": name, "host": host_allocation, "device": device_allocation, "shape": shape})

    def predict(self, landmarks):
        """
        landmarks: np.ndarray of shape (1, 99) or (99,)
        Returns dict with angles, scores, anomalies
        """
        # 1. Prepare input
        input_data = np.array(landmarks, dtype=np.float32).reshape(1, 99)
        np.copyto(self.inputs[0]["host"], input_data)
        
        # 2. Transfer to GPU
        cuda.memcpy_htod(self.inputs[0]["device"], self.inputs[0]["host"])
        
        # 3. Execute
        self.context.execute_v2(self.allocations)
        
        # 4. Transfer back from GPU
        results = {}
        for out in self.outputs:
            cuda.memcpy_dtoh(out["host"], out["device"])
            results[out["name"]] = out["host"]
            
        return results

if __name__ == "__main__":
    # Test script
    try:
        engine = ErgoTRTEngine()
        dummy_input = np.random.randn(99).astype(np.float32)
        out = engine.predict(dummy_input)
        print("[TRT] Test Success!")
        for k, v in out.items():
            print(f"  {k}: {v.shape}")
    except Exception as e:
        print(f"[TRT] Test Failed: {e}")
