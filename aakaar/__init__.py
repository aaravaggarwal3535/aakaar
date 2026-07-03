from . import _C
import numpy as np

def rand(size: int, device: str = "cpu", seed: int = 42):
    # 1. Create the Tensor (The C++ constructor handles allocation)
    t = _C.Tensor(size, device)
    
    # 2. Route to the correct backend
    if device == "cuda":
        # Call your existing CUDA random generator
        _C.generate_random(t, seed)
    else:
        # Call the CPU random generator
        _C.fill_cpu_random(t, seed)
        
    return t