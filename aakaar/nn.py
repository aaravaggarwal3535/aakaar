import numpy as np
from . import from_numpy, matmul

class Linear:
    """A fully-connected layer: y = x @ weight + bias."""
    def __init__(self, in_features, out_features, device="cpu"):
        limit = 1.0 / np.sqrt(in_features)
        w = np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32)
        b = np.zeros((1, out_features), dtype=np.float32)
        self.weight = from_numpy(w, device=device, requires_grad=True)
        self.bias = from_numpy(b, device=device, requires_grad=True)

    def __call__(self, x):
        return matmul(x, self.weight) + self.bias

    def parameters(self):
        return [self.weight, self.bias]