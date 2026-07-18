import numpy as np
from . import from_numpy, matmul, im2col_1d
import aakaar

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

class Conv1d:
    """1D convolution: y = conv(x, weight) + bias.

    x:      (batch, in_channels, length)
    weight: (out_channels, in_channels, kernel_size)
    y:      (batch, out_channels, out_length)

    Implemented as im2col (custom CPU/CUDA kernel) + matmul (cuBLAS/OpenBLAS-
    backed, reusing aakaar's existing autograd). groups>1 and non-zero
    padding modes are not implemented.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, device="cpu", dtype="float32"):
        if dtype != "float32":
            print(f"WARNING: Conv1d(dtype='{dtype}') will be forward-only — no gradients. "
                  f"Only float32 supports autograd anywhere in aakaar right now.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dtype = dtype

        np_dtype = {"float32": np.float32, "float64": np.float64,
                    "int32": np.int32, "int64": np.int64}[dtype]
        requires_grad = (dtype == "float32")

        fan_in = in_channels * kernel_size
        if dtype in ("float32", "float64"):
            limit = 1.0 / np.sqrt(fan_in)
            w = np.random.uniform(-limit, limit,
                                   (out_channels, in_channels, kernel_size)).astype(np_dtype)
        else:
            # No principled random init for integer weights — zero-init and
            # flagged; set self.weight manually if you actually need this.
            w = np.zeros((out_channels, in_channels, kernel_size), dtype=np_dtype)

        self.weight = from_numpy(w, device=device, requires_grad=requires_grad)

        self.has_bias = bias
        if bias:
            if dtype in ("float32", "float64"):
                bound = 1.0 / np.sqrt(fan_in)
                b = np.random.uniform(-bound, bound, (1, out_channels, 1)).astype(np_dtype)
            else:
                b = np.zeros((1, out_channels, 1), dtype=np_dtype)  # same int caveat as weights
            self.bias = from_numpy(b, device=device, requires_grad=requires_grad)
        else:
            self.bias = None

    def _out_length(self, L_in):
        return (L_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

    def __call__(self, x):
        if len(x.shape) != 3:
            raise ValueError(f"Conv1d expects input of shape (batch, in_channels, length), got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Conv1d: expected {self.in_channels} input channels, got {x.shape[1]}")
        out = aakaar.conv1d(x, self.weight, self.stride, self.padding, self.dilation)
        if self.has_bias:
            out = out + self.bias
        return out

    def parameters(self):
        return [self.weight, self.bias] if self.has_bias else [self.weight]