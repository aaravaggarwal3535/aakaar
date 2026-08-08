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

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, device="cpu", dtype="float32"):
        if dtype != "float32":
            print(f"WARNING: Conv2d(dtype='{dtype}') will be forward-only — no gradients.")

        KH, KW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (KH, KW)
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.dtype = dtype

        np_dtype = {"float32": np.float32, "float64": np.float64,
                    "int32": np.int32, "int64": np.int64}[dtype]
        requires_grad = (dtype == "float32")

        fan_in = in_channels * KH * KW
        if dtype in ("float32", "float64"):
            limit = 1.0 / np.sqrt(fan_in)
            w = np.random.uniform(-limit, limit, (out_channels, in_channels, KH, KW)).astype(np_dtype)
        else:
            w = np.zeros((out_channels, in_channels, KH, KW), dtype=np_dtype)
        self.weight = from_numpy(w, device=device, requires_grad=requires_grad)

        self.has_bias = bias
        if bias:
            if dtype in ("float32", "float64"):
                bound = 1.0 / np.sqrt(fan_in)
                b = np.random.uniform(-bound, bound, (1, out_channels, 1, 1)).astype(np_dtype)
            else:
                b = np.zeros((1, out_channels, 1, 1), dtype=np_dtype)
            self.bias = from_numpy(b, device=device, requires_grad=requires_grad)
        else:
            self.bias = None

    def __call__(self, x):
        if len(x.shape) != 4:
            raise ValueError(f"Conv2d expects input of shape (batch, in_channels, H, W), got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Conv2d: expected {self.in_channels} input channels, got {x.shape[1]}")
        out = aakaar.conv2d(x, self.weight, self.stride, self.padding, self.dilation)
        if self.has_bias:
            out = out + self.bias
        return out

    def parameters(self):
        return [self.weight, self.bias] if self.has_bias else [self.weight]

class Conv3d:
    """3D convolution: y = conv(x, weight) + bias.

    x:      (batch, in_channels, D, H, W)
    weight: (out_channels, in_channels, KD, KH, KW)
    y:      (batch, out_channels, OD, OH, OW)

    Same design as Conv1d/Conv2d: im2col (custom CPU/CUDA kernel) + matmul
    fallback, or cuDNN when available. groups>1 not implemented.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, device="cpu", dtype="float32"):
        if dtype != "float32":
            print(f"WARNING: Conv3d(dtype='{dtype}') will be forward-only — no gradients. "
                  f"Only float32 supports autograd anywhere in aakaar right now.")

        KD, KH, KW = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (KD, KH, KW)
        self.stride = (stride,) * 3 if isinstance(stride, int) else stride
        self.padding = (padding,) * 3 if isinstance(padding, int) else padding
        self.dilation = (dilation,) * 3 if isinstance(dilation, int) else dilation
        self.dtype = dtype

        np_dtype = {"float32": np.float32, "float64": np.float64,
                    "int32": np.int32, "int64": np.int64}[dtype]
        requires_grad = (dtype == "float32")

        fan_in = in_channels * KD * KH * KW
        if dtype in ("float32", "float64"):
            limit = 1.0 / np.sqrt(fan_in)
            w = np.random.uniform(-limit, limit,
                                   (out_channels, in_channels, KD, KH, KW)).astype(np_dtype)
        else:
            w = np.zeros((out_channels, in_channels, KD, KH, KW), dtype=np_dtype)

        self.weight = from_numpy(w, device=device, requires_grad=requires_grad)

        self.has_bias = bias
        if bias:
            if dtype in ("float32", "float64"):
                bound = 1.0 / np.sqrt(fan_in)
                b = np.random.uniform(-bound, bound, (1, out_channels, 1, 1, 1)).astype(np_dtype)
            else:
                b = np.zeros((1, out_channels, 1, 1, 1), dtype=np_dtype)
            self.bias = from_numpy(b, device=device, requires_grad=requires_grad)
        else:
            self.bias = None

    def __call__(self, x):
        if len(x.shape) != 5:
            raise ValueError(f"Conv3d expects input of shape (batch, in_channels, D, H, W), got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Conv3d: expected {self.in_channels} input channels, got {x.shape[1]}")
        out = aakaar.conv3d(x, self.weight, self.stride, self.padding, self.dilation)
        if self.has_bias:
            out = out + self.bias  # already hits the fast channel-bias add-kernel path
        return out

    def parameters(self):
        return [self.weight, self.bias] if self.has_bias else [self.weight]

class ConvTranspose1d:
    """1D transposed convolution ("deconvolution"): y = conv_transpose(x, weight) + bias.

    x:      (batch, in_channels, length)
    weight: (in_channels, out_channels, kernel_size) -- REVERSED channel
            order vs Conv1d, matching torch's ConvTranspose1d.
    y:      (batch, out_channels, out_length)

    Implemented as the adjoint of Conv1d (see aakaar.conv_transpose1d) —
    reuses Conv1d's existing cuDNN kernels, no new low-level code.
    output_padding is not yet supported. groups>1 not implemented (same
    limitation as Conv1d/Conv2d/Conv3d).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, dilation=1, bias=True,
                 device="cpu", dtype="float32"):
        if dtype != "float32":
            print(f"WARNING: ConvTranspose1d(dtype='{dtype}') will be forward-only — no gradients. "
                  f"Only float32 supports autograd anywhere in aakaar right now.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.dtype = dtype

        np_dtype = {"float32": np.float32, "float64": np.float64,
                    "int32": np.int32, "int64": np.int64}[dtype]
        requires_grad = (dtype == "float32")

        fan_in = in_channels * kernel_size
        if dtype in ("float32", "float64"):
            limit = 1.0 / np.sqrt(fan_in)
            w = np.random.uniform(-limit, limit,
                                   (in_channels, out_channels, kernel_size)).astype(np_dtype)
        else:
            w = np.zeros((in_channels, out_channels, kernel_size), dtype=np_dtype)

        self.weight = from_numpy(w, device=device, requires_grad=requires_grad)

        self.has_bias = bias
        if bias:
            if dtype in ("float32", "float64"):
                bound = 1.0 / np.sqrt(fan_in)
                b = np.random.uniform(-bound, bound, (1, out_channels, 1)).astype(np_dtype)
            else:
                b = np.zeros((1, out_channels, 1), dtype=np_dtype)
            self.bias = from_numpy(b, device=device, requires_grad=requires_grad)
        else:
            self.bias = None

    def __call__(self, x):
        if len(x.shape) != 3:
            raise ValueError(f"ConvTranspose1d expects input of shape (batch, in_channels, length), got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"ConvTranspose1d: expected {self.in_channels} input channels, got {x.shape[1]}")
        out = aakaar.conv_transpose1d(x, self.weight, self.stride, self.padding,
                                       self.output_padding, self.dilation)
        if self.has_bias:
            out = out + self.bias  # gets the fast channel-bias-add path for free
        return out

    def parameters(self):
        return [self.weight, self.bias] if self.has_bias else [self.weight]

class ConvTranspose2d:
    """2D transposed convolution. See ConvTranspose1d's docstring — same
    adjoint-of-Conv2d implementation, no new low-level kernel code.
    output_padding not yet supported. groups>1 not implemented."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, dilation=1, bias=True,
                 device="cpu", dtype="float32"):
        if dtype != "float32":
            print(f"WARNING: ConvTranspose2d(dtype='{dtype}') will be forward-only — no gradients.")

        KH, KW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (KH, KW)
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.dtype = dtype

        np_dtype = {"float32": np.float32, "float64": np.float64,
                    "int32": np.int32, "int64": np.int64}[dtype]
        requires_grad = (dtype == "float32")

        fan_in = in_channels * KH * KW
        if dtype in ("float32", "float64"):
            limit = 1.0 / np.sqrt(fan_in)
            w = np.random.uniform(-limit, limit, (in_channels, out_channels, KH, KW)).astype(np_dtype)
        else:
            w = np.zeros((in_channels, out_channels, KH, KW), dtype=np_dtype)
        self.weight = from_numpy(w, device=device, requires_grad=requires_grad)

        self.has_bias = bias
        if bias:
            if dtype in ("float32", "float64"):
                bound = 1.0 / np.sqrt(fan_in)
                b = np.random.uniform(-bound, bound, (1, out_channels, 1, 1)).astype(np_dtype)
            else:
                b = np.zeros((1, out_channels, 1, 1), dtype=np_dtype)
            self.bias = from_numpy(b, device=device, requires_grad=requires_grad)
        else:
            self.bias = None

    def __call__(self, x):
        if len(x.shape) != 4:
            raise ValueError(f"ConvTranspose2d expects input of shape (batch, in_channels, H, W), got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"ConvTranspose2d: expected {self.in_channels} input channels, got {x.shape[1]}")
        out = aakaar.conv_transpose2d(x, self.weight, self.stride, self.padding,
                                       self.output_padding, self.dilation)
        if self.has_bias:
            out = out + self.bias
        return out

    def parameters(self):
        return [self.weight, self.bias] if self.has_bias else [self.weight]


class ConvTranspose3d:
    """3D transposed convolution. Same adjoint-of-Conv3d design.
    output_padding not yet supported. groups>1 not implemented."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, dilation=1, bias=True,
                 device="cpu", dtype="float32"):
        if dtype != "float32":
            print(f"WARNING: ConvTranspose3d(dtype='{dtype}') will be forward-only — no gradients.")

        KD, KH, KW = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (KD, KH, KW)
        self.stride = (stride,) * 3 if isinstance(stride, int) else stride
        self.padding = (padding,) * 3 if isinstance(padding, int) else padding
        self.output_padding = (output_padding,) * 3 if isinstance(output_padding, int) else output_padding
        self.dilation = (dilation,) * 3 if isinstance(dilation, int) else dilation
        self.dtype = dtype

        np_dtype = {"float32": np.float32, "float64": np.float64,
                    "int32": np.int32, "int64": np.int64}[dtype]
        requires_grad = (dtype == "float32")

        fan_in = in_channels * KD * KH * KW
        if dtype in ("float32", "float64"):
            limit = 1.0 / np.sqrt(fan_in)
            w = np.random.uniform(-limit, limit, (in_channels, out_channels, KD, KH, KW)).astype(np_dtype)
        else:
            w = np.zeros((in_channels, out_channels, KD, KH, KW), dtype=np_dtype)
        self.weight = from_numpy(w, device=device, requires_grad=requires_grad)

        self.has_bias = bias
        if bias:
            if dtype in ("float32", "float64"):
                bound = 1.0 / np.sqrt(fan_in)
                b = np.random.uniform(-bound, bound, (1, out_channels, 1, 1, 1)).astype(np_dtype)
            else:
                b = np.zeros((1, out_channels, 1, 1, 1), dtype=np_dtype)
            self.bias = from_numpy(b, device=device, requires_grad=requires_grad)
        else:
            self.bias = None

    def __call__(self, x):
        if len(x.shape) != 5:
            raise ValueError(f"ConvTranspose3d expects input of shape (batch, in_channels, D, H, W), got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"ConvTranspose3d: expected {self.in_channels} input channels, got {x.shape[1]}")
        out = aakaar.conv_transpose3d(x, self.weight, self.stride, self.padding,
                                       self.output_padding, self.dilation)
        if self.has_bias:
            out = out + self.bias
        return out

    def parameters(self):
        return [self.weight, self.bias] if self.has_bias else [self.weight]