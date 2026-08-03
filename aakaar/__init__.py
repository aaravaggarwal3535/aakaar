import os
import sys
import contextlib

__version__ = "0.2.1"


def _add_windows_cuda_dll_dirs():
    if sys.platform != "win32":
        return

    # torch's official Windows CUDA wheels bundle their OWN cuDNN DLLs
    # directly inside torch/lib — a completely separate installation from
    # this package's pip-installed nvidia-cudnn-cu* dependency. If both
    # get registered on the DLL search path in the same process, cuDNN's
    # lazy per-sub-library loading (cudnn_ops64_9.dll, cudnn_cnn64_9.dll,
    # cudnn_graph64_9.dll, ...) can resolve different sub-libraries from
    # the two different builds, which cuDNN detects and refuses with
    # CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.
    #
    # There's no version-pinning fix that survives a torch upgrade, so
    # instead: whenever torch is already imported, skip registering
    # aakaar's own cuDNN directory entirely and defer to whatever cuDNN
    # torch already made resolvable. os.add_dll_directory() is process-
    # wide, so torch's directory (registered when `import torch` ran) is
    # already visible here — aakaar just needs to not add a competing one.
    # aakaar's cuDNN calls only use cuDNN 9.x's stable API surface, so
    # this works regardless of the exact minor/patch version torch bundles.
    torch_already_loaded = "torch" in sys.modules

    import site
    search_roots = site.getsitepackages() + [site.getusersitepackages()]
    for root in search_roots:
        nvidia_root = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for entry in os.listdir(nvidia_root):
            if entry.startswith("cu") and entry[2:].isdigit():
                consolidated_bin = os.path.join(nvidia_root, entry, "bin", "x86_64")
                if os.path.isdir(consolidated_bin):
                    os.add_dll_directory(consolidated_bin)
        for pkg in ("cuda_runtime", "curand", "cublas", "cuda_nvrtc", "cudnn"):
            if pkg == "cudnn" and torch_already_loaded:
                continue
            bin_dir = os.path.join(nvidia_root, pkg, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                if pkg == "cudnn":
                    if bin_dir not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

    cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_home and os.path.isdir(os.path.join(cuda_home, "bin")):
        os.add_dll_directory(os.path.join(cuda_home, "bin"))

    if not torch_already_loaded:
        nvidia_cudnn_root = r"C:\Program Files\NVIDIA\CUDNN"
        if os.path.isdir(nvidia_cudnn_root):
            for entry in os.listdir(nvidia_cudnn_root):
                bin_base = os.path.join(nvidia_cudnn_root, entry, "bin")
                if os.path.isdir(bin_base):
                    for sub in os.listdir(bin_base):
                        full = os.path.join(bin_base, sub)
                        if os.path.isdir(full):
                            os.add_dll_directory(full)


def _add_windows_openblas_dll_dir():
    if sys.platform != "win32":
        return
    bundled_bin = os.path.join(os.path.dirname(__file__), "_openblas_bin")
    if os.path.isdir(bundled_bin):
        os.add_dll_directory(bundled_bin)
        return
    dev_bin = os.environ.get("AAKAAR_OPENBLAS_BIN", r"C:\openblas-prebuilt\bin")
    if os.path.isdir(dev_bin):
        os.add_dll_directory(dev_bin)


def _preload_linux_cuda_libs():
    if sys.platform != "linux":
        return
    import ctypes, glob, site
    failures = []
    for root in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_root = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for pkg in ("cuda_runtime", "cublas", "cudnn", "curand"):
            lib_dir = os.path.join(nvidia_root, pkg, "lib")
            if not os.path.isdir(lib_dir):
                continue
            for so_path in sorted(glob.glob(os.path.join(lib_dir, "lib*.so*"))):
                try:
                    ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
                except OSError as e:
                    failures.append((so_path, str(e)))
    if failures:
        import warnings
        warnings.warn(
            f"aakaar: {len(failures)} CUDA library file(s) failed to preload: "
            f"{[f[0] for f in failures]}. GPU features may not work. "
            f"First error: {failures[0][1]}"
        )


if sys.platform == "win32" and "torch" not in sys.modules:
    import warnings
    warnings.warn(
        "aakaar is being imported before torch in this process. If you plan to "
        "use both aakaar and torch together, import torch first to avoid a known "
        "Windows cuDNN DLL-loading conflict.",
        stacklevel=2,
    )

_add_windows_cuda_dll_dirs()
_add_windows_openblas_dll_dir()
_preload_linux_cuda_libs()

from . import _C
from .data import Dataset, TensorDataset, DataLoader
import numpy as np

def _normalize_shape(size):
    if isinstance(size, int):
        return [size]
    return list(size)

def rand(size, device: str = "cpu", seed: int = 42, requires_grad: bool = False, dtype: str = "float32"):
    if device == "cuda" and not _C.HAS_CUDA:
        raise RuntimeError(
            "This installation of aakaar was built without CUDA support. "
            "Install on a machine with the CUDA toolkit available, or use device='cpu'."
        )
    shape = _normalize_shape(size)
    dt = getattr(_C.DType, dtype)
    t = _C.Tensor(shape, device, dt)
    if device == "cuda":
        _C.generate_random(t, seed)
    else:
        _C.fill_cpu_random(t, seed)
    t.requires_grad = requires_grad
    return t

def randint(size, low, high, device: str = "cpu", seed: int = 42, dtype: str = "int32"):
    if device == "cuda" and not _C.HAS_CUDA:
        raise RuntimeError(
            "This installation of aakaar was built without CUDA support. "
            "Install on a machine with the CUDA toolkit available, or use device='cpu'."
        )
    shape = _normalize_shape(size)
    dt = getattr(_C.DType, dtype)
    t = _C.Tensor(shape, device, dt)
    if device == "cuda":
        _C.generate_randint(t, low, high, seed)
    else:
        _C.fill_cpu_randint(t, low, high, seed)
    return t

def matmul(a, b):
    if a.device != b.device:
        raise ValueError(f"Tensors on different devices: {a.device} vs {b.device}")
    if a.device == "cuda":
        if not _C.HAS_CUDA:
            raise RuntimeError("This build of aakaar has no CUDA support.")
        return _C.cuda_matmul(a, b)
    else:
        return _C.cpu_matmul(a, b)

def add(a, b):
    if a.device != b.device:
        raise ValueError(f"Tensors on different devices: {a.device} vs {b.device}")
    if a.device == "cuda":
        if not _C.HAS_CUDA:
            raise RuntimeError("This build of aakaar has no CUDA support.")
        return _C.cuda_add(a, b)
    else:
        return _C.cpu_add(a, b)

def sub(a, b):
    if a.device != b.device:
        raise ValueError(f"Tensors on different devices: {a.device} vs {b.device}")
    if a.device == "cuda":
        if not _C.HAS_CUDA:
            raise RuntimeError("This build of aakaar has no CUDA support.")
        return _C.cuda_sub(a, b)
    else:
        return _C.cpu_sub(a, b)

def is_available():
    """Returns True if a CUDA-capable GPU is actually present and usable right now."""
    return _C.is_available()

def device_count():
    """Number of CUDA-capable GPUs detected on this machine."""
    return _C.device_count()

class _CudaNamespace:
    is_available = staticmethod(is_available)
    device_count = staticmethod(device_count)

cuda = _CudaNamespace()

@contextlib.contextmanager
def no_grad():
    """Disables gradient tracking for the duration of the with-block.
    Use this when updating parameters so the update itself isn't tracked:
        with aakaar.no_grad():
            w = w - lr * w.grad
    """
    prev = _C._is_grad_enabled()
    _C._set_grad_enabled(False)
    try:
        yield
    finally:
        _C._set_grad_enabled(prev)

def is_grad_enabled():
    return _C._is_grad_enabled()

# In __init__.py — no new C++ needed at all
def mse_loss(pred, target):
    diff = pred - target
    return (diff * diff).sum() / pred.size

def from_numpy(array, device: str = "cpu", requires_grad: bool = False):
    if device == "cuda" and not _C.HAS_CUDA:
        raise RuntimeError(
            "This installation of aakaar was built without CUDA support. "
            "Install on a machine with the CUDA toolkit available, or use device='cpu'."
        )
    return _C.from_numpy(array, device, requires_grad)

def zero_grad_all(parameters):
    """Zero the .grad of every tensor in an iterable, e.g. a list of parameters."""
    for p in parameters:
        p.zero_grad()
def softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True)
    shifted = x - m
    exp_x = shifted.exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def cross_entropy_from_probs(probs, target_onehot):
    eps = 1e-7
    log_probs = (probs + eps).log()
    return -(target_onehot * log_probs).sum() / probs.shape[0]

def set_tf32(enabled: bool = True):
    """Enable/disable TF32 tensor-core acceleration for matmul on CUDA (Ampere+ GPUs).
    Trades a small amount of precision (~10-bit mantissa vs FP32's 23-bit during the
    internal multiply-accumulate) for a significant speedup. Values remain float32
    in memory; only the matmul computation path changes. No effect on CPU or on
    GPUs older than Ampere (sm_80)."""
    if not _C.HAS_CUDA:
        return
    _C._set_tf32_enabled(enabled)

def is_tf32_enabled():
    return _C._get_tf32_enabled() if _C.HAS_CUDA else False

def synchronize():
    """Blocks until all queued CUDA operations complete. Call this before
    timing GPU code, since aakaar's CUDA ops run asynchronously by default —
    without this, a timer will only measure kernel-launch/dispatch time,
    not actual GPU execution time."""
    if _C.HAS_CUDA:
        _C._synchronize()

def softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True)
    shifted = x - m
    exp_x = shifted.exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def log_softmax(x, dim=-1):
    """Numerically stable log(softmax(x)) — computes log-sum-exp using the
    max-subtraction trick without materializing raw softmax probabilities
    first (avoids exponentiating then re-logging, which loses precision)."""
    m = x.max(dim=dim, keepdim=True)
    shifted = x - m
    log_sum_exp = shifted.exp().sum(dim=dim, keepdim=True).log()
    return shifted - log_sum_exp

def im2col_1d(x, kernel_size, stride, padding, dilation, out_length):
    """Unfolds a (B, C, L) tensor into (B, C*kernel_size, L_out) sliding windows.
    Internal primitive used by nn.Conv1d — not usually called directly."""
    return _C.im2col_1d(x, kernel_size, stride, padding, dilation, out_length)
def conv1d(x, weight, stride=1, padding=0, dilation=1):
    """Fastest available Conv1d forward+backward. Routes to cuDNN for
    float32 CUDA tensors when this build was compiled with cuDNN support;
    falls back to im2col + matmul (cuBLAS/OpenBLAS) otherwise — for CPU
    tensors, float64/int32/int64 dtypes, or builds where cuDNN wasn't
    found at compile time."""
    use_cudnn = (getattr(_C, "HAS_CUDNN", False) and x.device == "cuda" and weight.device == "cuda"
                 and x.dtype == "float32" and weight.dtype == "float32")
    if use_cudnn:
        return _C.conv1d_cudnn(x, weight, stride, padding, dilation)

    C_out, C_in, K = weight.shape
    B, _, L_in = x.shape
    L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    if L_out <= 0:
        raise ValueError(f"conv1d: computed output length {L_out} <= 0")
    col = im2col_1d(x, K, stride, padding, dilation, L_out)
    w_flat = weight.reshape([C_out, C_in * K]).reshape([1, C_out, C_in * K])
    return matmul(w_flat, col)

def set_cudnn_tf32(enabled: bool = True):
    """Enable/disable TF32 tensor-core math for cuDNN convolutions (Ampere+).
    Matches torch's torch.backends.cudnn.allow_tf32, which defaults to True
    on supported hardware. NOTE: on some cuDNN 9.x / driver / GPU combinations
    (confirmed on this project: cuDNN 9.2.0 + driver 610.74 + RTX 4060),
    TF32 mode can cause cudnnFindConvolutionForwardAlgorithmEx to report
    every candidate algorithm as failed, even though the GPU itself supports
    TF32. Root cause not fully isolated. aakaar defaults TF32 OFF for
    reliability; call set_cudnn_tf32(True) explicitly to opt in and test
    on your own hardware first."""
    if _C.HAS_CUDA:
        _C._set_cudnn_tf32_enabled(enabled)

def is_cudnn_tf32_enabled():
    return _C._get_cudnn_tf32_enabled() if _C.HAS_CUDA else False

def im2col_2d(x, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW):
    """Unfolds a (B, C, H, W) tensor into (B, C*KH*KW, OH*OW) sliding windows.
    Internal primitive used by nn.Conv2d — not usually called directly."""
    return _C.im2col_2d(x, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW)

def conv2d(x, weight, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    """Fastest available Conv2d forward+backward. Routes to cuDNN for
    float32 CUDA tensors when this build has cuDNN support; falls back to
    im2col + matmul (cuBLAS/OpenBLAS) otherwise."""
    SH, SW = (stride, stride) if isinstance(stride, int) else stride
    PH, PW = (padding, padding) if isinstance(padding, int) else padding
    DH, DW = (dilation, dilation) if isinstance(dilation, int) else dilation

    use_cudnn = (getattr(_C, "HAS_CUDNN", False) and x.device == "cuda" and weight.device == "cuda"
                 and x.dtype == "float32" and weight.dtype == "float32")
    if use_cudnn:
        return _C.conv2d_cudnn(x, weight, SH, SW, PH, PW, DH, DW)

    C_out, C_in, KH, KW = weight.shape
    B, _, H, W = x.shape
    OH = (H + 2 * PH - DH * (KH - 1) - 1) // SH + 1
    OW = (W + 2 * PW - DW * (KW - 1) - 1) // SW + 1
    if OH <= 0 or OW <= 0:
        raise ValueError(f"conv2d: computed output size ({OH},{OW}) <= 0")
    col = im2col_2d(x, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW)
    w_flat = weight.reshape([C_out, C_in * KH * KW]).reshape([1, C_out, C_in * KH * KW])
    out = matmul(w_flat, col)  # (B, C_out, OH*OW)
    return out.reshape([B, C_out, OH, OW])

def im2col_3d(x, kernel_size, stride, padding, dilation, out_size):
    """Unfolds a (B, C, D, H, W) tensor into (B, C*KD*KH*KW, OD*OH*OW)
    sliding windows. Internal primitive used by nn.Conv3d — not usually
    called directly."""
    KD, KH, KW = kernel_size
    SD, SH, SW = stride
    PD, PH, PW = padding
    DD, DH, DW = dilation
    OD, OH, OW = out_size
    return _C.im2col_3d(x, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW)

def conv3d(x, weight, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    """Fastest available Conv3d forward+backward. Routes to cuDNN for
    float32 CUDA tensors when this build was compiled with cuDNN support;
    falls back to im2col + matmul (cuBLAS/OpenBLAS) otherwise."""
    SD, SH, SW = (stride, stride, stride) if isinstance(stride, int) else stride
    PD, PH, PW = (padding, padding, padding) if isinstance(padding, int) else padding
    DD, DH, DW = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation

    use_cudnn = (getattr(_C, "HAS_CUDNN", False) and x.device == "cuda" and weight.device == "cuda"
                 and x.dtype == "float32" and weight.dtype == "float32")
    if use_cudnn:
        return _C.conv3d_cudnn(x, weight, SD, SH, SW, PD, PH, PW, DD, DH, DW)

    C_out, C_in, KD, KH, KW = weight.shape
    B, _, D, H, W = x.shape
    OD = (D + 2 * PD - DD * (KD - 1) - 1) // SD + 1
    OH = (H + 2 * PH - DH * (KH - 1) - 1) // SH + 1
    OW = (W + 2 * PW - DW * (KW - 1) - 1) // SW + 1
    if OD <= 0 or OH <= 0 or OW <= 0:
        raise ValueError(f"conv3d: computed output size ({OD},{OH},{OW}) <= 0")
    col = im2col_3d(x, (KD, KH, KW), (SD, SH, SW), (PD, PH, PW), (DD, DH, DW), (OD, OH, OW))
    w_flat = weight.reshape([C_out, C_in * KD * KH * KW]).reshape([1, C_out, C_in * KD * KH * KW])
    out = matmul(w_flat, col)  # (B, C_out, OD*OH*OW)
    return out.reshape([B, C_out, OD, OH, OW])