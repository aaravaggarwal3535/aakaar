import os
import sys
import contextlib

__version__ = "0.2.1"

def _add_windows_cuda_dll_dirs():
    if sys.platform != "win32":
        return
    import site
    search_roots = site.getsitepackages() + [site.getusersitepackages()]
    found_any = False
    for root in search_roots:
        nvidia_root = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for entry in os.listdir(nvidia_root):
            if entry.startswith("cu") and entry[2:].isdigit():
                consolidated_bin = os.path.join(nvidia_root, entry, "bin", "x86_64")
                if os.path.isdir(consolidated_bin):
                    os.add_dll_directory(consolidated_bin)
                    found_any = True
        for pkg in ("cuda_runtime", "curand", "cublas", "cuda_nvrtc", "cudnn"):
            bin_dir = os.path.join(nvidia_root, pkg, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                found_any = True
    # Also check the system CUDA toolkit / standalone cuDNN install, in case
    # the user installed those instead of the pip packages.
    cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_home and os.path.isdir(os.path.join(cuda_home, "bin")):
        os.add_dll_directory(os.path.join(cuda_home, "bin"))
        found_any = True
    nvidia_cudnn_root = r"C:\Program Files\NVIDIA\CUDNN"
    if os.path.isdir(nvidia_cudnn_root):
        for entry in os.listdir(nvidia_cudnn_root):
            bin_base = os.path.join(nvidia_cudnn_root, entry, "bin")
            if os.path.isdir(bin_base):
                for sub in os.listdir(bin_base):
                    full = os.path.join(bin_base, sub)
                    if os.path.isdir(full):
                        os.add_dll_directory(full)
                        found_any = True
    return found_any

def _add_windows_openblas_dll_dir():
    if sys.platform != "win32":
        return
    bundled_bin = os.path.join(os.path.dirname(__file__), "_openblas_bin")
    if os.path.isdir(bundled_bin):
        os.add_dll_directory(bundled_bin)
        return
    # Fallback for local development before the DLL is bundled/reinstalled
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
    Matches torch's torch.backends.cudnn.allow_tf32, which defaults to True.
    Trades a small amount of precision for significant speed on conv ops —
    same trade torch makes by default. Set to False if you need exact
    float32 precision and can accept the speed cost."""
    if _C.HAS_CUDA:
        _C._set_cudnn_tf32_enabled(enabled)

def is_cudnn_tf32_enabled():
    return _C._get_cudnn_tf32_enabled() if _C.HAS_CUDA else False