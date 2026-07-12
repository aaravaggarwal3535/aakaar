import os
import sys
import contextlib

__version__ = "0.1.13"

def _add_windows_cuda_dll_dirs():
    if sys.platform != "win32":
        return
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
        for pkg in ("cuda_runtime", "curand", "cublas", "cuda_nvrtc"):
            bin_dir = os.path.join(nvidia_root, pkg, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)

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

_add_windows_cuda_dll_dirs()
_add_windows_openblas_dll_dir()

from . import _C
import numpy as np

def _normalize_shape(size):
    if isinstance(size, int):
        return [size]
    return list(size)

def rand(size, device: str = "cpu", seed: int = 42, requires_grad: bool = False):
    if device == "cuda" and not _C.HAS_CUDA:
        raise RuntimeError(
            "This installation of aakaar was built without CUDA support. "
            "Install on a machine with the CUDA toolkit available, or use device='cpu'."
        )
    shape = _normalize_shape(size)
    t = _C.Tensor(shape, device)
    if device == "cuda":
        _C.generate_random(t, seed)
    else:
        _C.fill_cpu_random(t, seed)
    t.requires_grad = requires_grad
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

