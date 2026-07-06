import os
import sys

__version__ = "0.1.8"

def _add_windows_cuda_dll_dirs():
    if sys.platform != "win32":
        return
    import site
    search_roots = site.getsitepackages() + [site.getusersitepackages()]
    for root in search_roots:
        nvidia_root = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue

        # New unversioned CUDA-13+ packages: consolidated nvidia/cuXX/bin/x86_64/
        for entry in os.listdir(nvidia_root):
            if entry.startswith("cu") and entry[2:].isdigit():
                consolidated_bin = os.path.join(nvidia_root, entry, "bin", "x86_64")
                if os.path.isdir(consolidated_bin):
                    os.add_dll_directory(consolidated_bin)

        # Old per-package CUDA-12 style: nvidia/<pkg>/bin/
        for pkg in ("cuda_runtime", "curand", "cublas", "cuda_nvrtc"):
            bin_dir = os.path.join(nvidia_root, pkg, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)

_add_windows_cuda_dll_dirs()

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