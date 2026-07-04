import os
import sys

def _add_windows_cuda_dll_dirs():
    if sys.platform != "win32":
        return
    import site
    search_roots = site.getsitepackages() + [site.getusersitepackages()]
    for root in search_roots:
        for pkg in ("nvidia/cuda_runtime", "nvidia/curand"):
            bin_dir = os.path.join(root, *pkg.split("/"), "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)

_add_windows_cuda_dll_dirs()

from . import _C
import numpy as np

def _normalize_shape(size):
    if isinstance(size, int):
        return [size]
    return list(size)

def rand(size, device: str = "cpu", seed: int = 42):
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
    return t