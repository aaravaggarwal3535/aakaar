import os
import sys
import glob

def _add_windows_cuda_dll_dirs():
    """On Windows, point the DLL loader at the CUDA runtime DLLs shipped
    inside the nvidia-*-cu12 pip packages, since Windows won't find them
    on its own the way Linux does via RPATH."""
    if sys.platform != "win32":
        return
    for pkg in ("nvidia/cuda_runtime", "nvidia/curand"):
        pattern = os.path.join(sys.prefix, "Lib", "site-packages", pkg, "bin")
        if os.path.isdir(pattern):
            os.add_dll_directory(pattern)
        # also check user-site installs (pip install --user, common on Windows)
        user_pattern = os.path.join(os.path.expanduser("~"), "AppData", "Roaming",
                                     "Python", f"Python{sys.version_info.major}{sys.version_info.minor}",
                                     "site-packages", pkg, "bin")
        if os.path.isdir(user_pattern):
            os.add_dll_directory(user_pattern)

_add_windows_cuda_dll_dirs()

from . import _C
import numpy as np

def rand(size: int, device: str = "cpu", seed: int = 42):
    if device == "cuda" and not _C.HAS_CUDA:
        raise RuntimeError(
            "This installation of aakaar was built without CUDA support. "
            "Install on a machine with the CUDA toolkit available, or use device='cpu'."
        )
    t = _C.Tensor(size, device)
    if device == "cuda":
        _C.generate_random(t, seed)
    else:
        _C.fill_cpu_random(t, seed)
    return t  # stays an aakaar Tensor — no numpy conversion here