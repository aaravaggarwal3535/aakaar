import os
import sys
import shutil
import subprocess
import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

def has_nvcc():
    return shutil.which("nvcc") is not None

CUDA_AVAILABLE = has_nvcc()
print(f"CUDA toolkit detected: {CUDA_AVAILABLE}")


class CUDABuildExtension(build_ext):
    def build_extensions(self):
        is_windows = sys.platform == "win32"

        if is_windows:
            openblas_root = os.environ.get("OPENBLAS_ROOT", r"C:\openblas-prebuilt")
            openblas_include = os.path.join(openblas_root, "include")
            openblas_lib_dir = os.path.join(openblas_root, "lib")
            if os.path.isfile(os.path.join(openblas_include, "cblas.h")):
                for ext in self.extensions:
                    ext.include_dirs.append(openblas_include)
                    ext.library_dirs.append(openblas_lib_dir)
                    ext.libraries.append("libopenblas")
            else:
                print("WARNING: OpenBLAS cblas.h not found at", openblas_include)
        else:
            # Linux: openblas-devel (installed via CIBW_BEFORE_ALL_LINUX) puts
            # cblas.h somewhere on the system include path — location varies
            # by distro packaging, so check several real candidates rather
            # than assuming one.
            candidate_includes = [
                "/usr/include/x86_64-linux-gnu",  # Debian/Ubuntu multiarch layout
                "/usr/include/openblas",
                "/usr/include",
                "/usr/local/include/openblas",
                "/usr/local/include",
            ]
            found_include = None
            for cand in candidate_includes:
                if os.path.isfile(os.path.join(cand, "cblas.h")):
                    found_include = cand
                    break

            # Belt-and-suspenders: if still not found, ask the package manager
            # directly where it put the header, rather than guessing further.
            if not found_include:
                for pkg_cmd in (["dpkg", "-L", "libopenblas-dev"], ["rpm", "-ql", "openblas-devel"]):
                    try:
                        out = subprocess.check_output(pkg_cmd, text=True)
                        for line in out.splitlines():
                            if line.endswith("cblas.h"):
                                found_include = os.path.dirname(line)
                                break
                        if found_include:
                            break
                    except Exception:
                        pass

            if found_include:
                for ext in self.extensions:
                    if found_include not in ext.include_dirs:
                        ext.include_dirs.append(found_include)
                    ext.libraries.append("openblas")
                for libdir in ("/usr/lib64", "/usr/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu"):
                    if os.path.isdir(libdir):
                        for ext in self.extensions:
                            if libdir not in ext.library_dirs:
                                ext.library_dirs.append(libdir)
            else:
                print("WARNING: OpenBLAS cblas.h not found on Linux — build will fail to link matmul against BLAS.")

            # dlopen/dlsym support, needed on Linux regardless of whether
            # OpenBLAS was found via the block above.
            for ext in self.extensions:
                ext.libraries.append("dl")

        if CUDA_AVAILABLE:
            cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME") or "/usr/local/cuda"
            cuda_include = os.path.join(cuda_home,

host_compiler_flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]
aakaar_ext = Extension(
    "aakaar._C",
    sources=["src/bindings.cpp", "src/cpu_kernel.cpp", "src/random_kernel.cu", 'src/matmul_kernel.cu', 'src/elementwise_kernel.cu', 'src/reduction_kernel.cu', 'src/strided_copy_kernel.cu'],
    include_dirs=[
    pybind11.get_include(),
    "src",
],
    libraries=[],  # populated conditionally above
    language="c++",
    extra_compile_args=host_compiler_flags
)

setup(
    name="aakaar",
    version="0.1.11",
    author="Aarav Aggarwal",
    description="A custom standalone ML library featuring CUDA-accelerated operations (CPU fallback supported).",
    packages=["aakaar", "aakaar._openblas_bin"],
    package_data={"aakaar": ["_openblas_bin/*.dll"]},
    include_package_data=True,
    ext_modules=[aakaar_ext],
    cmdclass={"build_ext": CUDABuildExtension},
    install_requires=["numpy"],  # nvidia-* deps now conditional, see below
    setup_requires=["pybind11"],
)