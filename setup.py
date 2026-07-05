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

        if CUDA_AVAILABLE:
            cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME") or "/usr/local/cuda"
            cuda_include = os.path.join(cuda_home, "include")
            cuda_lib = os.path.join(cuda_home, "lib", "x64") if is_windows else os.path.join(cuda_home, "lib64")

            nvcc_flags = ["-O3",
                          "-std=c++17",
                          "-gencode=arch=compute_75,code=sm_75",
                          "-gencode=arch=compute_86,code=sm_86",
                          "-gencode=arch=compute_89,code=sm_89"]
            nvcc_flags.append("-Xcompiler=/MD" if is_windows else "-Xcompiler=-fPIC")

        for ext in self.extensions:
            includes = [f"-I{d}" for d in ext.include_dirs]
            includes.append(f"-I{sysconfig.get_path('include')}")

            cu_sources = [s for s in ext.sources if s.endswith(".cu")]
            cpp_sources = [s for s in ext.sources if s.endswith(".cpp")]

            objects = []
            if CUDA_AVAILABLE:
                includes.append(f"-I{cuda_include}")
                for cu_file in cu_sources:
                    obj_ext = ".obj" if is_windows else ".o"
                    obj_file = cu_file.replace(".cu", obj_ext)
                    nvcc_cmd = ["nvcc", "-c", cu_file, "-o", obj_file] + nvcc_flags + includes
                    print(f"Compiling CUDA: {' '.join(nvcc_cmd)}")
                    subprocess.check_call(nvcc_cmd)
                    objects.append(obj_file)
                ext.include_dirs.append(cuda_include)
                ext.library_dirs.append(cuda_lib)
                ext.libraries.extend(["curand", "cudart", "cublas"])
            else:
                # No CUDA toolkit: skip .cu sources entirely, CPU-only build
                print("No nvcc found — building CPU-only extension (no CUDA support).")
                ext.define_macros.append(("AAKAAR_NO_CUDA", "1"))

            ext.sources = cpp_sources
            ext.extra_objects = objects

        super().build_extensions()

host_compiler_flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]
aakaar_ext = Extension(
    "aakaar._C",
    sources=["src/bindings.cpp", "src/cpu_kernel.cpp", "src/random_kernel.cu", 'src/matmul_kernel.cu', 'src/elementwise_kernel.cu'],
    include_dirs=[pybind11.get_include()],
    libraries=[],  # populated conditionally above
    language="c++",
    extra_compile_args=host_compiler_flags
)

setup(
    name="aakaar",
    version="0.1.7",
    author="Aarav Aggarwal",
    description="A custom standalone ML library featuring CUDA-accelerated operations (CPU fallback supported).",
    packages=["aakaar"],
    ext_modules=[aakaar_ext],
    cmdclass={"build_ext": CUDABuildExtension},
    install_requires=["numpy"],  # nvidia-* deps now conditional, see below
    setup_requires=["pybind11"],
)