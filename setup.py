import os
import sys
import subprocess
import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


class CUDABuildExtension(build_ext):
    def build_extensions(self):
        is_windows = sys.platform == "win32"

        # Locate CUDA include dir cross-platform (don't hardcode /usr/local/cuda)
        cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME") or "/usr/local/cuda"
        cuda_include = os.path.join(cuda_home, "include")
        cuda_lib = os.path.join(cuda_home, "lib", "x64") if is_windows else os.path.join(cuda_home, "lib64")

        nvcc_flags = ["-O3",
              "-gencode=arch=compute_75,code=sm_75",
              "-gencode=arch=compute_86,code=sm_86",
              "-gencode=arch=compute_89,code=sm_89"]

        if not is_windows:
            nvcc_flags.append("-Xcompiler=-fPIC")
        else:
            nvcc_flags.append("-Xcompiler=/MD")   # <-- ADD THIS: match Python's dynamic CRT

        for ext in self.extensions:
            includes = [f"-I{d}" for d in ext.include_dirs]
            includes.append(f"-I{sysconfig.get_path('include')}")
            includes.append(f"-I{cuda_include}")

            cu_sources = [s for s in ext.sources if s.endswith(".cu")]
            cpp_sources = [s for s in ext.sources if s.endswith(".cpp")]

            objects = []
            for cu_file in cu_sources:
                obj_ext = ".obj" if is_windows else ".o"
                obj_file = cu_file.replace(".cu", obj_ext)
                nvcc_cmd = ["nvcc", "-c", cu_file, "-o", obj_file] + nvcc_flags + includes
                print(f"Compiling CUDA: {' '.join(nvcc_cmd)}")
                subprocess.check_call(nvcc_cmd)
                objects.append(obj_file)

            ext.sources = cpp_sources
            ext.extra_objects = objects
            ext.include_dirs.append(cuda_include)      # <-- ADD THIS LINE
            ext.library_dirs.append(cuda_lib)

        super().build_extensions()


aakaar_ext = Extension(
    "aakaar._C",
    sources=["src/bindings.cpp", "src/cpu_kernel.cpp", "src/random_kernel.cu"],
    include_dirs=[pybind11.get_include()],
    libraries=["curand", "cudart"],
    language="c++",
)

setup(
    name="aakaar",
    version="0.1.3",
    author="Aarav Aggarwal",
    description="A custom standalone ML library featuring CUDA-accelerated operations.",
    packages=["aakaar"],
    ext_modules=[aakaar_ext],
    cmdclass={"build_ext": CUDABuildExtension},
    install_requires=["numpy", "nvidia-curand-cu12", "nvidia-cuda-runtime-cu12"],
    setup_requires=["pybind11"],
)