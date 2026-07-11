# import os
# import sys
# import shutil
# import subprocess
# import sysconfig
# from setuptools import setup, Extension
# from setuptools.command.build_ext import build_ext
# import pybind11

# def has_nvcc():
#     return shutil.which("nvcc") is not None

# CUDA_AVAILABLE = has_nvcc()
# print(f"CUDA toolkit detected: {CUDA_AVAILABLE}")


# class CUDABuildExtension(build_ext):
#     def build_extensions(self):
#         is_windows = sys.platform == "win32"

#         if is_windows:
#             openblas_root = os.environ.get("OPENBLAS_ROOT", r"C:\openblas-prebuilt")
#             openblas_include = os.path.join(openblas_root, "include")
#             openblas_lib_dir = os.path.join(openblas_root, "lib")
#             if os.path.isfile(os.path.join(openblas_include, "cblas.h")):
#                 for ext in self.extensions:
#                     ext.include_dirs.append(openblas_include)
#                     ext.library_dirs.append(openblas_lib_dir)
#                     ext.libraries.append("libopenblas")
#             else:
#                 print("WARNING: OpenBLAS cblas.h not found at", openblas_include)
#         else:
#             # Linux: openblas-devel (installed via CIBW_BEFORE_ALL_LINUX) puts
#             # cblas.h somewhere on the system include path — location varies
#             # by distro packaging, so check several real candidates rather
#             # than assuming one.
#             candidate_includes = [
#                 "/usr/include/x86_64-linux-gnu",  # Debian/Ubuntu multiarch layout
#                 "/usr/include/openblas",
#                 "/usr/include",
#                 "/usr/local/include/openblas",
#                 "/usr/local/include",
#             ]
#             found_include = None
#             for cand in candidate_includes:
#                 if os.path.isfile(os.path.join(cand, "cblas.h")):
#                     found_include = cand
#                     break

#             # Belt-and-suspenders: if still not found, ask the package manager
#             # directly where it put the header, rather than guessing further.
#             if not found_include:
#                 for pkg_cmd in (["dpkg", "-L", "libopenblas-dev"], ["rpm", "-ql", "openblas-devel"]):
#                     try:
#                         out = subprocess.check_output(pkg_cmd, text=True)
#                         for line in out.splitlines():
#                             if line.endswith("cblas.h"):
#                                 found_include = os.path.dirname(line)
#                                 break
#                         if found_include:
#                             break
#                     except Exception:
#                         pass

#             if found_include:
#                 for ext in self.extensions:
#                     if found_include not in ext.include_dirs:
#                         ext.include_dirs.append(found_include)
#                     ext.libraries.append("openblas")
#                 for libdir in ("/usr/lib64", "/usr/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu"):
#                     if os.path.isdir(libdir):
#                         for ext in self.extensions:
#                             if libdir not in ext.library_dirs:
#                                 ext.library_dirs.append(libdir)
#             else:
#                 print("WARNING: OpenBLAS cblas.h not found on Linux — build will fail to link matmul against BLAS.")

#             # dlopen/dlsym support, needed on Linux regardless of whether
#             # OpenBLAS was found via the block above.
#             for ext in self.extensions:
#                 ext.libraries.append("dl")

#         if CUDA_AVAILABLE:
#             cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME") or "/usr/local/cuda"
#             cuda_include = os.path.join(cuda_home, "include")
#             cuda_lib = os.path.join(cuda_home, "lib", "x64") if is_windows else os.path.join(cuda_home, "lib64")

#             nvcc_flags = [
#                 "-O3",
#                 "-std=c++17",
#                 "-gencode=arch=compute_75,code=sm_75",
#                 "-gencode=arch=compute_86,code=sm_86",
#                 "-gencode=arch=compute_89,code=sm_89",
#             ]

#             # Point nvcc at GCC 13 explicitly — CUDA 12.4 does not support
#             # GCC 14 (the manylinux_2_28 default), and -allow-unsupported-compiler
#             # was tried and rejected: it produced genuine compile errors when
#             # nvcc actually parsed GCC 14's headers.
#             gcc13_path = "/opt/rh/gcc-toolset-13/root/usr/bin/g++"
#             if not is_windows and os.path.isfile(gcc13_path):
#                 nvcc_flags.append(f"-ccbin={gcc13_path}")

#             nvcc_flags.append("-Xcompiler=/MD" if is_windows else "-Xcompiler=-fPIC")

#         for ext in self.extensions:
#             includes = [f"-I{d}" for d in ext.include_dirs]
#             includes.append(f"-I{sysconfig.get_path('include')}")

#             cu_sources = [s for s in ext.sources if s.endswith(".cu")]
#             cpp_sources = [s for s in ext.sources if s.endswith(".cpp")]

#             objects = []
#             if CUDA_AVAILABLE:
#                 includes.append(f"-I{cuda_include}")
#                 for cu_file in cu_sources:
#                     obj_ext = ".obj" if is_windows else ".o"
#                     obj_file = cu_file.replace(".cu", obj_ext)
#                     nvcc_cmd = ["nvcc", "-c", cu_file, "-o", obj_file] + nvcc_flags + includes
#                     print(f"Compiling CUDA: {' '.join(nvcc_cmd)}")
#                     subprocess.check_call(nvcc_cmd)
#                     objects.append(obj_file)
#                 ext.include_dirs.append(cuda_include)
#                 ext.library_dirs.append(cuda_lib)
#                 ext.libraries.extend(["curand", "cudart", "cublas"])
#             else:
#                 print("No nvcc found — building CPU-only extension (no CUDA support).")
#                 ext.define_macros.append(("AAKAAR_NO_CUDA", "1"))

#             ext.sources = cpp_sources
#             ext.extra_objects = objects

#         super().build_extensions()


# host_compiler_flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]
# aakaar_ext = Extension(
#     "aakaar._C",
#     sources=[
#         "src/bindings.cpp",
#         "src/cpu_kernel.cpp",
#         "src/random_kernel.cu",
#         "src/matmul_kernel.cu",
#         "src/elementwise_kernel.cu",
#         "src/reduction_kernel.cu",
#         "src/strided_copy_kernel.cu",
#     ],
#     include_dirs=[
#         pybind11.get_include(),
#         "src",
#     ],
#     libraries=[],  # populated conditionally above
#     language="c++",
#     extra_compile_args=host_compiler_flags,
# )

# setup(
#     name="aakaar",
#     version="0.1.11",
#     author="Aarav Aggarwal",
#     description="A custom standalone ML library featuring CUDA-accelerated operations (CPU fallback supported).",
#     packages=["aakaar", "aakaar._openblas_bin"],
#     package_data={"aakaar": ["_openblas_bin/*.dll"]},
#     include_package_data=True,
#     ext_modules=[aakaar_ext],
#     cmdclass={"build_ext": CUDABuildExtension},
#     install_requires=["numpy"],
#     setup_requires=["pybind11"],
# )

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
        is_macos = sys.platform == "darwin"

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
        elif is_macos:
            # Homebrew installs OpenBLAS "keg-only" (not symlinked into /usr/local
            # or /opt/homebrew directly) to avoid clashing with Apple's own
            # Accelerate BLAS — so cblas.h/lib aren't on the default search path
            # and must be added explicitly.
            brew_prefixes = ["/opt/homebrew/opt/openblas", "/usr/local/opt/openblas"]
            found = None
            for prefix in brew_prefixes:
                if os.path.isfile(os.path.join(prefix, "include", "cblas.h")):
                    found = prefix
                    break
            if found:
                for ext in self.extensions:
                    ext.include_dirs.append(os.path.join(found, "include"))
                    ext.library_dirs.append(os.path.join(found, "lib"))
                    ext.libraries.append("openblas")
            else:
                print("WARNING: OpenBLAS not found via Homebrew at", brew_prefixes,
                      "— install with `brew install openblas`.")
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
            cuda_include = os.path.join(cuda_home, "include")
            cuda_lib = os.path.join(cuda_home, "lib", "x64") if is_windows else os.path.join(cuda_home, "lib64")

            nvcc_flags = [
                "-O3",
                "-std=c++17",
                "-gencode=arch=compute_75,code=sm_75",
                "-gencode=arch=compute_86,code=sm_86",
                "-gencode=arch=compute_89,code=sm_89",
            ]

            # Point nvcc at GCC 13 explicitly — CUDA 12.4 does not support
            # GCC 14 (the manylinux_2_28 default), and -allow-unsupported-compiler
            # was tried and rejected: it produced genuine compile errors when
            # nvcc actually parsed GCC 14's headers.
            gcc13_path = "/opt/rh/gcc-toolset-13/root/usr/bin/g++"
            if not is_windows and os.path.isfile(gcc13_path):
                nvcc_flags.append(f"-ccbin={gcc13_path}")

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
                print("No nvcc found — building CPU-only extension (no CUDA support).")
                ext.define_macros.append(("AAKAAR_NO_CUDA", "1"))

            ext.sources = cpp_sources
            ext.extra_objects = objects

        super().build_extensions()


host_compiler_flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]
aakaar_ext = Extension(
    "aakaar._C",
    sources=[
        "src/bindings.cpp",
        "src/cpu_kernel.cpp",
        "src/random_kernel.cu",
        "src/matmul_kernel.cu",
        "src/elementwise_kernel.cu",
        "src/reduction_kernel.cu",
        "src/strided_copy_kernel.cu",
    ],
    include_dirs=[
        pybind11.get_include(),
        "src",
    ],
    libraries=[],  # populated conditionally above
    language="c++",
    extra_compile_args=host_compiler_flags,
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
    install_requires=["numpy"],
    setup_requires=["pybind11"],
)