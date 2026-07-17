import os
import re
import sys
import shutil
import subprocess
import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


def has_nvcc():
    return shutil.which("nvcc") is not None
def get_nvcc_version():
    """Returns (major, minor) from `nvcc --version`, or None if nvcc unavailable."""
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        match = re.search(r"release (\d+)\.(\d+)", out)
        if match:
            return (int(match.group(1)), int(match.group(2)))
    except Exception:
        pass
    return None

def get_gencode_flags():
    version = get_nvcc_version()
    if version is None:
        version = (12, 0)

    major, minor = version
    ver = major * 10 + minor

    flags = []

    # sm_60/61/62 (Pascal): supported CUDA 8-12.x, DROPPED in CUDA 13.0+
    if ver < 130:
        flags.append("-gencode=arch=compute_60,code=sm_60")

    # sm_70/72 (Volta): supported CUDA 9-12.x, ALSO DROPPED in CUDA 13.0+
    # (CUDA 13's minimum supported architecture is Turing / sm_75 — this was
    # the actual cause of the "nvcc ... exit status 1" build failure under
    # CUDA 13.3: compute_70/sm_70 is no longer a recognized -gencode target.)
    if ver < 130:
        flags.append("-gencode=arch=compute_70,code=sm_70")

    # sm_75 (Turing, RTX 20-series): broadly supported, and is CUDA 13.x's
    # actual minimum supported architecture.
    flags.append("-gencode=arch=compute_75,code=sm_75")

    # sm_80/86 (Ampere): CUDA 11.0+
    flags.append("-gencode=arch=compute_80,code=sm_80")
    flags.append("-gencode=arch=compute_86,code=sm_86")

    # sm_89 (Ada, RTX 40-series): CUDA 11.8+
    if ver >= 118:
        flags.append("-gencode=arch=compute_89,code=sm_89")

    # sm_90 (Hopper): CUDA 11.8+ (12.0+ recommended)
    if ver >= 120:
        flags.append("-gencode=arch=compute_90,code=sm_90")

    # sm_120 (Blackwell, RTX 50-series): CUDA 12.8+
    if ver >= 128:
        flags.append("-gencode=arch=compute_120,code=sm_120")

    newest_arch = "compute_120" if ver >= 128 else ("compute_90" if ver >= 120 else "compute_86")
    flags.append(f"-gencode=arch={newest_arch},code={newest_arch}")

    print(f"Detected CUDA {major}.{minor} — using gencode targets: {flags}")
    return flags

CUDA_AVAILABLE = has_nvcc()
print(f"CUDA toolkit detected: {CUDA_AVAILABLE}")


class CUDABuildExtension(build_ext):
    def build_extensions(self):
        is_windows = sys.platform == "win32"
        is_macos = sys.platform == "darwin"
        is_linux = not is_windows and not is_macos

        # =================================================================
        # OpenBLAS wiring (platform-specific — cblas.h/lib live in very
        # different places depending on OS and package manager)
        # =================================================================
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
            # OpenBLAS — Homebrew installs it "keg-only" (not symlinked into
            # /opt/homebrew or /usr/local directly) to avoid clashing with
            # Apple's own Accelerate BLAS, so paths must be added explicitly.
            brew_prefixes_blas = ["/opt/homebrew/opt/openblas", "/usr/local/opt/openblas"]
            found_blas = None
            for prefix in brew_prefixes_blas:
                if os.path.isfile(os.path.join(prefix, "include", "cblas.h")):
                    found_blas = prefix
                    break
            if found_blas:
                for ext in self.extensions:
                    ext.include_dirs.append(os.path.join(found_blas, "include"))
                    ext.library_dirs.append(os.path.join(found_blas, "lib"))
                    ext.libraries.append("openblas")
            else:
                print("WARNING: OpenBLAS not found via Homebrew at", brew_prefixes_blas,
                      "— install with `brew install openblas`.")

            # libomp — Apple's clang ships with no OpenMP runtime at all.
            # Homebrew's libomp provides omp.h and libomp.dylib, but requires
            # -Xpreprocessor -fopenmp (NOT plain -fopenmp, which is GCC's
            # flag and unrecognized by Apple clang) plus explicit linking,
            # since libomp is also keg-only.
            brew_prefixes_omp = ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
            found_omp = None
            for prefix in brew_prefixes_omp:
                if os.path.isfile(os.path.join(prefix, "include", "omp.h")):
                    found_omp = prefix
                    break
            if found_omp:
                for ext in self.extensions:
                    ext.include_dirs.append(os.path.join(found_omp, "include"))
                    ext.library_dirs.append(os.path.join(found_omp, "lib"))
                    ext.libraries.append("omp")
                    ext.extra_compile_args = list(ext.extra_compile_args or []) + \
                        ["-Xpreprocessor", "-fopenmp"]
                    ext.extra_link_args = list(ext.extra_link_args or []) + \
                        ["-Xpreprocessor", "-fopenmp"]
            else:
                print("WARNING: libomp not found via Homebrew at", brew_prefixes_omp,
                      "— install with `brew install libomp`. "
                      "If cpu_kernel.cpp unconditionally includes <omp.h>, the build "
                      "will fail without this; guard that include behind a macro "
                      "(e.g. AAKAAR_NO_OPENMP) if you need graceful degradation.")

        else:
            # Linux
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

            # Belt-and-suspenders: ask the package manager directly if the
            # standard candidate paths didn't turn it up.
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
                print("WARNING: OpenBLAS cblas.h not found on Linux — "
                      "build will fail to link matmul against BLAS.")

            # dlopen/dlsym support (Linux only — not a normal link target on
            # Windows/macOS).
            for ext in self.extensions:
                ext.libraries.append("dl")
                ext.libraries.append("gomp")

        # =================================================================
        # CUDA wiring (Linux/Windows only — CUDA_AVAILABLE is always False
        # on macOS since there's no nvcc, so this block is naturally skipped
        # there and AAKAAR_NO_CUDA=1 is set below like any other CPU-only
        # build)
        # =================================================================
        if CUDA_AVAILABLE:
            cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME") or "/usr/local/cuda"
            cuda_include = os.path.join(cuda_home, "include")
            cuda_lib = os.path.join(cuda_home, "lib", "x64") if is_windows else os.path.join(cuda_home, "lib64")

            nvcc_flags = ["-O3", "-std=c++17"] + get_gencode_flags()
            nvcc_flags.append("-Xcompiler=/MD" if is_windows else "-Xcompiler=-fPIC")

            if CUDNN_PATHS is not None:
                cudnn_include, cudnn_lib = CUDNN_PATHS
                for ext in self.extensions:
                    if "src/conv_cudnn.cu" not in ext.sources:
                        ext.sources.append("src/conv_cudnn.cu")
                    if cudnn_include not in ext.include_dirs:
                        ext.include_dirs.append(cudnn_include)
                    if cudnn_lib not in ext.library_dirs:
                        ext.library_dirs.append(cudnn_lib)
                    ext.libraries.append("cudnn")
                    ext.define_macros.append(("AAKAAR_HAS_CUDNN", "1"))

            # Point nvcc at GCC 13 explicitly — CUDA 12.4 does not support
            # GCC 14 (the manylinux_2_28 default). -allow-unsupported-compiler
            # was tried first and rejected: forcing nvcc to parse GCC 14's
            # real headers produced genuine compile errors (std::make_shared
            # and others), confirming the incompatibility is real, not just
            # an overcautious version gate.
            gcc13_path = "/opt/rh/gcc-toolset-13/root/usr/bin/g++"
            if not is_windows and os.path.isfile(gcc13_path):
                nvcc_flags.append(f"-ccbin={gcc13_path}")

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
                # cudnn.h ships alongside cuda headers on most installs (apt libcudnn8-dev,
                # or bundled in nvidia pip wheels); if the build fails with "cudnn.h not
                # found", it needs to be installed/pointed at explicitly — not something
                # this script silently works around.
            else:
                # No CUDA toolkit: skip .cu sources entirely, CPU-only build.
                # This applies uniformly on Windows-without-CUDA, Linux-without-
                # CUDA, and always on macOS.
                print("No nvcc found — building CPU-only extension (no CUDA support).")
                ext.define_macros.append(("AAKAAR_NO_CUDA", "1"))

            ext.sources = cpp_sources
            ext.extra_objects = objects

        super().build_extensions()


if sys.platform == "win32":
    host_compiler_flags = ["/std:c++17", "/openmp"]
elif sys.platform == "darwin":
    host_compiler_flags = ["-std=c++17"]
else:
    host_compiler_flags = ["-std=c++17", "-fopenmp"]
aakaar_ext = Extension(
    "aakaar._C",
    sources=[
        "src/bindings.cpp",
        "src/cpu_kernel.cpp",
        "src/conv_kernel.cpp",
        "src/random_kernel.cu",
        "src/matmul_kernel.cu",
        "src/elementwise_kernel.cu",
        "src/reduction_kernel.cu",
        "src/strided_copy_kernel.cu",
        "src/conv_kernel.cu",
    ],
    include_dirs=[
        pybind11.get_include(),
        "src",
    ],
    libraries=[],  # populated conditionally above
    language="c++",
    extra_compile_args=host_compiler_flags,
)

def log_softmax(x, dim=-1):
    """Numerically stable log(softmax(x)) — computes log-sum-exp using the
    max-subtraction trick without ever materializing the (potentially huge
    or tiny) raw softmax probabilities first."""
    m = x.max(dim=dim, keepdim=True)
    shifted = x - m
    log_sum_exp = shifted.exp().sum(dim=dim, keepdim=True).log()
    return shifted - log_sum_exp


def nll_loss(log_probs, target_onehot):
    """Negative log likelihood given log-probabilities (e.g. from
    log_softmax) and one-hot targets. Matches torch.nn.functional.nll_loss
    semantics but takes one-hot targets rather than integer class indices,
    since aakaar has no gather/fancy-indexing op yet (see cross_entropy)."""
    return -(target_onehot * log_probs).sum() / log_probs.shape[0]


def cross_entropy(logits, target_onehot):
    """Numerically stable cross-entropy from raw logits — computes
    log_softmax internally rather than softmax-then-log (avoids the
    precision loss of exponentiating then re-logging). This is the
    torch-equivalent-signature version; cross_entropy_from_probs (above,
    already in this file) stays available for callers who already have
    probabilities.

    NOTE: takes one-hot targets, not integer class-index targets like
    torch.nn.functional.cross_entropy — aakaar has no gather/fancy-indexing
    operation yet to select log_probs[i, target[i]] directly. Convert
    integer labels to one-hot first (see one_hot() pattern in examples)."""
    return nll_loss(log_softmax(logits, dim=-1), target_onehot)


def l1_loss(pred, target):
    """Mean absolute error. NOTE: aakaar has no abs() yet — implemented via
    sqrt(x^2 + eps) is a bad idea (biases small errors); the honest
    implementation needs a real elementwise abs(). Flagged as blocked,
    same as Adam above, rather than faked."""
    raise NotImplementedError(
        "l1_loss requires an elementwise abs() operation, which doesn't exist "
        "in aakaar yet. This is a real, tracked gap — not implemented here to "
        "avoid a numerically incorrect approximation (e.g. sqrt(x^2) has poor "
        "gradient behavior near zero)."
    )


def binary_cross_entropy_with_logits(logits, target):
    """Numerically stable BCE from raw logits, using the standard
    log-sum-exp reformulation: max(x,0) - x*target + log(1+exp(-|x|)).
    NOTE: this needs elementwise abs() for the |x| term and doesn't have
    one yet either — same gap as l1_loss. Flagged rather than faked."""
    raise NotImplementedError(
        "binary_cross_entropy_with_logits requires elementwise abs(), which "
        "doesn't exist in aakaar yet. Tracked gap, not implemented here."
    )

def find_cudnn():
    candidates = []
    cudnn_root = os.environ.get("CUDNN_ROOT")
    if cudnn_root:
        candidates.append((os.path.join(cudnn_root, "include"), os.path.join(cudnn_root, "lib", "x64")))
    cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append((os.path.join(cuda_home, "include"), os.path.join(cuda_home, "lib", "x64")))

    # pip-installed nvidia-cudnn-cu12: ships headers+libs under site-packages/nvidia/cudnn
    import site
    for root in site.getsitepackages() + [site.getusersitepackages()]:
        base = os.path.join(root, "nvidia", "cudnn")
        inc = os.path.join(base, "include")
        lib_win = os.path.join(base, "lib", "x64")
        lib_linux = os.path.join(base, "lib")
        if os.path.isfile(os.path.join(inc, "cudnn.h")):
            candidates.append((inc, lib_win if sys.platform == "win32" else lib_linux))

    nvidia_cudnn_root = r"C:\Program Files\NVIDIA\CUDNN"
    if os.path.isdir(nvidia_cudnn_root):
        version = get_nvcc_version()
        cuda_ver_str = f"{version[0]}.{version[1]}" if version else None
        for entry in sorted(os.listdir(nvidia_cudnn_root), reverse=True):
            base = os.path.join(nvidia_cudnn_root, entry)
            inc_base = os.path.join(base, "include")
            if not os.path.isdir(inc_base):
                continue
            subdirs = ([cuda_ver_str] if cuda_ver_str else []) + \
                      [d for d in os.listdir(inc_base) if d != cuda_ver_str]
            for sub in subdirs:
                if sub:
                    candidates.append((os.path.join(inc_base, sub),
                                        os.path.join(base, "lib", sub, "x64")))

    candidates.append(("/usr/local/cuda/include", "/usr/local/cuda/lib64"))
    candidates.append(("/usr/include", "/usr/lib/x86_64-linux-gnu"))

    for inc, lib in candidates:
        if os.path.isfile(os.path.join(inc, "cudnn.h")):
            return (inc, lib)
    return None

CUDNN_PATHS = find_cudnn() if CUDA_AVAILABLE else None
print(f"cuDNN detected: {CUDNN_PATHS is not None}" + ("" if CUDNN_PATHS is None else f" ({CUDNN_PATHS[0]})"))

setup(
    name="aakaar",
    version="0.2.0",
    author="Aarav Aggarwal",
    description="A custom standalone ML library featuring CUDA-accelerated operations (CPU fallback supported).",
    packages=["aakaar", "aakaar._openblas_bin"],
    package_data={"aakaar": ["_openblas_bin/*.dll"]},
    include_package_data=True,
    ext_modules=[aakaar_ext],
    cmdclass={"build_ext": CUDABuildExtension},
    install_requires=[
        "numpy",
        'nvidia-cuda-runtime; platform_system=="Windows" or platform_system=="Linux"',
        'nvidia-cublas; platform_system=="Windows" or platform_system=="Linux"',
        'nvidia-curand; platform_system=="Windows" or platform_system=="Linux"',
        'nvidia-cudnn-cu13; platform_system=="Windows" or platform_system=="Linux"',
    ],
    setup_requires=["pybind11"],
)