import os
import subprocess
import sysconfig # <--- ADD THIS
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

class CUDABuildExtension(build_ext):
    def build_extensions(self):
        nvcc_flags = [
            '-O3',
            '-gencode=arch=compute_75,code=sm_75', 
            '-gencode=arch=compute_86,code=sm_86', 
            '-gencode=arch=compute_89,code=sm_89', 
            '-Xcompiler=-fPIC'                     
        ]

        for ext in self.extensions:
            # 1. Grab all include directories from the Extension definition
            includes = [f"-I{d}" for d in ext.include_dirs]
            # 2. Explicitly add Python's core header directory (needed for pybind11)
            includes.append(f"-I{sysconfig.get_path('include')}")

            cu_sources = [s for s in ext.sources if s.endswith('.cu')]
            cpp_sources = [s for s in ext.sources if s.endswith('.cpp')]
            
            objects = []
            
            for cu_file in cu_sources:
                obj_file = cu_file.replace('.cu', '.o')
                # 3. Add the `includes` list to the nvcc command!
                nvcc_cmd = ['nvcc', '-c', cu_file, '-o', obj_file] + nvcc_flags + includes
                print(f"Compiling CUDA: {' '.join(nvcc_cmd)}")
                subprocess.check_call(nvcc_cmd)
                objects.append(obj_file)
                
            ext.sources = cpp_sources
            ext.extra_objects = objects

        super().build_extensions()

# Define the Python Extension
aakaar_ext = Extension(
    'aakaar._C',
    sources=['src/bindings.cpp', 'src/random_kernel.cu'],
    include_dirs=[
        pybind11.get_include(),
        '/usr/local/cuda/include'
    ],
    library_dirs=[
        '/usr/local/cuda/lib64'
    ],
    libraries=['curand', 'cudart'], # Added cudart here
    language='c++'
)

setup(
    name='aakaar',
    version='0.1.0',
    author='Aarav Aggarwal',
    description='A custom standalone ML library featuring CUDA-accelerated operations.',
    packages=['aakaar'],
    ext_modules=[aakaar_ext],
    cmdclass={'build_ext': CUDABuildExtension},
    install_requires=[
        'numpy', 
        'nvidia-curand-cu12',
        'nvidia-cuda-runtime-cu12'
    ],
    setup_requires=['pybind11']
)   