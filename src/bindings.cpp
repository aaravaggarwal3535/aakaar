#include <pybind11/pybind11.h>
#include <memory>
#include "tensor.h"

namespace py = pybind11;

// Forward declaration
std::shared_ptr<Tensor> run_curand_uniform(int size, unsigned long long seed);

PYBIND11_MODULE(_C, m) {
    
    // 1. Expose our Custom Tensor Class to Python
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<int>()) // Allow creation via aakaar._C.Tensor(size)
        .def_readonly("size", &Tensor::size)
        .def("cpu", &Tensor::cpu, "Copy tensor data to a CPU NumPy array");

    // 2. Expose the random function
    m.def("generate_random", &run_curand_uniform, "Generate random numbers directly on the GPU");
}