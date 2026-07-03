#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "tensor.h"
#include "allocator.h"

namespace py = pybind11;

void run_curand_uniform(std::shared_ptr<Tensor> t, unsigned long long seed);
void fill_cpu_random(std::shared_ptr<Tensor> t, unsigned long long seed);

void empty_cache() {
    CachingAllocator::get_instance().empty_cache();
}

PYBIND11_MODULE(_C, m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<int, std::string>())
        .def_readonly("device", &Tensor::device)
        .def("to_numpy", &Tensor::to_numpy);

    m.def("generate_random", &run_curand_uniform, "Fill GPU Tensor with random numbers");
    m.def("fill_cpu_random", &fill_cpu_random, "Fill CPU Tensor with random numbers");

    m.def("empty_cache", &empty_cache, "Release cached GPU memory");
}