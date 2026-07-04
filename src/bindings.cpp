#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "tensor.h"

namespace py = pybind11;

void fill_cpu_random(std::shared_ptr<Tensor> t, unsigned long long seed);

#ifndef AAKAAR_NO_CUDA
#include "allocator.h"
void run_curand_uniform(std::shared_ptr<Tensor> t, unsigned long long seed);
void empty_cache() { CachingAllocator::get_instance().empty_cache(); }
#endif

PYBIND11_MODULE(_C, m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<int, std::string>())
        .def_readonly("device", &Tensor::device)
        .def_readonly("size", &Tensor::size)
        .def("to_numpy", &Tensor::to_numpy)
        .def("__repr__", &Tensor::repr)
        .def("__str__", &Tensor::repr)
        .def("__len__", [](Tensor &t) { return t.size; })
        .def("__getitem__", [](Tensor &t, int i) {
            if (i < 0) i += t.size;
            return t.get_item(i);
        });

    m.def("fill_cpu_random", &fill_cpu_random, "Fill CPU Tensor with random numbers");

#ifndef AAKAAR_NO_CUDA
    m.def("generate_random", &run_curand_uniform, "Fill GPU Tensor with random numbers");
    m.def("empty_cache", &empty_cache, "Release cached GPU memory");
    m.attr("HAS_CUDA") = true;
#else
    m.attr("HAS_CUDA") = false;
#endif
}