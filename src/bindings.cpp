#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "tensor.h"

namespace py = pybind11;

void fill_cpu_random(std::shared_ptr<Tensor> t, unsigned long long seed);
std::shared_ptr<Tensor> run_cpu_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

#ifndef AAKAAR_NO_CUDA
#include "allocator.h"
void run_curand_uniform(std::shared_ptr<Tensor> t, unsigned long long seed);
std::shared_ptr<Tensor> run_cublas_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
void empty_cache() { CachingAllocator::get_instance().empty_cache(); }
#endif

PYBIND11_MODULE(_C, m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<int>, std::string>())
        .def_readonly("device", &Tensor::device)
        .def_readonly("size", &Tensor::size)
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("strides", &Tensor::strides)
        .def("to_numpy", &Tensor::to_numpy)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("__repr__", &Tensor::repr)
        .def("__str__", &Tensor::repr)
        .def("__len__", [](Tensor &t) { return t.shape.empty() ? 0 : t.shape[0]; })
        .def("__getitem__", [](std::shared_ptr<Tensor> self, py::object key) -> py::object {
            std::vector<py::object> items;
            if (py::isinstance<py::tuple>(key)) {
                for (auto item : key) items.push_back(py::reinterpret_borrow<py::object>(item));
            } else {
                items.push_back(key);
            }

            size_t ndim = self->shape.size();
            if (items.size() > ndim)
                throw std::out_of_range("Too many indices for tensor of dimension " + std::to_string(ndim));

            std::vector<int> new_shape, new_strides;
            int offset = 0;
            bool all_int = true;

            for (size_t d = 0; d < ndim; ++d) {
                if (d < items.size()) {
                    py::object sel = items[d];
                    if (py::isinstance<py::int_>(sel)) {
                        int i = sel.cast<int>();
                        if (i < 0) i += self->shape[d];
                        if (i < 0 || i >= self->shape[d])
                            throw std::out_of_range("Index out of range on dimension " + std::to_string(d));
                        offset += i * self->strides[d];
                    } else if (py::isinstance<py::slice>(sel)) {
                        all_int = false;
                        py::slice s = sel.cast<py::slice>();
                        size_t start, stop, step, slicelength;
                        if (!s.compute((size_t)self->shape[d], &start, &stop, &step, &slicelength))
                            throw std::runtime_error("Invalid slice");
                        offset += (int)start * self->strides[d];
                        new_shape.push_back((int)slicelength);
                        new_strides.push_back(self->strides[d] * (int)step);
                    } else {
                        throw std::runtime_error("Index must be int or slice");
                    }
                } else {
                    all_int = false;
                    new_shape.push_back(self->shape[d]);
                    new_strides.push_back(self->strides[d]);
                }
            }

            if (all_int) {
                float value;
#ifndef AAKAAR_NO_CUDA
                if (self->device == "cuda") {
                    cudaMemcpy(&value, self->data_ptr + offset, sizeof(float), cudaMemcpyDeviceToHost);
                    return py::float_(value);
                }
#endif
                value = self->data_ptr[offset];
                return py::float_(value);
            }

            auto view = std::make_shared<Tensor>(self, offset, new_shape, new_strides);
            return py::cast(view);
        });

    m.def("fill_cpu_random", &fill_cpu_random, "Fill CPU Tensor with random numbers");
    m.def("cpu_matmul", &run_cpu_matmul, "CPU matrix multiplication");

#ifndef AAKAAR_NO_CUDA
    m.def("generate_random", &run_curand_uniform, "Fill GPU Tensor with random numbers");
    m.def("cuda_matmul", &run_cublas_matmul, "cuBLAS GPU matrix multiplication");
    m.def("empty_cache", &empty_cache, "Release cached GPU memory");
    m.attr("HAS_CUDA") = true;
#else
    m.attr("HAS_CUDA") = false;
#endif
}