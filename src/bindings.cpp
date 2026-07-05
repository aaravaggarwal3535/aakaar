#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "tensor.h"

namespace py = pybind11;

// Forward declarations for underlying backend variants
void fill_cpu_random(std::shared_ptr<Tensor> t, unsigned long long seed);
std::shared_ptr<Tensor> run_cpu_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_add_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cpu_sub_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cpu_mul_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cpu_div_scalar(std::shared_ptr<Tensor> a, float s);

#ifndef AAKAAR_NO_CUDA
#include "allocator.h"
void run_curand_uniform(std::shared_ptr<Tensor> t, unsigned long long seed);
std::shared_ptr<Tensor> run_cublas_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_add_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cuda_sub_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cuda_mul_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cuda_div_scalar(std::shared_ptr<Tensor> a, float s);
void empty_cache() { CachingAllocator::get_instance().empty_cache(); }
#endif

// ============================================================================
// === NEW STATIC DISPATCHERS (Pattern generated for Add, Sub, Mul, Div) ===
// ============================================================================

static std::shared_ptr<Tensor> dispatch_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") return run_cuda_add(a, b);
#endif
    return run_cpu_add(a, b);
}
static std::shared_ptr<Tensor> dispatch_add_scalar(std::shared_ptr<Tensor> a, float s) {
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") return run_cuda_add_scalar(a, s);
#endif
    return run_cpu_add_scalar(a, s);
}

static std::shared_ptr<Tensor> dispatch_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") return run_cuda_sub(a, b);
#endif
    return run_cpu_sub(a, b);
}
static std::shared_ptr<Tensor> dispatch_sub_scalar(std::shared_ptr<Tensor> a, float s) {
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") return run_cuda_sub_scalar(a, s);
#endif
    return run_cpu_sub_scalar(a, s);
}

static std::shared_ptr<Tensor> dispatch_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") return run_cuda_mul(a, b);
#endif
    return run_cpu_mul(a, b);
}
static std::shared_ptr<Tensor> dispatch_mul_scalar(std::shared_ptr<Tensor> a, float s) {
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") return run_cuda_mul_scalar(a, s);
#endif
    return run_cpu_mul_scalar(a, s);
}

static std::shared_ptr<Tensor> dispatch_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") return run_cuda_div(a, b);
#endif
    return run_cpu_div(a, b);
}
static std::shared_ptr<Tensor> dispatch_div_scalar(std::shared_ptr<Tensor> a, float s) {
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") return run_cuda_div_scalar(a, s);
#endif
    return run_cpu_div_scalar(a, s);
}

int cuda_device_count() {
#ifdef AAKAAR_NO_CUDA
    return 0;
#else
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return 0;  // no driver, no GPU, etc. — not an error condition, just "unavailable"
    return count;
#endif
}

bool cuda_is_available() {
    return cuda_device_count() > 0;
}

// ============================================================================
// === PYBIND11 MODULE DEFINITIONS ===
// ============================================================================

PYBIND11_MODULE(_C, m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<int>, std::string>())
        .def_readonly("device", &Tensor::device)
        .def_readonly("size", &Tensor::size)
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("strides", &Tensor::strides)
        .def("to_numpy", &Tensor::to_numpy)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("contiguous", &Tensor::contiguous)
        .def("__repr__", &Tensor::repr)
        .def("__str__", &Tensor::repr)
        .def("to_device", &Tensor::to_device)
        .def("to", &Tensor::to_device)
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
        })

        // === SIMPLIFIED DUNDER OPERATORS USING THE DISPATCHERS ===
        .def("__add__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_add(a, b); })
        .def("__add__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_add_scalar(a, s); })
        .def("__radd__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_add_scalar(a, s); })

        .def("__sub__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_sub(a, b); })
        .def("__sub__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_sub_scalar(a, s); })

        .def("__mul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_mul(a, b); })
        .def("__mul__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_mul_scalar(a, s); })
        .def("__rmul__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_mul_scalar(a, s); })

        .def("__truediv__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_div(a, b); })
        .def("__truediv__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_div_scalar(a, s); });

    // Standalone module functions
    m.def("fill_cpu_random", &fill_cpu_random, "Fill CPU Tensor with random numbers");
    m.def("cpu_matmul", &run_cpu_matmul, "CPU matrix multiplication");
    m.def("cpu_add", &run_cpu_add, "CPU elementwise add");
    m.def("cpu_sub", &run_cpu_sub, "CPU elementwise subtract");
    m.def("cpu_mul", &run_cpu_mul, "CPU elementwise multiply");
    m.def("cpu_div", &run_cpu_div, "CPU elementwise divide");
    m.def("is_available", &cuda_is_available, "Check if a CUDA-capable GPU is actually present and usable");
    m.def("device_count", &cuda_device_count, "Number of CUDA-capable GPUs detected");

#ifndef AAKAAR_NO_CUDA
    m.def("generate_random", &run_curand_uniform, "Fill GPU Tensor with random numbers");
    m.def("cuda_matmul", &run_cublas_matmul, "cuBLAS GPU matrix multiplication");
    m.def("empty_cache", &empty_cache, "Release cached GPU memory");
    m.def("cuda_add", &run_cuda_add, "CUDA elementwise add");
    m.def("cuda_sub", &run_cuda_sub, "CUDA elementwise subtract");
    m.def("cuda_mul", &run_cuda_mul, "CUDA elementwise multiply");
    m.def("cuda_div", &run_cuda_div, "CUDA elementwise divide");
    m.attr("HAS_CUDA") = true;
#else
    m.attr("HAS_CUDA") = false;
#endif
}
