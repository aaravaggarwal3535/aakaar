#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "tensor.h"

namespace py = pybind11;

// ---- Forward declarations: raw ops ----
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
std::shared_ptr<Tensor> run_cpu_sum_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cpu_sum_all(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_broadcast_axis(std::shared_ptr<Tensor> a, int dim, int target_size);

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
std::shared_ptr<Tensor> run_cuda_sum_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cuda_sum_all(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_broadcast_axis(std::shared_ptr<Tensor> a, int dim, int target_size);
void empty_cache() { CachingAllocator::get_instance().empty_cache(); }
#endif

// ---- Forward declarations: autograd-aware dispatch ----
static std::shared_ptr<Tensor> dispatch_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
static std::shared_ptr<Tensor> dispatch_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
static std::shared_ptr<Tensor> dispatch_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
static std::shared_ptr<Tensor> dispatch_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
static std::shared_ptr<Tensor> dispatch_add_scalar(std::shared_ptr<Tensor> a, float s);
static std::shared_ptr<Tensor> dispatch_sub_scalar(std::shared_ptr<Tensor> a, float s);
static std::shared_ptr<Tensor> dispatch_mul_scalar(std::shared_ptr<Tensor> a, float s);
static std::shared_ptr<Tensor> dispatch_div_scalar(std::shared_ptr<Tensor> a, float s);
static std::shared_ptr<Tensor> dispatch_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
static std::shared_ptr<Tensor> dispatch_sum_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim);

// ---- Dispatch implementations ----

static std::shared_ptr<Tensor> dispatch_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_add(a, b);
    else
#endif
    result = run_cpu_add(a, b);

    if (a->requires_grad || b->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "add";
        node->backward_fn = [](std::shared_ptr<Tensor> grad_out) {
            return std::vector<std::shared_ptr<Tensor>>{grad_out, grad_out};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sub(a, b);
    else
#endif
    result = run_cpu_sub(a, b);

    if (a->requires_grad || b->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "sub";
        node->backward_fn = [](std::shared_ptr<Tensor> grad_out) {
            auto neg = dispatch_mul_scalar(grad_out, -1.0f);
            return std::vector<std::shared_ptr<Tensor>>{grad_out, neg};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_mul(a, b);
    else
#endif
    result = run_cpu_mul(a, b);

    if (a->requires_grad || b->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "mul";
        node->backward_fn = [a, b](std::shared_ptr<Tensor> grad_out) {
            auto da = dispatch_mul(grad_out, b);
            auto db = dispatch_mul(grad_out, a);
            return std::vector<std::shared_ptr<Tensor>>{da, db};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_div(a, b);
    else
#endif
    result = run_cpu_div(a, b);

    if (a->requires_grad || b->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "div";
        node->backward_fn = [a, b](std::shared_ptr<Tensor> grad_out) {
            auto da = dispatch_div(grad_out, b);
            auto b_sq = dispatch_mul(b, b);
            auto a_over_bsq = dispatch_div(a, b_sq);
            auto neg = dispatch_mul_scalar(a_over_bsq, -1.0f);
            auto db = dispatch_mul(grad_out, neg);
            return std::vector<std::shared_ptr<Tensor>>{da, db};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_add_scalar(std::shared_ptr<Tensor> a, float s) {
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_add_scalar(a, s);
    else
#endif
    result = run_cpu_add_scalar(a, s);

    if (a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "add_scalar";
        node->backward_fn = [](std::shared_ptr<Tensor> grad_out) {
            return std::vector<std::shared_ptr<Tensor>>{grad_out};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_sub_scalar(std::shared_ptr<Tensor> a, float s) {
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sub_scalar(a, s);
    else
#endif
    result = run_cpu_sub_scalar(a, s);

    if (a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "sub_scalar";
        node->backward_fn = [](std::shared_ptr<Tensor> grad_out) {
            return std::vector<std::shared_ptr<Tensor>>{grad_out};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_mul_scalar(std::shared_ptr<Tensor> a, float s) {
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_mul_scalar(a, s);
    else
#endif
    result = run_cpu_mul_scalar(a, s);

    if (a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "mul_scalar";
        node->backward_fn = [s](std::shared_ptr<Tensor> grad_out) {
            auto da = dispatch_mul_scalar(grad_out, s);
            return std::vector<std::shared_ptr<Tensor>>{da};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_div_scalar(std::shared_ptr<Tensor> a, float s) {
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_div_scalar(a, s);
    else
#endif
    result = run_cpu_div_scalar(a, s);

    if (a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "div_scalar";
        node->backward_fn = [s](std::shared_ptr<Tensor> grad_out) {
            auto da = dispatch_div_scalar(grad_out, s);
            return std::vector<std::shared_ptr<Tensor>>{da};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cublas_matmul(a, b);
    else
#endif
    result = run_cpu_matmul(a, b);

    if (a->requires_grad || b->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "matmul";
        node->backward_fn = [a, b](std::shared_ptr<Tensor> grad_out) {
            int nd_a = (int)a->shape.size();
            int nd_b = (int)b->shape.size();

            auto bT = b->transpose(nd_b-2, nd_b-1)->contiguous();
            auto aT = a->transpose(nd_a-2, nd_a-1)->contiguous();

            auto da_full = dispatch_matmul(grad_out, bT);  // shape matches grad_out's batch dims
            auto db_full = dispatch_matmul(aT, grad_out);

            // If a broadcast along any batch axis, da_full's batch dims are larger than a's;
            // sum those axes back down to match a's original shape (the broadcast gradient rule).
            auto reduce_broadcast = [](std::shared_ptr<Tensor> grad, const std::vector<int>& target_shape) {
                int nd_g = (int)grad->shape.size();
                int nd_t = (int)target_shape.size();
                int offset = nd_g - nd_t;
                auto g = grad;
                // sum any leading extra batch dims entirely (target didn't have them at all)
                for (int i = 0; i < offset; ++i) {
                    g = dispatch_sum_axis(g, 0, false);
                }
                // for remaining aligned dims, sum any axis where target was 1 but grad isn't
                for (int i = 0; i < nd_t - 2; ++i) {  // skip the trailing M,K/K,N dims
                    if (target_shape[i] == 1 && g->shape[i] != 1) {
                        g = dispatch_sum_axis(g, i, true);
                    }
                }
                return g;
            };

            auto da = reduce_broadcast(da_full, a->shape);
            auto db = reduce_broadcast(db_full, b->shape);
            return std::vector<std::shared_ptr<Tensor>>{da, db};
        };
        result->grad_fn = node;
    }
    return result;
}

// ---- backward() driver: reverse-mode topological traversal ----

static void tensor_backward(std::shared_ptr<Tensor> root, std::shared_ptr<Tensor> grad_output) {
    if (!grad_output) {
        if (root->size != 1)
            throw std::runtime_error("backward() requires an explicit gradient for non-scalar tensors");
        grad_output = std::make_shared<Tensor>(root->shape, root->device);
        float one = 1.0f;
#ifndef AAKAAR_NO_CUDA
        if (root->device == "cuda") {
            cudaMemcpy(grad_output->data_ptr, &one, sizeof(float), cudaMemcpyHostToDevice);
        } else
#endif
        {
            grad_output->data_ptr[0] = one;
        }
    }

    std::vector<std::shared_ptr<Tensor>> topo;
    std::unordered_set<Tensor*> visited;
    std::function<void(std::shared_ptr<Tensor>)> dfs = [&](std::shared_ptr<Tensor> t) {
        if (visited.count(t.get())) return;
        visited.insert(t.get());
        if (t->grad_fn) {
            for (auto& inp : t->grad_fn->inputs) dfs(inp);
        }
        topo.push_back(t);
    };
    dfs(root);

    std::unordered_map<Tensor*, std::shared_ptr<Tensor>> grad_map;
    grad_map[root.get()] = grad_output;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        auto t = *it;
        auto grad_it = grad_map.find(t.get());
        if (grad_it == grad_map.end()) continue;
        auto g = grad_it->second;

        if (t->requires_grad && !t->grad_fn) {
            if (!t->grad) t->grad = g;
            else t->grad = dispatch_add(t->grad, g);
        }

        if (t->grad_fn) {
            auto input_grads = t->grad_fn->backward_fn(g);
            for (size_t i = 0; i < t->grad_fn->inputs.size(); ++i) {
                auto& inp = t->grad_fn->inputs[i];
                if (!inp->requires_grad) continue;
                auto existing = grad_map.find(inp.get());
                if (existing == grad_map.end()) grad_map[inp.get()] = input_grads[i];
                else grad_map[inp.get()] = dispatch_add(existing->second, input_grads[i]);
            }
        }
    }
}

static std::shared_ptr<Tensor> dispatch_broadcast_axis(std::shared_ptr<Tensor> a, int dim, int target_size) {
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") return run_cuda_broadcast_axis(a, dim, target_size);
#endif
    return run_cpu_broadcast_axis(a, dim, target_size);
}

static std::shared_ptr<Tensor> dispatch_sum_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sum_axis(a, dim, keepdim);
    else
#endif
    result = run_cpu_sum_axis(a, dim, keepdim);

    if (a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "sum_axis";
        int original_size = a->shape[dim < 0 ? dim + (int)a->shape.size() : dim];
        int norm_dim = dim < 0 ? dim + (int)a->shape.size() : dim;
        node->backward_fn = [norm_dim, original_size, keepdim](std::shared_ptr<Tensor> grad_out) {
            auto g = grad_out;
            // if keepdim was False, the reduced axis is missing entirely; broadcast_axis
            // expects that axis to exist as size 1, so this assumes keepdim=True upstream
            // usage, or the caller reshapes first — documented limitation for now.
            auto expanded = dispatch_broadcast_axis(g, norm_dim, original_size);
            return std::vector<std::shared_ptr<Tensor>>{expanded};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_sum_all(std::shared_ptr<Tensor> a) {
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sum_all(a);
    else
#endif
    result = run_cpu_sum_all(a);

    if (a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "sum_all";
        auto orig_shape = a->shape;
        node->backward_fn = [orig_shape](std::shared_ptr<Tensor> grad_out) {
            // grad_out is shape (1,). Reshape it to all-ones matching orig_shape's rank,
            // then broadcast each axis up to its original size in turn.
            std::vector<int> ones_shape(orig_shape.size(), 1);
            auto g = grad_out->reshape(ones_shape);
            for (size_t d = 0; d < orig_shape.size(); ++d) {
                if (orig_shape[d] > 1) {
                    g = dispatch_broadcast_axis(g, (int)d, orig_shape[d]);
                }
            }
            return std::vector<std::shared_ptr<Tensor>>{g};
        };
        result->grad_fn = node;
    }
    return result;
}

// ---- Module definition ----

PYBIND11_MODULE(_C, m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<int>, std::string>())
        .def_readonly("device", &Tensor::device)
        .def_readonly("size", &Tensor::size)
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("strides", &Tensor::strides)
        .def_readwrite("requires_grad", &Tensor::requires_grad)
        .def_readonly("grad", &Tensor::grad)
        .def("to_numpy", &Tensor::to_numpy)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("contiguous", [](std::shared_ptr<Tensor> self) {
            auto result = self->contiguous();
            if (self->requires_grad && result.get() != self.get()) {
                result->requires_grad = true;
                auto node = std::make_shared<Node>();
                node->inputs = {self};
                node->op_name = "contiguous";
                node->backward_fn = [](std::shared_ptr<Tensor> grad_out) {
                    return std::vector<std::shared_ptr<Tensor>>{grad_out};
                };
                result->grad_fn = node;
            }
            return result;
        })
        .def("to_device", &Tensor::to_device)
        .def("to", &Tensor::to_device)
        .def("transpose", [](std::shared_ptr<Tensor> self, int dim0, int dim1) {
            auto result = self->transpose(dim0, dim1);
            if (self->requires_grad) {
                result->requires_grad = true;
                auto node = std::make_shared<Node>();
                node->inputs = {self};
                node->op_name = "transpose";
                node->backward_fn = [dim0, dim1](std::shared_ptr<Tensor> grad_out) {
                    return std::vector<std::shared_ptr<Tensor>>{grad_out->transpose(dim0, dim1)->contiguous()};
                };
                result->grad_fn = node;
            }
            return result;
        })
        .def("transpose2d", [](std::shared_ptr<Tensor> self) {
            return self->transpose(0, 1);  // kept for backward compat; no grad tracking here, use .T or .transpose()
        })
        .def_property_readonly("T", [](std::shared_ptr<Tensor> self) {
            auto result = self->transpose_all();
            if (self->requires_grad) {
                result->requires_grad = true;
                auto node = std::make_shared<Node>();
                node->inputs = {self};
                node->op_name = "transpose_all";
                node->backward_fn = [](std::shared_ptr<Tensor> grad_out) {
                    return std::vector<std::shared_ptr<Tensor>>{grad_out->transpose_all()->contiguous()};
                };
                result->grad_fn = node;
            }
            return result;
        })
        .def("zero_grad", &Tensor::zero_grad)
        .def("backward", &tensor_backward, py::arg("grad_output") = nullptr)
        .def("__repr__", &Tensor::repr)
        .def("__str__", &Tensor::repr)
        .def("__len__", [](Tensor &t) { return t.shape.empty() ? 0 : t.shape[0]; })
        .def("view", [](std::shared_ptr<Tensor> self, std::vector<int> new_shape) {
            auto result = self->view(new_shape);
            if (self->requires_grad) {
                result->requires_grad = true;
                auto node = std::make_shared<Node>();
                node->inputs = {self};
                node->op_name = "view";
                auto orig_shape = self->shape;
                node->backward_fn = [orig_shape](std::shared_ptr<Tensor> grad_out) {
                    return std::vector<std::shared_ptr<Tensor>>{grad_out->reshape(orig_shape)};
                };
                result->grad_fn = node;
            }
            return result;
        })
        .def("reshape", [](std::shared_ptr<Tensor> self, std::vector<int> new_shape) {
            auto result = self->reshape(new_shape);
            if (self->requires_grad) {
                result->requires_grad = true;
                auto node = std::make_shared<Node>();
                node->inputs = {self};
                node->op_name = "reshape";
                auto orig_shape = self->shape;
                node->backward_fn = [orig_shape](std::shared_ptr<Tensor> grad_out) {
                    return std::vector<std::shared_ptr<Tensor>>{grad_out->reshape(orig_shape)};
                };
                result->grad_fn = node;
            }
            return result;
        })
        .def("sum", [](std::shared_ptr<Tensor> self, py::object dim, bool keepdim) {
            if (dim.is_none()) return dispatch_sum_all(self);
            return dispatch_sum_axis(self, dim.cast<int>(), keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
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
        .def("__add__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_add(a, b); })
        .def("__add__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_add_scalar(a, s); })
        .def("__radd__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_add_scalar(a, s); })
        .def("__sub__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_sub(a, b); })
        .def("__sub__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_sub_scalar(a, s); })
        .def("__mul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_mul(a, b); })
        .def("__mul__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_mul_scalar(a, s); })
        .def("__rmul__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_mul_scalar(a, s); })
        .def("__truediv__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_div(a, b); })
        .def("__truediv__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_div_scalar(a, s); })
        .def("__matmul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_matmul(a, b); });

    m.def("fill_cpu_random", &fill_cpu_random, "Fill CPU Tensor with random numbers");
    m.def("cpu_matmul", &dispatch_matmul, "CPU matrix multiplication (autograd-aware)");
    m.def("cpu_add", &dispatch_add);
    m.def("cpu_sub", &dispatch_sub);

#ifndef AAKAAR_NO_CUDA
    m.def("generate_random", &run_curand_uniform, "Fill GPU Tensor with random numbers");
    m.def("cuda_matmul", &dispatch_matmul, "cuBLAS GPU matrix multiplication");
    m.def("empty_cache", &empty_cache, "Release cached GPU memory");
    m.attr("HAS_CUDA") = true;
#else
    m.attr("HAS_CUDA") = false;
#endif
}