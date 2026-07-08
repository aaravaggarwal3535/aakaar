#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "tensor.h"
#include <utility>


namespace py = pybind11;

static bool g_grad_enabled = true;

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
std::shared_ptr<Tensor> run_cpu_relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cpu_sigmoid(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_sigmoid_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> sig_output);
std::shared_ptr<Tensor> run_cpu_tanh(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_tanh_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> tanh_output);
std::shared_ptr<Tensor> run_cpu_leaky_relu(std::shared_ptr<Tensor> a, float slope);
std::shared_ptr<Tensor> run_cpu_leaky_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, float slope);
std::shared_ptr<Tensor> run_cpu_exp(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_exp_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> exp_output);
std::shared_ptr<Tensor> run_cpu_log(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_log_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cpu_max_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cpu_max_axis_backward(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                   std::vector<int> orig_shape, int dim, int reduce_size, int inner_size);


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
std::shared_ptr<Tensor> run_cuda_relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cuda_sigmoid(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_sigmoid_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> sig_output);
std::shared_ptr<Tensor> run_cuda_tanh(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_tanh_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> tanh_output);
std::shared_ptr<Tensor> run_cuda_leaky_relu(std::shared_ptr<Tensor> a, float slope);
std::shared_ptr<Tensor> run_cuda_leaky_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, float slope);
std::shared_ptr<Tensor> run_cuda_exp(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_exp_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> exp_output);
std::shared_ptr<Tensor> run_cuda_log(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_log_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cuda_max_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cuda_max_axis_backward(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                    std::vector<int> orig_shape, int dim, int reduce_size, int inner_size);
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
static std::shared_ptr<Tensor> dispatch_broadcast_axis(std::shared_ptr<Tensor> a, int dim, int target_size);
static std::shared_ptr<Tensor> dispatch_contiguous(std::shared_ptr<Tensor> a);

// ---- Dispatch implementations ----
static std::shared_ptr<Tensor> dispatch_leaky_relu(std::shared_ptr<Tensor> a, float slope) {
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_leaky_relu(a, slope);
    else
#endif
    result = run_cpu_leaky_relu(a, slope);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "leaky_relu";
        node->backward_fn = [a, slope](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (a->device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_leaky_relu_backward(grad_out, a, slope)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_leaky_relu_backward(grad_out, a, slope)};
        };
        result->grad_fn = node;
    }
    return result;
}
static std::shared_ptr<Tensor> dispatch_max_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
    if (!a->is_contiguous()) a = dispatch_contiguous(a);

    int ndim = (int)a->shape.size();
    int norm_dim = dim < 0 ? dim + ndim : dim;
    int reduce_size = a->shape[norm_dim];
    int inner_size = 1;
    for (int i = norm_dim + 1; i < ndim; ++i) inner_size *= a->shape[i];

    std::shared_ptr<Tensor> result;
    std::vector<int> argmax;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") { auto pr = run_cuda_max_axis(a, dim, keepdim); result = pr.first; argmax = pr.second; }
    else
#endif
    { auto pr = run_cpu_max_axis(a, dim, keepdim); result = pr.first; argmax = pr.second; }

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "max_axis";
        auto orig_shape = a->shape;
        auto dev = a->device;
        node->backward_fn = [argmax, orig_shape, norm_dim, reduce_size, inner_size, dev]
                             (std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (dev == "cuda") return std::vector<std::shared_ptr<Tensor>>{
                run_cuda_max_axis_backward(grad_out, argmax, orig_shape, norm_dim, reduce_size, inner_size)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{
                run_cpu_max_axis_backward(grad_out, argmax, orig_shape, norm_dim, reduce_size, inner_size)};
        };
        result->grad_fn = node;
    }
    return result;
}
static std::shared_ptr<Tensor> dispatch_exp(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_exp(a);
    else
#endif
    result = run_cpu_exp(a);
    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "exp";
        auto output_copy = result;
        node->backward_fn = [output_copy](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (output_copy->device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_exp_backward(grad_out, output_copy)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_exp_backward(grad_out, output_copy)};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_log(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_log(a);
    else
#endif
    result = run_cpu_log(a);
    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "log";
        node->backward_fn = [a](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (a->device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_log_backward(grad_out, a)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_log_backward(grad_out, a)};
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

    if (g_grad_enabled && a->requires_grad) {
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

    if (g_grad_enabled && a->requires_grad) {
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

    if (g_grad_enabled && a->requires_grad) {
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

    if (g_grad_enabled && a->requires_grad) {
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

static std::shared_ptr<Tensor> dispatch_relu(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_relu(a);
    else
#endif
    result = run_cpu_relu(a);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "relu";
        node->backward_fn = [a](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (a->device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_relu_backward(grad_out, a)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_relu_backward(grad_out, a)};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_sigmoid(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sigmoid(a);
    else
#endif
    result = run_cpu_sigmoid(a);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "sigmoid";
        auto output_copy = result;  // capture the OUTPUT, since sigmoid's derivative uses it, not the input
        node->backward_fn = [output_copy](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (output_copy->device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_sigmoid_backward(grad_out, output_copy)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_sigmoid_backward(grad_out, output_copy)};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_tanh(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_tanh(a);
    else
#endif
    result = run_cpu_tanh(a);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "tanh";
        auto output_copy = result;
        node->backward_fn = [output_copy](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (output_copy->device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_tanh_backward(grad_out, output_copy)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_tanh_backward(grad_out, output_copy)};
        };
        result->grad_fn = node;
    }
    return result;
}

// ---- backward() driver: reverse-mode topological traversal ----

static void tensor_backward(std::shared_ptr<Tensor> root, std::shared_ptr<Tensor> grad_output, bool retain_graph) {
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
            if (t->grad_fn->freed) {
                throw std::runtime_error(
                    "Trying to backward through the graph a second time (or a part of it), but the "
                    "intermediate results needed have already been freed. Pass retain_graph=True to "
                    "backward() the first time if you need to backward through this part of the graph "
                    "more than once."
                );
            }
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

    if (!retain_graph) {
        for (auto& t : topo) {
            if (t->grad_fn) t->grad_fn->freed = true;
        }
    }
}

static std::shared_ptr<Tensor> dispatch_broadcast_axis(std::shared_ptr<Tensor> a, int dim, int target_size) {
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") return run_cuda_broadcast_axis(a, dim, target_size);
#endif
    return run_cpu_broadcast_axis(a, dim, target_size);
}
static std::shared_ptr<Tensor> dispatch_contiguous(std::shared_ptr<Tensor> a) {
    auto result = a->contiguous();
    if (g_grad_enabled && a->requires_grad && result.get() != a.get()) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "contiguous";
        node->backward_fn = [](std::shared_ptr<Tensor> grad_out) {
            return std::vector<std::shared_ptr<Tensor>>{grad_out};
        };
        result->grad_fn = node;
    }
    return result;
}
static std::shared_ptr<Tensor> dispatch_sum_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
    if (!a->is_contiguous()) {
        a = a->contiguous();  // auto-materialize, matching torch's ergonomic sum() behavior
    }
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sum_axis(a, dim, keepdim);
    else
#endif
    result = run_cpu_sum_axis(a, dim, keepdim);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "sum_axis";
        int norm_dim = dim < 0 ? dim + (int)a->shape.size() : dim;
        int original_size = a->shape[norm_dim];
        node->backward_fn = [norm_dim, original_size, keepdim](std::shared_ptr<Tensor> grad_out) {
            auto g = grad_out;
            if (!keepdim) {
                // The forward pass dropped this axis entirely (keepdim=False), so
                // grad_out is missing it too. Reinsert a size-1 axis at norm_dim
                // before broadcasting, since broadcast_axis expects that axis to
                // already exist (as size 1) in order to expand it back out.
                auto reshaped = g->shape;
                reshaped.insert(reshaped.begin() + norm_dim, 1);
                g = g->reshape(reshaped);
            }
            auto expanded = dispatch_broadcast_axis(g, norm_dim, original_size);
            return std::vector<std::shared_ptr<Tensor>>{expanded};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_sum_all(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) {
        a = dispatch_contiguous(a);  // auto-materialize, matching torch's ergonomic sum() behavior
    }
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sum_all(a);
    else
#endif
    result = run_cpu_sum_all(a);

    if (g_grad_enabled && a->requires_grad) {
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


// Sums `grad` down to `target_shape` following numpy/torch broadcasting-gradient
// rules: any leading dims present in `grad` but absent from `target_shape` are
// summed away entirely (they existed only because of broadcasting), and any
// aligned dim where `target_shape` is 1 but `grad` is not gets summed with
// keepdim=true (that axis was broadcast from size 1).
//
// `skip_trailing` lets matmul reuse this for its batch dims only, leaving the
// trailing (M,K)/(K,N) dims untouched — those are never broadcast targets for
// matmul, they're the actual contraction/output dims.
static std::shared_ptr<Tensor> reduce_grad_to_shape(std::shared_ptr<Tensor> grad,
                                                     const std::vector<int>& target_shape,
                                                     int skip_trailing = 0) {
    int nd_g = (int)grad->shape.size();
    int nd_t = (int)target_shape.size();

    if (nd_g < nd_t)
        throw std::runtime_error(
            "reduce_grad_to_shape: gradient has fewer dims (" + std::to_string(nd_g) +
            ") than its target shape (" + std::to_string(nd_t) +
            "). This indicates a bug upstream in the forward/backward broadcasting logic.");

    auto g = grad;
    int offset = nd_g - nd_t;
    // Leading extra dims: target never had them, so they're pure broadcast axes.
    // Always sum axis 0 repeatedly since each removal shifts everything down.
    for (int i = 0; i < offset; ++i) {
        g = dispatch_sum_axis(g, 0, false);
    }

    // Aligned dims: sum where target was 1 but the (now offset-adjusted) grad isn't.
    int nd_g_now = (int)g->shape.size();
    for (int i = 0; i < nd_t - skip_trailing; ++i) {
        if (i >= nd_g_now) break;  // defensive; shouldn't happen given the checks above
        if (target_shape[i] == 1 && g->shape[i] != 1) {
            g = dispatch_sum_axis(g, i, true);
        }
    }

    if (g->shape != target_shape && skip_trailing == 0) {
        throw std::runtime_error(
            "reduce_grad_to_shape: reduced gradient shape does not match target shape "
            "after reduction. This means the forward op's broadcasting and this backward "
            "reduction have gone out of sync — check the forward kernel's broadcast rules.");
    }
    return g;
}

static std::shared_ptr<Tensor> dispatch_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_add(a, b);
    else
#endif
    result = run_cpu_add(a, b);

    if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "add";
        auto a_shape = a->shape;
        auto b_shape = b->shape;
        node->backward_fn = [a_shape, b_shape](std::shared_ptr<Tensor> grad_out) {
            auto da = reduce_grad_to_shape(grad_out, a_shape);
            auto db = reduce_grad_to_shape(grad_out, b_shape);
            return std::vector<std::shared_ptr<Tensor>>{da, db};
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

    if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "sub";
        auto a_shape = a->shape;
        auto b_shape = b->shape;
        node->backward_fn = [a_shape, b_shape](std::shared_ptr<Tensor> grad_out) {
            auto neg = dispatch_mul_scalar(grad_out, -1.0f);
            auto da = reduce_grad_to_shape(grad_out, a_shape);
            auto db = reduce_grad_to_shape(neg, b_shape);
            return std::vector<std::shared_ptr<Tensor>>{da, db};
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

    if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "mul";
        node->backward_fn = [a, b](std::shared_ptr<Tensor> grad_out) {
            // grad_out * b (or a) is computed at the OUTPUT's (broadcasted) shape,
            // then reduced back down to a's (or b's) original shape.
            auto da_full = dispatch_mul(grad_out, b);
            auto db_full = dispatch_mul(grad_out, a);
            auto da = reduce_grad_to_shape(da_full, a->shape);
            auto db = reduce_grad_to_shape(db_full, b->shape);
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

    if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "div";
        node->backward_fn = [a, b](std::shared_ptr<Tensor> grad_out) {
            auto da_full = dispatch_div(grad_out, b);
            auto b_sq = dispatch_mul(b, b);
            auto a_over_bsq = dispatch_div(a, b_sq);
            auto neg = dispatch_mul_scalar(a_over_bsq, -1.0f);
            auto db_full = dispatch_mul(grad_out, neg);
            auto da = reduce_grad_to_shape(da_full, a->shape);
            auto db = reduce_grad_to_shape(db_full, b->shape);
            return std::vector<std::shared_ptr<Tensor>>{da, db};
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

    if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "matmul";
        node->backward_fn = [a, b](std::shared_ptr<Tensor> grad_out) {
            int nd_a = (int)a->shape.size();
            int nd_b = (int)b->shape.size();

            auto bT = b->transpose(nd_b-2, nd_b-1)->contiguous();
            auto aT = a->transpose(nd_a-2, nd_a-1)->contiguous();

            auto da_full = dispatch_matmul(grad_out, bT);
            auto db_full = dispatch_matmul(aT, grad_out);

            // skip_trailing=2: the last two dims are the actual M/K and K/N
            // matmul dims, never broadcast targets — only batch dims (0..ndim-2)
            // can differ due to broadcasting.
            auto da = reduce_grad_to_shape(da_full, a->shape, 2);
            auto db = reduce_grad_to_shape(db_full, b->shape, 2);
            return std::vector<std::shared_ptr<Tensor>>{da, db};
        };
        result->grad_fn = node;
    }
    return result;
}
static std::shared_ptr<Tensor> tensor_from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                                                  std::string device, bool requires_grad) {
    py::buffer_info buf = arr.request();
    std::vector<int> shape;
    for (auto d : buf.shape) shape.push_back((int)d);
    if (shape.empty()) shape.push_back(1);  // treat a numpy scalar as a size-1 tensor

    auto result = Tensor::from_buffer(static_cast<const float*>(buf.ptr), shape, device);
    result->requires_grad = requires_grad;
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
        .def("relu", [](std::shared_ptr<Tensor> self) { return dispatch_relu(self); })
        .def("sigmoid", [](std::shared_ptr<Tensor> self) { return dispatch_sigmoid(self); })
        .def("tanh", [](std::shared_ptr<Tensor> self) { return dispatch_tanh(self); })
        .def("copy_", &Tensor::copy_)
        .def("leaky_relu", [](std::shared_ptr<Tensor> self, float slope) { return dispatch_leaky_relu(self, slope); }, py::arg("slope") = 0.01f)
        .def("max", [](std::shared_ptr<Tensor> self, int dim, bool keepdim) {
            return dispatch_max_axis(self, dim, keepdim);
        }, py::arg("dim"), py::arg("keepdim") = false)
        .def("__neg__", [](std::shared_ptr<Tensor> a) { return dispatch_mul_scalar(a, -1.0f); })
        .def("contiguous", [](std::shared_ptr<Tensor> self) { return dispatch_contiguous(self); })
.def("to", [](std::shared_ptr<Tensor> self, std::string target_device) {
    auto result = self->to_device(target_device);
    if (g_grad_enabled && self->requires_grad && result.get() != self.get()) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {self};
        node->op_name = "to_device";
        auto origin_device = self->device;
        node->backward_fn = [origin_device](std::shared_ptr<Tensor> grad_out) {
            auto grad_input = grad_out->to_device(origin_device);
            return std::vector<std::shared_ptr<Tensor>>{grad_input};
        };
        result->grad_fn = node;
    }
    return result;
}, py::arg("target_device"))

        .def("to", &Tensor::to_device)
        .def("transpose", [](std::shared_ptr<Tensor> self, int dim0, int dim1) {
            auto result = self->transpose(dim0, dim1);
            if (g_grad_enabled && self->requires_grad) {
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
            if (g_grad_enabled && self->requires_grad) {
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
        .def("backward", &tensor_backward, py::arg("grad_output") = nullptr, py::arg("retain_graph") = false)
        .def("zero_grad", &Tensor::zero_grad)
        .def("item", &Tensor::item)
        .def("__repr__", &Tensor::repr)
        .def("__str__", &Tensor::repr)
        .def("__len__", [](Tensor &t) { return t.shape.empty() ? 0 : t.shape[0]; })
        .def("exp", [](std::shared_ptr<Tensor> self) { return dispatch_exp(self); })
        .def("log", [](std::shared_ptr<Tensor> self) { return dispatch_log(self); })
        .def("view", [](std::shared_ptr<Tensor> self, std::vector<int> new_shape) {
            auto result = self->view(new_shape);
            if (g_grad_enabled && self->requires_grad) {
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
            if (g_grad_enabled && self->requires_grad) {
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
        .def("detach", &Tensor::detach)
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

    // One IndexSpec per ORIGINAL dimension. Captured by the backward closure
    // so it can reconstruct exactly how each original position maps to (or
    // is dropped from) the sliced/indexed output — the inverse of the
    // forward mapping computed below.
    struct IndexSpec {
        bool is_int;
        int int_index;  // valid when is_int
        int start;      // valid when !is_int
        int step;       // valid when !is_int
    };
    std::vector<IndexSpec> specs(ndim);

    for (size_t d = 0; d < ndim; ++d) {
        if (d < items.size()) {
            py::object sel = items[d];
            if (py::isinstance<py::int_>(sel)) {
                int i = sel.cast<int>();
                int i_orig = i;
                if (i < 0) i += self->shape[d];
                if (i < 0 || i >= self->shape[d])
                    throw std::out_of_range("Index " + std::to_string(i_orig) +
                                             " out of range on dimension " + std::to_string(d) +
                                             " (size " + std::to_string(self->shape[d]) + ")");
                offset += i * self->strides[d];
                specs[d] = {true, i, 0, 0};
            } else if (py::isinstance<py::slice>(sel)) {
                py::slice s = sel.cast<py::slice>();
                size_t start, stop, step, slicelength;
                if (!s.compute((size_t)self->shape[d], &start, &stop, &step, &slicelength))
                    throw std::runtime_error("Invalid slice on dimension " + std::to_string(d));
                offset += (int)start * self->strides[d];
                new_shape.push_back((int)slicelength);
                new_strides.push_back(self->strides[d] * (int)step);
                specs[d] = {false, 0, (int)start, (int)step};
            } else {
                throw std::runtime_error("Index must be int or slice, got " +
                                          py::str(sel.get_type()).cast<std::string>());
            }
        } else {
            // Dimension not indexed at all: implicit full slice, kept as-is.
            new_shape.push_back(self->shape[d]);
            new_strides.push_back(self->strides[d]);
            specs[d] = {false, 0, 0, 1};
        }
    }

    // Zero-copy view in all cases (including full-int indexing, which now
    // yields a 0-d Tensor rather than a raw float — see note above).
    auto result = std::make_shared<Tensor>(self, offset, new_shape, new_strides);

    if (g_grad_enabled && self->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {self};
        node->op_name = "getitem";
        auto orig_shape = self->shape;
        auto orig_device = self->device;
        node->backward_fn = [orig_shape, orig_device, specs, new_shape](std::shared_ptr<Tensor> grad_out) {
            // Scatter grad_out back into a zero tensor shaped like the
            // original (pre-indexing) tensor. Every original element not
            // covered by this indexing op correctly receives zero gradient.
            auto grad_input = std::make_shared<Tensor>(orig_shape, orig_device);
            grad_input->fill_zero();

            size_t ndim_orig = orig_shape.size();
            size_t ndim_new = new_shape.size();

            int total_new = 1;
            for (int s : new_shape) total_new *= s;  // == 1 if new_shape is empty (full-int case)

            std::vector<int> new_idx(ndim_new, 0);
            for (int flat = 0; flat < total_new; ++flat) {
                float val = grad_out->get_scalar(new_idx);

                std::vector<int> orig_idx(ndim_orig);
                size_t j = 0;  // walks new_idx in lockstep with the non-int original dims
                for (size_t d = 0; d < ndim_orig; ++d) {
                    if (specs[d].is_int) {
                        orig_idx[d] = specs[d].int_index;
                    } else {
                        orig_idx[d] = specs[d].start + new_idx[j] * specs[d].step;
                        ++j;
                    }
                }
                grad_input->set_scalar(orig_idx, val);

                for (int d = (int)ndim_new - 1; d >= 0; --d) {
                    if (++new_idx[d] < new_shape[d]) break;
                    new_idx[d] = 0;
                }
            }

            return std::vector<std::shared_ptr<Tensor>>{grad_input};
        };
        result->grad_fn = node;
    }

    return py::cast(result);
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

    m.def("_set_grad_enabled", [](bool enabled) { g_grad_enabled = enabled; });
    m.def("_is_grad_enabled", []() { return g_grad_enabled; });

    m.def("fill_cpu_random", &fill_cpu_random, "Fill CPU Tensor with random numbers");
    m.def("cpu_matmul", &dispatch_matmul, "CPU matrix multiplication (autograd-aware)");
    m.def("cpu_add", &dispatch_add);
    m.def("cpu_sub", &dispatch_sub);
    m.def("from_numpy", &tensor_from_numpy,
          py::arg("array"), py::arg("device") = "cpu", py::arg("requires_grad") = false,
          "Create a Tensor from an existing numpy array, copying its data.");

#ifndef AAKAAR_NO_CUDA
    m.def("generate_random", &run_curand_uniform, "Fill GPU Tensor with random numbers");
    m.def("cuda_matmul", &dispatch_matmul, "cuBLAS GPU matrix multiplication");
    m.def("empty_cache", &empty_cache, "Release cached GPU memory");
    m.attr("HAS_CUDA") = true;
#else
    m.attr("HAS_CUDA") = false;
#endif
}