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
void set_tf32_enabled(bool enabled);
bool get_tf32_enabled();
void set_cudnn_tf32_enabled(bool enabled);
bool get_cudnn_tf32_enabled();
// ---- Forward declarations: raw ops ----
void fill_cpu_random(std::shared_ptr<Tensor> t, unsigned long long seed);
void fill_cpu_randint(std::shared_ptr<Tensor> t, long long low, long long high, unsigned long long seed);
std::shared_ptr<Tensor> run_cpu_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_add_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_sub_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_mul_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_div_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_add_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cpu_sub_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cpu_mul_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cpu_div_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cpu_sum_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cpu_sum_all(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_sum_axis_typed(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cpu_sum_all_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_broadcast_axis(std::shared_ptr<Tensor> a, int dim, int target_size);
std::shared_ptr<Tensor> run_cpu_relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cpu_sigmoid(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_sigmoid_backward(std::shared_ptr<Tensor> grad_out, const float* sig_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cpu_tanh(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_tanh_backward(std::shared_ptr<Tensor> grad_out, const float* tanh_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cpu_leaky_relu(std::shared_ptr<Tensor> a, float slope);
std::shared_ptr<Tensor> run_cpu_leaky_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, float slope);
std::shared_ptr<Tensor> run_cpu_exp(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_exp_backward(std::shared_ptr<Tensor> grad_out, const float* exp_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cpu_log(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_log_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cpu_max_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cpu_max_axis_backward(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                   std::vector<int> orig_shape, int dim, int reduce_size, int inner_size);
std::string get_openblas_diagnostic();
std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cpu_max_axis_typed(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cpu_max_axis_backward_typed(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                         std::vector<int> orig_shape, int dim, int reduce_size, int inner_size);
std::shared_ptr<Tensor> run_cpu_relu_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_relu_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cpu_add_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cpu_sub_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cpu_mul_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cpu_div_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cpu_leaky_relu_typed(std::shared_ptr<Tensor> a, double slope);
std::shared_ptr<Tensor> run_cpu_leaky_relu_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, double slope);
std::shared_ptr<Tensor> run_cpu_matmul_f64(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_matmul_int_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_sqrt(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_sqrt_backward(std::shared_ptr<Tensor> grad_out, const float* sqrt_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cpu_sqrt_f64(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_abs(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_abs_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cpu_abs_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_abs_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cpu_sign(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_sign_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_cross_entropy_per_row(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot);
std::shared_ptr<Tensor> run_cpu_cross_entropy_backward(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot, float grad_scale);
float run_cpu_huber_loss_forward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float delta);
std::shared_ptr<Tensor> run_cpu_huber_loss_backward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float delta, float grad_scale);

#ifndef AAKAAR_NO_CUDA
#include "allocator.h"
void run_curand_uniform(std::shared_ptr<Tensor> t, unsigned long long seed);
void run_curand_randint(std::shared_ptr<Tensor> t, long long low, long long high, unsigned long long seed);
void cuda_synchronize();
std::shared_ptr<Tensor> run_cublas_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_add_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_sub_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_mul_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_div_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_add_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cuda_sub_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cuda_mul_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cuda_div_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cuda_sum_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cuda_sum_all(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_sum_axis_typed(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cuda_sum_all_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_broadcast_axis(std::shared_ptr<Tensor> a, int dim, int target_size);
std::shared_ptr<Tensor> run_cuda_relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cuda_sigmoid(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_sigmoid_backward(std::shared_ptr<Tensor> grad_out, const float* sig_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cuda_tanh(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_tanh_backward(std::shared_ptr<Tensor> grad_out, const float* tanh_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cuda_leaky_relu(std::shared_ptr<Tensor> a, float slope);
std::shared_ptr<Tensor> run_cuda_leaky_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, float slope);
std::shared_ptr<Tensor> run_cuda_exp(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_exp_backward(std::shared_ptr<Tensor> grad_out, const float* exp_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cuda_log(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_log_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cuda_max_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cuda_max_axis_typed(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cuda_max_axis_backward_typed(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                          std::vector<int> orig_shape, int dim, int reduce_size, int inner_size);
std::shared_ptr<Tensor> run_cuda_max_axis_backward(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                    std::vector<int> orig_shape, int dim, int reduce_size, int inner_size);
std::shared_ptr<Tensor> run_cuda_relu_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_relu_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cuda_add_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cuda_sub_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cuda_mul_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cuda_div_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cuda_leaky_relu_typed(std::shared_ptr<Tensor> a, double slope);
std::shared_ptr<Tensor> run_cuda_leaky_relu_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, double slope);
std::shared_ptr<Tensor> run_cuda_matmul_f64(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_matmul_int_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_sqrt(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_sqrt_backward(std::shared_ptr<Tensor> grad_out, const float* sqrt_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cuda_sqrt_f64(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_abs(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_abs_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cuda_abs_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_abs_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cuda_sign(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_sign_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cudnn_conv1d_forward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,
                                                   int stride, int padding, int dilation, int L_out);
std::shared_ptr<Tensor> run_cudnn_conv1d_backward_data(std::shared_ptr<Tensor> grad_y, std::shared_ptr<Tensor> w,
                                                         int B, int C_in, int L_in, int stride, int padding,
                                                         int dilation, int L_out);
std::shared_ptr<Tensor> run_cudnn_conv1d_backward_filter(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> grad_y,
                                                           int C_out, int K, int stride, int padding,
                                                           int dilation, int L_out);
std::pair<float, std::shared_ptr<Tensor>> run_cuda_huber_loss_forward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float delta);
std::shared_ptr<Tensor> run_cuda_huber_loss_backward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float delta, float grad_scale);
std::shared_ptr<Tensor> run_cuda_cross_entropy_per_row(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot);
std::shared_ptr<Tensor> run_cuda_cross_entropy_backward(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot, float grad_scale);
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
static std::shared_ptr<Tensor> dispatch_huber_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float delta);

// Add near your other includes/declarations
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

// ---- Dispatch implementations ----
static std::shared_ptr<Tensor> dispatch_leaky_relu(std::shared_ptr<Tensor> a, double slope) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                    "'. Only float32 currently supports gradients.");
        if (!a->is_contiguous()) a = dispatch_contiguous(a);
        std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") result = run_cuda_leaky_relu_typed(a, slope);
        else
#endif
        result = run_cpu_leaky_relu_typed(a, slope);
        return result;
    }
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_leaky_relu(a, (float)slope);
    else
#endif
    result = run_cpu_leaky_relu(a, (float)slope);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "leaky_relu";
        auto slope_f = (float)slope;
        node->backward_fn = [a, slope_f](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (a->device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_leaky_relu_backward(grad_out, a, slope_f)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_leaky_relu_backward(grad_out, a, slope_f)};
        };
        result->grad_fn = node;
    }
    return result;
}
static std::shared_ptr<Tensor> dispatch_max_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
        if (!a->is_contiguous()) a = dispatch_contiguous(a);

        int ndim = (int)a->shape.size();
        int norm_dim = dim < 0 ? dim + ndim : dim;

#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") { auto pr = run_cuda_max_axis_typed(a, dim, keepdim); return pr.first; }
#endif
        auto pr = run_cpu_max_axis_typed(a, dim, keepdim);
        return pr.first;
    }
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
        
        float* out_ptr = result->fptr();
        int out_size = result->size;
        auto out_shape = result->shape;
        auto out_device = result->device;
        
        node->backward_fn = [out_ptr, out_size, out_shape, out_device](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (out_device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_exp_backward(grad_out, out_ptr, out_size, out_shape)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_exp_backward(grad_out, out_ptr, out_size, out_shape)};
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
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_add_scalar_typed(a, (double)s);
#endif
        return run_cpu_add_scalar_typed(a, (double)s);
    }

    // --- THE MISSING COMPUTATION ---
    // We must declare and compute 'result' for FLOAT32 before doing autograd wiring
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") {
        // Use your specific float32 CUDA function here (or the typed one if they share it)
        result = run_cuda_add_scalar(a, s); 
    } else 
#endif
    {
        result = run_cpu_add_scalar(a, s);
    }
    // -------------------------------

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
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_sub_scalar_typed(a, (double)s);
#endif
        return run_cpu_sub_scalar_typed(a, (double)s);
    }

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
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_mul_scalar_typed(a, (double)s);
#endif
        return run_cpu_mul_scalar_typed(a, (double)s);
    }

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
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_div_scalar_typed(a, (double)s);
#endif
        return run_cpu_div_scalar_typed(a, (double)s);
    }

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
    if (a->dtype != DType::FLOAT32) {
        if (!a->is_contiguous()) a = dispatch_contiguous(a);
        std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") result = run_cuda_relu_typed(a);
        else
#endif
        result = run_cpu_relu_typed(a);

        if (g_grad_enabled && a->requires_grad) {
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
        }
        return result;
    }
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

static std::shared_ptr<Tensor> dispatch_sqrt(std::shared_ptr<Tensor> a) {
    if (a->dtype == DType::FLOAT64) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype 'float64'. Only float32 currently supports gradients.");
        if (!a->is_contiguous()) a = dispatch_contiguous(a);
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_sqrt_f64(a);
#endif
        return run_cpu_sqrt_f64(a);
    }
    if (a->dtype == DType::INT32 || a->dtype == DType::INT64) {
        throw std::runtime_error("sqrt(): dtype '" + dtype_name(a->dtype) + "' is not supported. Use a float dtype.");
    }
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sqrt(a);
    else
#endif
    result = run_cpu_sqrt(a);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "sqrt";
        float* out_ptr = result->fptr();
        int out_size = result->size;
        auto out_shape = result->shape;
        auto out_device = result->device;
        node->backward_fn = [out_ptr, out_size, out_shape, out_device](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (out_device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_sqrt_backward(grad_out, out_ptr, out_size, out_shape)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_sqrt_backward(grad_out, out_ptr, out_size, out_shape)};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_abs(std::shared_ptr<Tensor> a) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) + "'. Only float32 currently supports gradients.");
        if (!a->is_contiguous()) a = dispatch_contiguous(a);
        std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") result = run_cuda_abs_typed(a);
        else
#endif
        result = run_cpu_abs_typed(a);
        return result;
    }
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_abs(a);
    else
#endif
    result = run_cpu_abs(a);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "abs";
        node->backward_fn = [a](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (a->device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_abs_backward(grad_out, a)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_abs_backward(grad_out, a)};
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
        // Capture raw pointer/shape only — NOT `result` itself, which would
        // create a shared_ptr cycle (result->grad_fn->backward_fn->result)
        // that leaks memory forever since refcounting can't collect cycles.
        float* out_ptr = result->fptr();
        int out_size = result->size;
        auto out_shape = result->shape;
        auto out_device = result->device;
        node->backward_fn = [out_ptr, out_size, out_shape, out_device](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (out_device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_sigmoid_backward(grad_out, out_ptr, out_size, out_shape)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_sigmoid_backward(grad_out, out_ptr, out_size, out_shape)};
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
        
        float* out_ptr = result->fptr();
        int out_size = result->size;
        auto out_shape = result->shape;
        auto out_device = result->device;
        
        node->backward_fn = [out_ptr, out_size, out_shape, out_device](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (out_device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_tanh_backward(grad_out, out_ptr, out_size, out_shape)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_tanh_backward(grad_out, out_ptr, out_size, out_shape)};
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
            cudaMemcpy(grad_output->fptr(), &one, sizeof(float), cudaMemcpyHostToDevice);
        } else
#endif
        {
            grad_output->fptr()[0] = one;
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
        if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_sum_axis_typed(a, dim, keepdim);
#endif
        return run_cpu_sum_axis_typed(a, dim, keepdim);
    }
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
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_sum_all_typed(a);
#endif
        return run_cpu_sum_all_typed(a);
    }
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

static std::shared_ptr<Tensor> dispatch_im2col_1d(std::shared_ptr<Tensor> x, int kernel_size, int stride,
                                                    int padding, int dilation, int out_length) {
    if (x->shape.size() != 3)
        throw std::invalid_argument("im2col_1d: expected input of shape (batch, channels, length), got rank " +
                                     std::to_string(x->shape.size()));
    if (!x->is_contiguous()) x = dispatch_contiguous(x);

    int B = x->shape[0], C = x->shape[1], L_in = x->shape[2];
    std::vector<int> col_shape = {B, C * kernel_size, out_length};

    if (x->dtype != DType::FLOAT32) {
        if (g_grad_enabled && x->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(x->dtype) +
                                      "'. Only float32 currently supports gradients.");
        auto result = std::make_shared<Tensor>(col_shape, x->device, x->dtype);
#ifndef AAKAAR_NO_CUDA
        if (x->device == "cuda") {
            run_cuda_im2col_1d_typed(x, result, kernel_size, stride, padding, dilation, out_length);
            return result;
        }
#endif
        run_cpu_im2col_1d_typed(x, result, kernel_size, stride, padding, dilation, out_length);
        return result;
    }

    auto result = std::make_shared<Tensor>(col_shape, x->device);
#ifndef AAKAAR_NO_CUDA
    if (x->device == "cuda") run_cuda_im2col_1d(x, result, kernel_size, stride, padding, dilation, out_length);
    else
#endif
    run_cpu_im2col_1d(x, result, kernel_size, stride, padding, dilation, out_length);

    if (g_grad_enabled && x->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {x};
        node->op_name = "im2col_1d";
        int B_c = B, C_c = C, L_in_c = L_in, K_c = kernel_size, S_c = stride, P_c = padding, D_c = dilation, Lo_c = out_length;
        auto dev = x->device;
        node->backward_fn = [B_c, C_c, L_in_c, K_c, S_c, P_c, D_c, Lo_c, dev](std::shared_ptr<Tensor> grad_out) {
            auto grad_x = std::make_shared<Tensor>(std::vector<int>{B_c, C_c, L_in_c}, dev);
#ifndef AAKAAR_NO_CUDA
            if (dev == "cuda") run_cuda_col2im_1d(grad_out, grad_x, B_c, C_c, L_in_c, K_c, S_c, P_c, D_c, Lo_c);
            else
#endif
            run_cpu_col2im_1d(grad_out, grad_x, B_c, C_c, L_in_c, K_c, S_c, P_c, D_c, Lo_c);
            return std::vector<std::shared_ptr<Tensor>>{grad_x};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_conv1d_cudnn(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,
                                                       int stride, int padding, int dilation) {
#ifdef AAKAAR_NO_CUDA
    throw std::runtime_error("conv1d_cudnn: this build of aakaar has no CUDA/cuDNN support.");
#else
    if (x->device != "cuda" || w->device != "cuda")
        throw std::runtime_error("conv1d_cudnn: both input and weight must be on device='cuda'.");
    if (x->dtype != DType::FLOAT32 || w->dtype != DType::FLOAT32)
        throw std::runtime_error(
            "conv1d_cudnn: only float32 is supported. cuDNN's double-precision conv support is "
            "unreliable/slow across versions and it has no integer conv path at all — aakaar "
            "intentionally does not route those dtypes here. Use conv1d_im2col (matmul-based) instead.");
    if (!x->is_contiguous()) x = dispatch_contiguous(x);
    if (!w->is_contiguous()) w = dispatch_contiguous(w);

    int B = x->shape[0], C_in = x->shape[1], L_in = x->shape[2];
    int C_out = w->shape[0], K = w->shape[2];
    int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    if (L_out <= 0)
        throw std::invalid_argument("conv1d_cudnn: computed output length <= 0.");

    auto y = run_cudnn_conv1d_forward(x, w, stride, padding, dilation, L_out);

    if (g_grad_enabled && (x->requires_grad || w->requires_grad)) {
        y->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {x, w};
        node->op_name = "conv1d_cudnn";
        int B_c = B, Cin_c = C_in, Lin_c = L_in, Cout_c = C_out, K_c = K, S_c = stride, P_c = padding, D_c = dilation, Lo_c = L_out;
        node->backward_fn = [x, w, B_c, Cin_c, Lin_c, Cout_c, K_c, S_c, P_c, D_c, Lo_c]
                             (std::shared_ptr<Tensor> grad_out) {
            auto grad_x = run_cudnn_conv1d_backward_data(grad_out, w, B_c, Cin_c, Lin_c, S_c, P_c, D_c, Lo_c);
            auto grad_w = run_cudnn_conv1d_backward_filter(x, grad_out, Cout_c, K_c, S_c, P_c, D_c, Lo_c);
            return std::vector<std::shared_ptr<Tensor>>{grad_x, grad_w};
        };
        y->grad_fn = node;
    }
    return y;
#endif
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
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && (a->requires_grad || b->requires_grad))
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_add_typed(a, b);
#endif
        return run_cpu_add_typed(a, b);
    }

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
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && (a->requires_grad || b->requires_grad))
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_sub_typed(a, b);
#endif
        return run_cpu_sub_typed(a, b);
    }

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
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && (a->requires_grad || b->requires_grad))
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_mul_typed(a, b);
#endif
        return run_cpu_mul_typed(a, b);
    }

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
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && (a->requires_grad || b->requires_grad))
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_div_typed(a, b);
#endif
        return run_cpu_div_typed(a, b);
    }

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
    // 1. Float64 Execution Path (No Autograd Support)
    if (a->dtype == DType::FLOAT64) {
        if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
            throw std::runtime_error("Autograd is not yet supported for dtype 'float64'. "
                                     "Only float32 currently supports gradients.");
        }
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_matmul_f64(a, b);
#endif
        return run_cpu_matmul_f64(a, b);
    }

    // 2. Updated Integer Execution Path (No Autograd Support)
    if (a->dtype == DType::INT32 || a->dtype == DType::INT64) {
        if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
            throw std::runtime_error("Autograd is not yet supported for dtype '" + 
                                     dtype_name(a->dtype) + "'. Only float32 currently supports gradients.");
        }
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_matmul_int_typed(a, b);
#endif
        return run_cpu_matmul_int_typed(a, b);
    }

    // 3. Existing Float32 Execution Path
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda")
        result = run_cublas_matmul(a, b);
    else
#endif
        result = run_cpu_matmul(a, b);

    // 4. Float32 Autograd Graph Construction
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

static std::shared_ptr<Tensor> dispatch_sign(std::shared_ptr<Tensor> a) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
        if (!a->is_contiguous()) a = dispatch_contiguous(a);
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_sign_typed(a);
#endif
        return run_cpu_sign_typed(a);
    }
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sign(a);
    else
#endif
    result = run_cpu_sign(a);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "sign";
        node->backward_fn = [](std::shared_ptr<Tensor> grad_out) {
            // sign's derivative is zero almost everywhere — matches torch's
            // convention of returning zero gradient, not an error.
            auto zero_grad = dispatch_mul_scalar(grad_out, 0.0f);
            return std::vector<std::shared_ptr<Tensor>>{zero_grad};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_huber_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float delta) {
    if (pred->dtype != DType::FLOAT32 || target->dtype != DType::FLOAT32)
        throw std::runtime_error("huber_loss(): only float32 is currently supported by the fused kernel.");
    if (!pred->is_contiguous()) pred = dispatch_contiguous(pred);
    if (!target->is_contiguous()) target = dispatch_contiguous(target);

    float total;
#ifndef AAKAAR_NO_CUDA
    if (pred->device == "cuda") {
        auto r = run_cuda_huber_loss_forward(pred, target, delta);
        total = r.first;
    } else
#endif
    total = run_cpu_huber_loss_forward(pred, target, delta);

    int n = pred->size;
    float mean_loss = total / n;

    auto result = std::make_shared<Tensor>(std::vector<int>{1}, pred->device);
#ifndef AAKAAR_NO_CUDA
    if (pred->device == "cuda") cudaMemcpy(result->fptr(), &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
    else
#endif
    result->fptr()[0] = mean_loss;

    if (g_grad_enabled && (pred->requires_grad || target->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {pred, target};
        node->op_name = "huber_loss";
        node->backward_fn = [pred, target, delta, n](std::shared_ptr<Tensor> grad_out) {
            float grad_out_scalar;
#ifndef AAKAAR_NO_CUDA
            if (grad_out->device == "cuda") cudaMemcpy(&grad_out_scalar, grad_out->fptr(), sizeof(float), cudaMemcpyDeviceToHost);
            else
#endif
            grad_out_scalar = grad_out->fptr()[0];

            float grad_scale = grad_out_scalar / n;
            std::shared_ptr<Tensor> grad_pred, grad_target;
#ifndef AAKAAR_NO_CUDA
            if (pred->device == "cuda") {
                grad_pred = run_cuda_huber_loss_backward(pred, target, delta, grad_scale);
                grad_target = run_cuda_huber_loss_backward(pred, target, delta, -grad_scale);
            } else
#endif
            {
                grad_pred = run_cpu_huber_loss_backward(pred, target, delta, grad_scale);
                grad_target = run_cpu_huber_loss_backward(pred, target, delta, -grad_scale);
            }
            return std::vector<std::shared_ptr<Tensor>>{grad_pred, grad_target};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_cross_entropy_fused(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot) {
    if (logits->dtype != DType::FLOAT32 || onehot->dtype != DType::FLOAT32)
        throw std::runtime_error("cross_entropy(): only float32 is currently supported by the fused kernel.");
    if (logits->shape.size() != 2)
        throw std::runtime_error("cross_entropy(): fused kernel currently requires 2D (batch, classes) input.");
    if (!logits->is_contiguous()) logits = dispatch_contiguous(logits);
    if (!onehot->is_contiguous()) onehot = dispatch_contiguous(onehot);

    std::shared_ptr<Tensor> per_row;
#ifndef AAKAAR_NO_CUDA
    if (logits->device == "cuda") per_row = run_cuda_cross_entropy_per_row(logits, onehot);
    else
#endif
    per_row = run_cpu_cross_entropy_per_row(logits, onehot);

    int N = logits->shape[0];
    float total = 0.0f;
    auto per_row_np = per_row->fptr();
#ifndef AAKAAR_NO_CUDA
    if (logits->device == "cuda") {
        std::vector<float> host_buf(N);
        cudaMemcpy(host_buf.data(), per_row->fptr(), N * sizeof(float), cudaMemcpyDeviceToHost);
        for (float v : host_buf) total += v;
    } else
#endif
    { for (int i = 0; i < N; ++i) total += per_row_np[i]; }

    float mean_loss = total / N;
    auto result = std::make_shared<Tensor>(std::vector<int>{1}, logits->device);
#ifndef AAKAAR_NO_CUDA
    if (logits->device == "cuda") cudaMemcpy(result->fptr(), &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
    else
#endif
    result->fptr()[0] = mean_loss;

    if (g_grad_enabled && (logits->requires_grad || onehot->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {logits, onehot};
        node->op_name = "cross_entropy_fused";
        node->backward_fn = [logits, onehot, N](std::shared_ptr<Tensor> grad_out) {
            float grad_out_scalar;
#ifndef AAKAAR_NO_CUDA
            if (grad_out->device == "cuda") cudaMemcpy(&grad_out_scalar, grad_out->fptr(), sizeof(float), cudaMemcpyDeviceToHost);
            else
#endif
            grad_out_scalar = grad_out->fptr()[0];
            float grad_scale = grad_out_scalar / N;

            std::shared_ptr<Tensor> grad_logits;
#ifndef AAKAAR_NO_CUDA
            if (logits->device == "cuda") grad_logits = run_cuda_cross_entropy_backward(logits, onehot, grad_scale);
            else
#endif
            grad_logits = run_cpu_cross_entropy_backward(logits, onehot, grad_scale);

            // No meaningful gradient w.r.t. the one-hot target in this
            // formulation (it's a fixed label, not a learned input) —
            // return a zero tensor of matching shape for it.
            auto zero_target_grad = std::make_shared<Tensor>(onehot->shape, onehot->device);
            zero_target_grad->fill_zero();
            return std::vector<std::shared_ptr<Tensor>>{grad_logits, zero_target_grad};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> tensor_from_numpy_typed(py::array arr, std::string device, bool requires_grad) {
    py::buffer_info buf = arr.request();
    std::vector<int> shape;
    for (auto d : buf.shape) shape.push_back((int)d);
    if (shape.empty()) shape.push_back(1);
    py::dtype np_dtype = arr.dtype();
    char kind = np_dtype.kind();
    py::ssize_t itemsize = np_dtype.itemsize();

    DType dt;
    if (kind == 'f' && itemsize == 4) dt = DType::FLOAT32;
    else if (kind == 'f' && itemsize == 8) dt = DType::FLOAT64;
    else if (kind == 'i' && itemsize == 4) dt = DType::INT32;
    else if (kind == 'i' && itemsize == 8) dt = DType::INT64;
    else throw std::runtime_error("from_numpy(): unsupported numpy dtype (kind='" +
                                   std::string(1, kind) + "', itemsize=" + std::to_string(itemsize) +
                                   "). Supported: float32, float64, int32, int64.");

    auto result = std::make_shared<Tensor>(shape, device, dt);
    size_t bytes = (size_t)result->size * dtype_size(dt);

    py::array contiguous_arr = py::array::ensure(arr, py::array::c_style | py::array::forcecast);
    py::buffer_info cbuf = contiguous_arr.request();

#ifndef AAKAAR_NO_CUDA
    if (device == "cuda") {
        cudaMemcpy(result->data_ptr, cbuf.ptr, bytes, cudaMemcpyHostToDevice);
        result->requires_grad = requires_grad;
        return result;
    }
#endif
    std::memcpy(result->data_ptr, cbuf.ptr, bytes);
    result->requires_grad = requires_grad;
    return result;
}
// ---- Module definition ----

PYBIND11_MODULE(_C, m) {
    py::enum_<DType>(m, "DType")
        .value("float32", DType::FLOAT32)
        .value("float64", DType::FLOAT64)
        .value("int32", DType::INT32)
        .value("int64", DType::INT64);

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<int>, std::string>())
        .def(py::init<std::vector<int>, std::string, DType>())  
        .def_property_readonly("dtype", [](Tensor &t) { return dtype_name(t.dtype); })
        .def_readonly("device", &Tensor::device)
        .def_readonly("size", &Tensor::size)
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("strides", &Tensor::strides)
        .def_readwrite("requires_grad", &Tensor::requires_grad)
        .def_property("grad",
            [](Tensor &t) { return t.grad; },
            [](Tensor &t, std::shared_ptr<Tensor> g) { t.grad = g; }
        )
        .def("to_numpy", &Tensor::to_numpy)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("relu", [](std::shared_ptr<Tensor> self) { return dispatch_relu(self); })
        .def("sigmoid", [](std::shared_ptr<Tensor> self) { return dispatch_sigmoid(self); })
        .def("tanh", [](std::shared_ptr<Tensor> self) { return dispatch_tanh(self); })
        .def("copy_", &Tensor::copy_)
        .def("leaky_relu", [](std::shared_ptr<Tensor> self, double slope) { return dispatch_leaky_relu(self, slope); }, py::arg("slope") = 0.01)
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
        .def("to_device", [](std::shared_ptr<Tensor> self, std::string target_device) {
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
        .def("huber_loss", [](std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> target, float delta) {
            return dispatch_huber_loss(self, target, delta);
        }, py::arg("target"), py::arg("delta") = 1.0f)
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
        auto self_dtype = self->dtype;
        node->backward_fn = [orig_shape, orig_device, self_dtype, specs, new_shape](std::shared_ptr<Tensor> grad_out) {
            auto grad_input = std::make_shared<Tensor>(orig_shape, orig_device, self_dtype);
            grad_input->fill_zero_typed(self_dtype);

            size_t ndim_orig = orig_shape.size();
            size_t ndim_new = new_shape.size();

            int total_new = 1;
            for (int s : new_shape) total_new *= s;

            auto scatter = [&](auto tag) {
                using T = decltype(tag);
                std::vector<int> new_idx(ndim_new, 0);
                for (int flat = 0; flat < total_new; ++flat) {
                    T val = grad_out->get_scalar_typed<T>(new_idx);

                    std::vector<int> orig_idx(ndim_orig);
                    size_t j = 0;
                    for (size_t d = 0; d < ndim_orig; ++d) {
                        if (specs[d].is_int) {
                            orig_idx[d] = specs[d].int_index;
                        } else {
                            orig_idx[d] = specs[d].start + new_idx[j] * specs[d].step;
                            ++j;
                        }
                    }
                    grad_input->set_scalar_typed<T>(orig_idx, val);

                    for (int d = (int)ndim_new - 1; d >= 0; --d) {
                        if (++new_idx[d] < new_shape[d]) break;
                        new_idx[d] = 0;
                    }
                }
            };
            switch (self_dtype) {
                case DType::FLOAT32: scatter(float{});   break;
                case DType::FLOAT64: scatter(double{});  break;
                case DType::INT32:   scatter(int32_t{}); break;
                case DType::INT64:   scatter(int64_t{}); break;
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
        .def("__rsub__", [](std::shared_ptr<Tensor> a, float s) {
            auto neg_a = dispatch_mul_scalar(a, -1.0f);
            return dispatch_add_scalar(neg_a, s);
        })
        .def("__mul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_mul(a, b); })
        .def("__mul__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_mul_scalar(a, s); })
        .def("__rmul__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_mul_scalar(a, s); })
        .def("__truediv__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_div(a, b); })
        .def("__truediv__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_div_scalar(a, s); })
        .def("__rtruediv__", [](std::shared_ptr<Tensor> a, float s) {
            auto s_tensor = dispatch_add_scalar(dispatch_mul_scalar(a, 0.0f), s);
            return dispatch_div(s_tensor, a);
        })
        .def("__matmul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_matmul(a, b); })
        .def("sqrt", [](std::shared_ptr<Tensor> self) { return dispatch_sqrt(self); })
        .def("sign", [](std::shared_ptr<Tensor> self) { return dispatch_sign(self); })
        .def("abs", [](std::shared_ptr<Tensor> self) { return dispatch_abs(self); });

    m.def("_set_grad_enabled", [](bool enabled) { g_grad_enabled = enabled; });
    m.def("_is_grad_enabled", []() { return g_grad_enabled; });

    m.def("fill_cpu_random", &fill_cpu_random, "Fill CPU Tensor with random numbers");
    m.def("fill_cpu_randint", &fill_cpu_randint, "Fill CPU Tensor with random integers");
    m.def("cpu_matmul", &dispatch_matmul, "CPU matrix multiplication (autograd-aware)");
    m.def("cpu_add", &dispatch_add);
    m.def("cpu_sub", &dispatch_sub);
    m.def("from_numpy", &tensor_from_numpy_typed,
      py::arg("array"), py::arg("device") = "cpu", py::arg("requires_grad") = false,
      "Create a Tensor from an existing numpy array, preserving its dtype (float32/float64/int32/int64).");
    m.def("is_available", &cuda_is_available, "Check if a CUDA-capable GPU is actually present and usable");
    m.def("device_count", &cuda_device_count, "Number of CUDA-capable GPUs detected");
    
    #ifndef AAKAAR_NO_CUDA
    m.def("generate_random", &run_curand_uniform, "Fill GPU Tensor with random numbers");
    m.def("generate_randint", &run_curand_randint);
    m.def("cuda_matmul", &dispatch_matmul, "cuBLAS GPU matrix multiplication");
    m.def("empty_cache", &empty_cache, "Release cached GPU memory");
    m.def("_set_tf32_enabled", &set_tf32_enabled);
    m.def("_get_tf32_enabled", &get_tf32_enabled);
    m.def("_synchronize", &cuda_synchronize);
    m.def("im2col_1d", &dispatch_im2col_1d,
          py::arg("x"), py::arg("kernel_size"), py::arg("stride"),
          py::arg("padding"), py::arg("dilation"), py::arg("out_length"));
    m.def("conv1d_cudnn", &dispatch_conv1d_cudnn,
          py::arg("x"), py::arg("w"), py::arg("stride"), py::arg("padding"), py::arg("dilation"));
    m.def("_huber_loss_fused", &dispatch_huber_loss, py::arg("pred"), py::arg("target"), py::arg("delta") = 1.0f);
    m.def("_cross_entropy_fused", &dispatch_cross_entropy_fused, py::arg("logits"), py::arg("onehot"));
    m.def("_set_cudnn_tf32_enabled", &set_cudnn_tf32_enabled);
    m.def("_get_cudnn_tf32_enabled", &get_cudnn_tf32_enabled);
    m.def("_allocator_stats", []() -> py::tuple {
    auto [hits, misses] = CachingAllocator::get_instance().get_stats();
    return py::make_tuple(hits, misses);
});
    m.attr("HAS_CUDA") = true;
#else
    m.attr("HAS_CUDA") = false;
#endif
    m.def("im2col_1d", &dispatch_im2col_1d,
          py::arg("x"), py::arg("kernel_size"), py::arg("stride"),
          py::arg("padding"), py::arg("dilation"), py::arg("out_length"));
}