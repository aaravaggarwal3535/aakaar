#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include "tensor.h"

enum class ElementwiseOp { ADD, SUB, MUL, DIV };

// ---- Tensor op Tensor ----

template <ElementwiseOp OP>
__global__ void vector_elementwise_vectorized(const float* __restrict__ a,
                                               const float* __restrict__ b,
                                               float* __restrict__ c,
                                               int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n_vec = n / 4;

    const float4* a_vec = reinterpret_cast<const float4*>(a);
    const float4* b_vec = reinterpret_cast<const float4*>(b);
    float4* c_vec = reinterpret_cast<float4*>(c);

    auto apply = [](float x, float y) -> float {
        if constexpr (OP == ElementwiseOp::ADD) return x + y;
        else if constexpr (OP == ElementwiseOp::SUB) return x - y;
        else if constexpr (OP == ElementwiseOp::MUL) return x * y;
        else return x / y;
    };

    for (int i = tid; i < n_vec; i += stride) {
        float4 a_val = a_vec[i];
        float4 b_val = b_vec[i];
        float4 c_val;
        c_val.x = apply(a_val.x, b_val.x);
        c_val.y = apply(a_val.y, b_val.y);
        c_val.z = apply(a_val.z, b_val.z);
        c_val.w = apply(a_val.w, b_val.w);
        c_vec[i] = c_val;
    }

    int tail_start = n_vec * 4;
    for (int i = tail_start + tid; i < n; i += stride) {
        c[i] = apply(a[i], b[i]);
    }
}

// ---- Tensor op Scalar ----

template <ElementwiseOp OP>
__global__ void scalar_elementwise_vectorized(const float* __restrict__ a,
                                               float scalar,
                                               float* __restrict__ c,
                                               int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n_vec = n / 4;

    const float4* a_vec = reinterpret_cast<const float4*>(a);
    float4* c_vec = reinterpret_cast<float4*>(c);

    auto apply = [scalar](float x) -> float {
        if constexpr (OP == ElementwiseOp::ADD) return x + scalar;
        else if constexpr (OP == ElementwiseOp::SUB) return x - scalar;
        else if constexpr (OP == ElementwiseOp::MUL) return x * scalar;
        else return x / scalar;
    };

    for (int i = tid; i < n_vec; i += stride) {
        float4 a_val = a_vec[i];
        float4 c_val;
        c_val.x = apply(a_val.x);
        c_val.y = apply(a_val.y);
        c_val.z = apply(a_val.z);
        c_val.w = apply(a_val.w);
        c_vec[i] = c_val;
    }

    int tail_start = n_vec * 4;
    for (int i = tail_start + tid; i < n; i += stride) {
        c[i] = apply(a[i]);
    }
}

static void check_elementwise_inputs(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if (a->shape != b->shape) {
        throw std::invalid_argument("Shape mismatch in elementwise op: " +
                                     a->shape_str() + " vs " + b->shape_str());
    }
    if (!a->is_contiguous() || !b->is_contiguous()) {
        throw std::invalid_argument(
            "Elementwise CUDA op requires contiguous tensors. Call .contiguous() first."
        );
    }
}

static void check_scalar_input(const std::shared_ptr<Tensor>& a) {
    if (!a->is_contiguous()) {
        throw std::invalid_argument(
            "Elementwise CUDA op requires a contiguous tensor. Call .contiguous() first."
        );
    }
}

static int compute_blocks(int n) {
    int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return blocks;
}

template <ElementwiseOp OP>
static std::shared_ptr<Tensor> run_cuda_elementwise(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    check_elementwise_inputs(a, b);
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"));
    int n = a->size;
    vector_elementwise_vectorized<OP><<<compute_blocks(n), 256>>>(a->data_ptr, b->data_ptr, result->data_ptr, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    cudaDeviceSynchronize();
    return result;
}

template <ElementwiseOp OP>
static std::shared_ptr<Tensor> run_cuda_scalar(std::shared_ptr<Tensor> a, float scalar) {
    check_scalar_input(a);
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"));
    int n = a->size;
    scalar_elementwise_vectorized<OP><<<compute_blocks(n), 256>>>(a->data_ptr, scalar, result->data_ptr, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    cudaDeviceSynchronize();
    return result;
}

std::shared_ptr<Tensor> run_cuda_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return run_cuda_elementwise<ElementwiseOp::ADD>(a, b); }
std::shared_ptr<Tensor> run_cuda_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return run_cuda_elementwise<ElementwiseOp::SUB>(a, b); }
std::shared_ptr<Tensor> run_cuda_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return run_cuda_elementwise<ElementwiseOp::MUL>(a, b); }
std::shared_ptr<Tensor> run_cuda_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return run_cuda_elementwise<ElementwiseOp::DIV>(a, b); }

std::shared_ptr<Tensor> run_cuda_add_scalar(std::shared_ptr<Tensor> a, float s) { return run_cuda_scalar<ElementwiseOp::ADD>(a, s); }
std::shared_ptr<Tensor> run_cuda_sub_scalar(std::shared_ptr<Tensor> a, float s) { return run_cuda_scalar<ElementwiseOp::SUB>(a, s); }
std::shared_ptr<Tensor> run_cuda_mul_scalar(std::shared_ptr<Tensor> a, float s) { return run_cuda_scalar<ElementwiseOp::MUL>(a, s); }
std::shared_ptr<Tensor> run_cuda_div_scalar(std::shared_ptr<Tensor> a, float s) { return run_cuda_scalar<ElementwiseOp::DIV>(a, s); }