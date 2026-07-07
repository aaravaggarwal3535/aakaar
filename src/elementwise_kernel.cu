#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include "tensor.h"

enum class ElementwiseOp { ADD, SUB, MUL, DIV };

// ---- Tensor op Tensor ----
// Add these broadcast-aware kernels alongside your existing vectorized ones.
// Non-vectorized (no float4) since broadcast index math varies per-element;
// correctness-first, can optimize later if profiling shows this matters.

__device__ __forceinline__ int broadcast_index(int out_idx, const int* out_shape,
                                                 const int* in_strides, const int* in_shape,
                                                 int ndim) {
    int idx = 0;
    int remaining = out_idx;
    for (int d = ndim - 1; d >= 0; --d) {
        int coord = remaining % out_shape[d];
        remaining /= out_shape[d];
        int in_coord = (in_shape[d] == 1) ? 0 : coord;
        idx += in_coord * in_strides[d];
    }
    return idx;
}

template <ElementwiseOp OP>
__global__ void broadcast_elementwise_kernel(const float* __restrict__ a, const int* a_shape, const int* a_strides,
                                              const float* __restrict__ b, const int* b_shape, const int* b_strides,
                                              float* __restrict__ c, const int* out_shape,
                                              int ndim, int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int a_off = broadcast_index(idx, out_shape, a_strides, a_shape, ndim);
    int b_off = broadcast_index(idx, out_shape, b_strides, b_shape, ndim);

    float av = a[a_off], bv = b[b_off];
    float result;
    if constexpr (OP == ElementwiseOp::ADD) result = av + bv;
    else if constexpr (OP == ElementwiseOp::SUB) result = av - bv;
    else if constexpr (OP == ElementwiseOp::MUL) result = av * bv;
    else result = av / bv;
    c[idx] = result;
}

// Computes right-aligned broadcast output shape + per-input padded shape/strides.
// Returns false if shapes are not broadcast-compatible.
static bool compute_broadcast_plan(const std::vector<int>& sa, const std::vector<int>& sta,
                                    const std::vector<int>& sb, const std::vector<int>& stb,
                                    std::vector<int>& out_shape,
                                    std::vector<int>& pa_shape, std::vector<int>& pa_strides,
                                    std::vector<int>& pb_shape, std::vector<int>& pb_strides) {
    int nd = std::max(sa.size(), sb.size());
    out_shape.resize(nd); pa_shape.resize(nd); pa_strides.resize(nd);
    pb_shape.resize(nd); pb_strides.resize(nd);
    for (int i = 0; i < nd; ++i) {
        int ai = (int)sa.size() - nd + i;
        int bi = (int)sb.size() - nd + i;
        int da = ai >= 0 ? sa[ai] : 1;
        int db = bi >= 0 ? sb[bi] : 1;
        if (da != db && da != 1 && db != 1) return false;
        out_shape[i] = std::max(da, db);
        pa_shape[i] = da; pa_strides[i] = ai >= 0 ? sta[ai] : 0;
        pb_shape[i] = db; pb_strides[i] = bi >= 0 ? stb[bi] : 0;
    }
    return true;
}

template <ElementwiseOp OP>
static std::shared_ptr<Tensor> run_cuda_broadcast_elementwise(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::vector<int> out_shape, pa_shape, pa_strides, pb_shape, pb_strides;
    if (!compute_broadcast_plan(a->shape, a->strides, b->shape, b->strides,
                                 out_shape, pa_shape, pa_strides, pb_shape, pb_strides))
        throw std::invalid_argument("Shapes are not broadcastable for elementwise op: " +
                                     a->shape_str() + " vs " + b->shape_str());

    auto result = std::make_shared<Tensor>(out_shape, std::string("cuda"));
    int out_size = result->size;
    int ndim = (int)out_shape.size();

    int *d_out_shape, *d_a_shape, *d_a_strides, *d_b_shape, *d_b_strides;
    cudaMalloc(&d_out_shape, ndim * sizeof(int));
    cudaMalloc(&d_a_shape, ndim * sizeof(int));
    cudaMalloc(&d_a_strides, ndim * sizeof(int));
    cudaMalloc(&d_b_shape, ndim * sizeof(int));
    cudaMalloc(&d_b_strides, ndim * sizeof(int));
    cudaMemcpy(d_out_shape, out_shape.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, pa_shape.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, pa_strides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, pb_shape.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, pb_strides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (out_size + threads - 1) / threads;
    broadcast_elementwise_kernel<OP><<<blocks, threads>>>(
        a->data_ptr, d_a_shape, d_a_strides,
        b->data_ptr, d_b_shape, d_b_strides,
        result->data_ptr, d_out_shape, ndim, out_size
    );
    cudaError_t err = cudaGetLastError();
    cudaFree(d_out_shape); cudaFree(d_a_shape); cudaFree(d_a_strides); cudaFree(d_b_shape); cudaFree(d_b_strides);
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    cudaDeviceSynchronize();
    return result;
}

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

std::shared_ptr<Tensor> run_cuda_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { if (a->shape != b->shape) return run_cuda_broadcast_elementwise<ElementwiseOp::ADD>(a, b);
    return run_cuda_elementwise<ElementwiseOp::ADD>(a, b); }
std::shared_ptr<Tensor> run_cuda_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { if (a->shape != b->shape) return run_cuda_broadcast_elementwise<ElementwiseOp::SUB>(a, b);
    return run_cuda_elementwise<ElementwiseOp::SUB>(a, b); }
std::shared_ptr<Tensor> run_cuda_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { if (a->shape != b->shape) return run_cuda_broadcast_elementwise<ElementwiseOp::MUL>(a, b);
    return run_cuda_elementwise<ElementwiseOp::MUL>(a, b); }
std::shared_ptr<Tensor> run_cuda_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { if (a->shape != b->shape) return run_cuda_broadcast_elementwise<ElementwiseOp::DIV>(a, b);
    return run_cuda_elementwise<ElementwiseOp::DIV>(a, b); }

std::shared_ptr<Tensor> run_cuda_add_scalar(std::shared_ptr<Tensor> a, float s) { return run_cuda_scalar<ElementwiseOp::ADD>(a, s); }
std::shared_ptr<Tensor> run_cuda_sub_scalar(std::shared_ptr<Tensor> a, float s) { return run_cuda_scalar<ElementwiseOp::SUB>(a, s); }
std::shared_ptr<Tensor> run_cuda_mul_scalar(std::shared_ptr<Tensor> a, float s) { return run_cuda_scalar<ElementwiseOp::MUL>(a, s); }
std::shared_ptr<Tensor> run_cuda_div_scalar(std::shared_ptr<Tensor> a, float s) { return run_cuda_scalar<ElementwiseOp::DIV>(a, s); }