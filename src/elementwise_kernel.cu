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
// ============================================================================
// Elementwise activation kernels: relu, sigmoid, tanh (forward + backward)
// ============================================================================

#include <cstdint>

static inline bool is_aligned16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

static int elem_blocks(int n) {
    if (n <= 0) return 0;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    return blocks;
}

// ---------------------------------------------------------------------------
// RELU
// ---------------------------------------------------------------------------

__global__ void relu_forward_scalar_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float v = __ldg(&in[i]);
        out[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void relu_forward_vec4_kernel(const float4* __restrict__ in, float4* __restrict__ out, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 v = __ldg(&in[i]);
        float4 r;
        r.x = v.x > 0.0f ? v.x : 0.0f;
        r.y = v.y > 0.0f ? v.y : 0.0f;
        r.z = v.z > 0.0f ? v.z : 0.0f;
        r.w = v.w > 0.0f ? v.w : 0.0f;
        out[i] = r;
    }
}

__global__ void relu_backward_scalar_kernel(const float* __restrict__ grad_out, const float* __restrict__ in,
                                             float* __restrict__ grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float x = __ldg(&in[i]);
        float g = __ldg(&grad_out[i]);
        grad_in[i] = x > 0.0f ? g : 0.0f;
    }
}

__global__ void relu_backward_vec4_kernel(const float4* __restrict__ grad_out, const float4* __restrict__ in,
                                           float4* __restrict__ grad_in, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 x = __ldg(&in[i]);
        float4 g = __ldg(&grad_out[i]);
        float4 r;
        r.x = x.x > 0.0f ? g.x : 0.0f;
        r.y = x.y > 0.0f ? g.y : 0.0f;
        r.z = x.z > 0.0f ? g.z : 0.0f;
        r.w = x.w > 0.0f ? g.w : 0.0f;
        grad_in[i] = r;
    }
}

std::shared_ptr<Tensor> run_cuda_relu(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) throw std::invalid_argument("relu requires a contiguous tensor. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"));
    int n = a->size;
    if (n == 0) return result;
    bool can_vec = (n >= 4) && is_aligned16(a->data_ptr) && is_aligned16(result->data_ptr);
    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        relu_forward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(a->data_ptr),
            reinterpret_cast<float4*>(result->data_ptr), n4);
        if (tail > 0) {
            relu_forward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                a->data_ptr + n4 * 4, result->data_ptr + n4 * 4, tail);
        }
    } else {
        relu_forward_scalar_kernel<<<elem_blocks(n), 256>>>(a->data_ptr, result->data_ptr, n);
    }
    cudaDeviceSynchronize();
    return result;
}

std::shared_ptr<Tensor> run_cuda_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cuda"));
    int n = input->size;
    if (n == 0) return result;
    bool can_vec = (n >= 4) && is_aligned16(grad_out->data_ptr)
                            && is_aligned16(input->data_ptr)
                            && is_aligned16(result->data_ptr);
    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        relu_backward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(grad_out->data_ptr),
            reinterpret_cast<const float4*>(input->data_ptr),
            reinterpret_cast<float4*>(result->data_ptr), n4);
        if (tail > 0) {
            relu_backward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                grad_out->data_ptr + n4 * 4, input->data_ptr + n4 * 4,
                result->data_ptr + n4 * 4, tail);
        }
    } else {
        relu_backward_scalar_kernel<<<elem_blocks(n), 256>>>(
            grad_out->data_ptr, input->data_ptr, result->data_ptr, n);
    }
    cudaDeviceSynchronize();
    return result;
}

// ---------------------------------------------------------------------------
// SIGMOID
// ---------------------------------------------------------------------------

__global__ void sigmoid_forward_scalar_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float v = __ldg(&in[i]);
        out[i] = 1.0f / (1.0f + expf(-v));
    }
}

__global__ void sigmoid_forward_vec4_kernel(const float4* __restrict__ in, float4* __restrict__ out, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 v = __ldg(&in[i]);
        float4 r;
        r.x = 1.0f / (1.0f + expf(-v.x));
        r.y = 1.0f / (1.0f + expf(-v.y));
        r.z = 1.0f / (1.0f + expf(-v.z));
        r.w = 1.0f / (1.0f + expf(-v.w));
        out[i] = r;
    }
}

__global__ void sigmoid_backward_scalar_kernel(const float* __restrict__ grad_out, const float* __restrict__ sig_out,
                                                float* __restrict__ grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float s = __ldg(&sig_out[i]);
        float g = __ldg(&grad_out[i]);
        grad_in[i] = g * s * (1.0f - s);
    }
}

__global__ void sigmoid_backward_vec4_kernel(const float4* __restrict__ grad_out, const float4* __restrict__ sig_out,
                                              float4* __restrict__ grad_in, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 s = __ldg(&sig_out[i]);
        float4 g = __ldg(&grad_out[i]);
        float4 r;
        r.x = g.x * s.x * (1.0f - s.x);
        r.y = g.y * s.y * (1.0f - s.y);
        r.z = g.z * s.z * (1.0f - s.z);
        r.w = g.w * s.w * (1.0f - s.w);
        grad_in[i] = r;
    }
}

std::shared_ptr<Tensor> run_cuda_sigmoid(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) throw std::invalid_argument("sigmoid requires a contiguous tensor. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"));
    int n = a->size;
    if (n == 0) return result;
    bool can_vec = (n >= 4) && is_aligned16(a->data_ptr) && is_aligned16(result->data_ptr);
    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        sigmoid_forward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(a->data_ptr),
            reinterpret_cast<float4*>(result->data_ptr), n4);
        if (tail > 0) {
            sigmoid_forward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                a->data_ptr + n4 * 4, result->data_ptr + n4 * 4, tail);
        }
    } else {
        sigmoid_forward_scalar_kernel<<<elem_blocks(n), 256>>>(a->data_ptr, result->data_ptr, n);
    }
    cudaDeviceSynchronize();
    return result;
}

std::shared_ptr<Tensor> run_cuda_sigmoid_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> sig_output) {
    auto result = std::make_shared<Tensor>(sig_output->shape, std::string("cuda"));
    int n = sig_output->size;
    if (n == 0) return result;
    bool can_vec = (n >= 4) && is_aligned16(grad_out->data_ptr)
                            && is_aligned16(sig_output->data_ptr)
                            && is_aligned16(result->data_ptr);
    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        sigmoid_backward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(grad_out->data_ptr),
            reinterpret_cast<const float4*>(sig_output->data_ptr),
            reinterpret_cast<float4*>(result->data_ptr), n4);
        if (tail > 0) {
            sigmoid_backward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                grad_out->data_ptr + n4 * 4, sig_output->data_ptr + n4 * 4,
                result->data_ptr + n4 * 4, tail);
        }
    } else {
        sigmoid_backward_scalar_kernel<<<elem_blocks(n), 256>>>(
            grad_out->data_ptr, sig_output->data_ptr, result->data_ptr, n);
    }
    cudaDeviceSynchronize();
    return result;
}

// ---------------------------------------------------------------------------
// TANH
// ---------------------------------------------------------------------------

__global__ void tanh_forward_scalar_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = tanhf(__ldg(&in[i]));
    }
}

__global__ void tanh_forward_vec4_kernel(const float4* __restrict__ in, float4* __restrict__ out, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 v = __ldg(&in[i]);
        float4 r;
        r.x = tanhf(v.x);
        r.y = tanhf(v.y);
        r.z = tanhf(v.z);
        r.w = tanhf(v.w);
        out[i] = r;
    }
}

__global__ void tanh_backward_scalar_kernel(const float* __restrict__ grad_out, const float* __restrict__ tanh_out,
                                             float* __restrict__ grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float t = __ldg(&tanh_out[i]);
        float g = __ldg(&grad_out[i]);
        grad_in[i] = g * (1.0f - t * t);
    }
}

__global__ void tanh_backward_vec4_kernel(const float4* __restrict__ grad_out, const float4* __restrict__ tanh_out,
                                           float4* __restrict__ grad_in, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 t = __ldg(&tanh_out[i]);
        float4 g = __ldg(&grad_out[i]);
        float4 r;
        r.x = g.x * (1.0f - t.x * t.x);
        r.y = g.y * (1.0f - t.y * t.y);
        r.z = g.z * (1.0f - t.z * t.z);
        r.w = g.w * (1.0f - t.w * t.w);
        grad_in[i] = r;
    }
}

std::shared_ptr<Tensor> run_cuda_tanh(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) throw std::invalid_argument("tanh requires a contiguous tensor. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"));
    int n = a->size;
    if (n == 0) return result;
    bool can_vec = (n >= 4) && is_aligned16(a->data_ptr) && is_aligned16(result->data_ptr);
    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        tanh_forward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(a->data_ptr),
            reinterpret_cast<float4*>(result->data_ptr), n4);
        if (tail > 0) {
            tanh_forward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                a->data_ptr + n4 * 4, result->data_ptr + n4 * 4, tail);
        }
    } else {
        tanh_forward_scalar_kernel<<<elem_blocks(n), 256>>>(a->data_ptr, result->data_ptr, n);
    }
    cudaDeviceSynchronize();
    return result;
}

std::shared_ptr<Tensor> run_cuda_tanh_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> tanh_output) {
    auto result = std::make_shared<Tensor>(tanh_output->shape, std::string("cuda"));
    int n = tanh_output->size;
    if (n == 0) return result;
    bool can_vec = (n >= 4) && is_aligned16(grad_out->data_ptr)
                            && is_aligned16(tanh_output->data_ptr)
                            && is_aligned16(result->data_ptr);
    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        tanh_backward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(grad_out->data_ptr),
            reinterpret_cast<const float4*>(tanh_output->data_ptr),
            reinterpret_cast<float4*>(result->data_ptr), n4);
        if (tail > 0) {
            tanh_backward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                grad_out->data_ptr + n4 * 4, tanh_output->data_ptr + n4 * 4,
                result->data_ptr + n4 * 4, tail);
        }
    } else {
        tanh_backward_scalar_kernel<<<elem_blocks(n), 256>>>(
            grad_out->data_ptr, tanh_output->data_ptr, result->data_ptr, n);
    }
    cudaDeviceSynchronize();
    return result;
}

// ---------------------------------------------------------------------------
// LEAKY RELU
// Same optimization approach as relu: __ldg() global memory caching (already
// present), float4 vectorization with runtime 16-byte alignment guard (views/
// slices are NOT guaranteed float4-aligned even when "contiguous" in the
// stride sense — see is_aligned16/elem_blocks defined earlier in this file),
// scalar fallback serving both the "unaligned" and "tail" cases.
// ---------------------------------------------------------------------------

__global__ void leaky_relu_forward_scalar_kernel(const float* __restrict__ in, float* __restrict__ out,
                                                  float slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float v = __ldg(&in[i]);
        out[i] = v > 0.0f ? v : v * slope;
    }
}

__global__ void leaky_relu_forward_vec4_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                                                float slope, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 v = __ldg(&in[i]);
        float4 r;
        r.x = v.x > 0.0f ? v.x : v.x * slope;
        r.y = v.y > 0.0f ? v.y : v.y * slope;
        r.z = v.z > 0.0f ? v.z : v.z * slope;
        r.w = v.w > 0.0f ? v.w : v.w * slope;
        out[i] = r;
    }
}

__global__ void leaky_relu_backward_scalar_kernel(const float* __restrict__ grad_out, const float* __restrict__ in,
                                                   float* __restrict__ grad_in, float slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float x = __ldg(&in[i]);
        float g = __ldg(&grad_out[i]);
        grad_in[i] = x > 0.0f ? g : g * slope;
    }
}

__global__ void leaky_relu_backward_vec4_kernel(const float4* __restrict__ grad_out, const float4* __restrict__ in,
                                                 float4* __restrict__ grad_in, float slope, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 x = __ldg(&in[i]);
        float4 g = __ldg(&grad_out[i]);
        float4 r;
        r.x = x.x > 0.0f ? g.x : g.x * slope;
        r.y = x.y > 0.0f ? g.y : g.y * slope;
        r.z = x.z > 0.0f ? g.z : g.z * slope;
        r.w = x.w > 0.0f ? g.w : g.w * slope;
        grad_in[i] = r;
    }
}

std::shared_ptr<Tensor> run_cuda_leaky_relu(std::shared_ptr<Tensor> a, float slope) {
    if (!a->is_contiguous()) throw std::invalid_argument("leaky_relu requires a contiguous tensor. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"));
    int n = a->size;
    if (n == 0) return result;

    bool can_vec = (n >= 4) && is_aligned16(a->data_ptr) && is_aligned16(result->data_ptr);

    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        leaky_relu_forward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(a->data_ptr),
            reinterpret_cast<float4*>(result->data_ptr), slope, n4);
        if (tail > 0) {
            leaky_relu_forward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                a->data_ptr + n4 * 4, result->data_ptr + n4 * 4, slope, tail);
        }
    } else {
        leaky_relu_forward_scalar_kernel<<<elem_blocks(n), 256>>>(a->data_ptr, result->data_ptr, slope, n);
    }
    cudaDeviceSynchronize();
    return result;
}

std::shared_ptr<Tensor> run_cuda_leaky_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, float slope) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cuda"));
    int n = input->size;
    if (n == 0) return result;

    bool can_vec = (n >= 4) && is_aligned16(grad_out->data_ptr)
                            && is_aligned16(input->data_ptr)
                            && is_aligned16(result->data_ptr);

    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        leaky_relu_backward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(grad_out->data_ptr),
            reinterpret_cast<const float4*>(input->data_ptr),
            reinterpret_cast<float4*>(result->data_ptr), slope, n4);
        if (tail > 0) {
            leaky_relu_backward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                grad_out->data_ptr + n4 * 4, input->data_ptr + n4 * 4,
                result->data_ptr + n4 * 4, slope, tail);
        }
    } else {
        leaky_relu_backward_scalar_kernel<<<elem_blocks(n), 256>>>(
            grad_out->data_ptr, input->data_ptr, result->data_ptr, slope, n);
    }
    cudaDeviceSynchronize();
    return result;
}

// ---------------------------------------------------------------------------
// EXP
// ---------------------------------------------------------------------------

__global__ void exp_forward_scalar_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = expf(__ldg(&in[i]));
    }
}

__global__ void exp_forward_vec4_kernel(const float4* __restrict__ in, float4* __restrict__ out, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 v = __ldg(&in[i]);
        float4 r;
        r.x = expf(v.x);
        r.y = expf(v.y);
        r.z = expf(v.z);
        r.w = expf(v.w);
        out[i] = r;
    }
}

__global__ void exp_backward_scalar_kernel(const float* __restrict__ grad_out, const float* __restrict__ exp_out,
                                            float* __restrict__ grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        grad_in[i] = __ldg(&grad_out[i]) * __ldg(&exp_out[i]);
    }
}

__global__ void exp_backward_vec4_kernel(const float4* __restrict__ grad_out, const float4* __restrict__ exp_out,
                                          float4* __restrict__ grad_in, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 g = __ldg(&grad_out[i]);
        float4 e = __ldg(&exp_out[i]);
        float4 r;
        r.x = g.x * e.x;
        r.y = g.y * e.y;
        r.z = g.z * e.z;
        r.w = g.w * e.w;
        grad_in[i] = r;
    }
}

std::shared_ptr<Tensor> run_cuda_exp(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) throw std::invalid_argument("exp requires a contiguous tensor. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"));
    int n = a->size;
    if (n == 0) return result;

    bool can_vec = (n >= 4) && is_aligned16(a->data_ptr) && is_aligned16(result->data_ptr);

    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        exp_forward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(a->data_ptr),
            reinterpret_cast<float4*>(result->data_ptr), n4);
        if (tail > 0) {
            exp_forward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                a->data_ptr + n4 * 4, result->data_ptr + n4 * 4, tail);
        }
    } else {
        exp_forward_scalar_kernel<<<elem_blocks(n), 256>>>(a->data_ptr, result->data_ptr, n);
    }
    cudaDeviceSynchronize();
    return result;
}

std::shared_ptr<Tensor> run_cuda_exp_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> exp_output) {
    auto result = std::make_shared<Tensor>(exp_output->shape, std::string("cuda"));
    int n = exp_output->size;
    if (n == 0) return result;

    bool can_vec = (n >= 4) && is_aligned16(grad_out->data_ptr)
                            && is_aligned16(exp_output->data_ptr)
                            && is_aligned16(result->data_ptr);

    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        exp_backward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(grad_out->data_ptr),
            reinterpret_cast<const float4*>(exp_output->data_ptr),
            reinterpret_cast<float4*>(result->data_ptr), n4);
        if (tail > 0) {
            exp_backward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                grad_out->data_ptr + n4 * 4, exp_output->data_ptr + n4 * 4,
                result->data_ptr + n4 * 4, tail);
        }
    } else {
        exp_backward_scalar_kernel<<<elem_blocks(n), 256>>>(
            grad_out->data_ptr, exp_output->data_ptr, result->data_ptr, n);
    }
    cudaDeviceSynchronize();
    return result;
}

// ---------------------------------------------------------------------------
// LOG
// ---------------------------------------------------------------------------

__global__ void log_forward_scalar_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = logf(__ldg(&in[i]));
    }
}

__global__ void log_forward_vec4_kernel(const float4* __restrict__ in, float4* __restrict__ out, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 v = __ldg(&in[i]);
        float4 r;
        r.x = logf(v.x);
        r.y = logf(v.y);
        r.z = logf(v.z);
        r.w = logf(v.w);
        out[i] = r;
    }
}

__global__ void log_backward_scalar_kernel(const float* __restrict__ grad_out, const float* __restrict__ in,
                                            float* __restrict__ grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        grad_in[i] = __ldg(&grad_out[i]) / __ldg(&in[i]);
    }
}

__global__ void log_backward_vec4_kernel(const float4* __restrict__ grad_out, const float4* __restrict__ in,
                                          float4* __restrict__ grad_in, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 g = __ldg(&grad_out[i]);
        float4 x = __ldg(&in[i]);
        float4 r;
        r.x = g.x / x.x;
        r.y = g.y / x.y;
        r.z = g.z / x.z;
        r.w = g.w / x.w;
        grad_in[i] = r;
    }
}

std::shared_ptr<Tensor> run_cuda_log(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) throw std::invalid_argument("log requires a contiguous tensor. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"));
    int n = a->size;
    if (n == 0) return result;

    bool can_vec = (n >= 4) && is_aligned16(a->data_ptr) && is_aligned16(result->data_ptr);

    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        log_forward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(a->data_ptr),
            reinterpret_cast<float4*>(result->data_ptr), n4);
        if (tail > 0) {
            log_forward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                a->data_ptr + n4 * 4, result->data_ptr + n4 * 4, tail);
        }
    } else {
        log_forward_scalar_kernel<<<elem_blocks(n), 256>>>(a->data_ptr, result->data_ptr, n);
    }
    cudaDeviceSynchronize();
    return result;
}

std::shared_ptr<Tensor> run_cuda_log_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cuda"));
    int n = input->size;
    if (n == 0) return result;

    bool can_vec = (n >= 4) && is_aligned16(grad_out->data_ptr)
                            && is_aligned16(input->data_ptr)
                            && is_aligned16(result->data_ptr);

    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        log_backward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(grad_out->data_ptr),
            reinterpret_cast<const float4*>(input->data_ptr),
            reinterpret_cast<float4*>(result->data_ptr), n4);
        if (tail > 0) {
            log_backward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                grad_out->data_ptr + n4 * 4, input->data_ptr + n4 * 4,
                result->data_ptr + n4 * 4, tail);
        }
    } else {
        log_backward_scalar_kernel<<<elem_blocks(n), 256>>>(
            grad_out->data_ptr, input->data_ptr, result->data_ptr, n);
    }
    cudaDeviceSynchronize();
    return result;
}