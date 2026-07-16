#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include "tensor.h"

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include "tensor.h"

// Assuming MAX_DIMS is already defined (e.g., #define MAX_DIMS 8)
#ifndef MAX_DIMS
#define MAX_DIMS 8
#endif

enum class ElementwiseOp { ADD, SUB, MUL, DIV };

// Struct to pass broadcast dimensions and strides by value
struct BroadcastShapeInfo {
    int out_shape[MAX_DIMS];
    int a_shape[MAX_DIMS];
    int a_strides[MAX_DIMS];
    int b_shape[MAX_DIMS];
    int b_strides[MAX_DIMS];
    int ndim;
};

// ---- Tensor op Tensor ----

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
__global__ void broadcast_elementwise_kernel(const float* __restrict__ a, 
                                             const float* __restrict__ b,
                                             float* __restrict__ c, 
                                             BroadcastShapeInfo info, 
                                             int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int a_off = broadcast_index(idx, info.out_shape, info.a_strides, info.a_shape, info.ndim);
    int b_off = broadcast_index(idx, info.out_shape, info.b_strides, info.b_shape, info.ndim);

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

    if (ndim > MAX_DIMS) {
        throw std::runtime_error("broadcast_elementwise: tensor rank exceeds MAX_DIMS (8)");
    }

    // Populate the struct to pass by value
    BroadcastShapeInfo info;
    info.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        info.out_shape[i] = out_shape[i];
        info.a_shape[i] = pa_shape[i];
        info.a_strides[i] = pa_strides[i];
        info.b_shape[i] = pb_shape[i];
        info.b_strides[i] = pb_strides[i];
    }

    int threads = 256;
    int blocks = (out_size + threads - 1) / threads;
    
    // Launch kernel with by-value struct
    broadcast_elementwise_kernel<OP><<<blocks, threads>>>(
        a->fptr(),
        b->fptr(),
        result->fptr(), 
        info, 
        out_size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    
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
    vector_elementwise_vectorized<OP><<<compute_blocks(n), 256>>>(a->fptr(), b->fptr(), result->fptr(), n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

template <ElementwiseOp OP>
static std::shared_ptr<Tensor> run_cuda_scalar(std::shared_ptr<Tensor> a, float scalar) {
    check_scalar_input(a);
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"));
    int n = a->size;
    scalar_elementwise_vectorized<OP><<<compute_blocks(n), 256>>>(a->fptr(), scalar, result->fptr(), n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
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
    bool can_vec = (n >= 4) && is_aligned16(a->fptr()) && is_aligned16(result->fptr());
    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        relu_forward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(a->fptr()),
            reinterpret_cast<float4*>(result->fptr()), n4);
        if (tail > 0) {
            relu_forward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                a->fptr() + n4 * 4, result->fptr() + n4 * 4, tail);
        }
    } else {
        relu_forward_scalar_kernel<<<elem_blocks(n), 256>>>(a->fptr(), result->fptr(), n);
    }
    return result;
}

std::shared_ptr<Tensor> run_cuda_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cuda"));
    int n = input->size;
    if (n == 0) return result;
    bool can_vec = (n >= 4) && is_aligned16(grad_out->fptr())
                            && is_aligned16(input->fptr())
                            && is_aligned16(result->fptr());
    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        relu_backward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(grad_out->fptr()),
            reinterpret_cast<const float4*>(input->fptr()),
            reinterpret_cast<float4*>(result->fptr()), n4);
        if (tail > 0) {
            relu_backward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                grad_out->fptr() + n4 * 4, input->fptr() + n4 * 4,
                result->fptr() + n4 * 4, tail);
        }
    } else {
        relu_backward_scalar_kernel<<<elem_blocks(n), 256>>>(
            grad_out->fptr(), input->fptr(), result->fptr(), n);
    }
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
    bool can_vec = (n >= 4) && is_aligned16(a->fptr()) && is_aligned16(result->fptr());
    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        sigmoid_forward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(a->fptr()),
            reinterpret_cast<float4*>(result->fptr()), n4);
        if (tail > 0) {
            sigmoid_forward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                a->fptr() + n4 * 4, result->fptr() + n4 * 4, tail);
        }
    } else {
        sigmoid_forward_scalar_kernel<<<elem_blocks(n), 256>>>(a->fptr(), result->fptr(), n);
    }
    return result;
}

std::shared_ptr<Tensor> run_cuda_sigmoid_backward(std::shared_ptr<Tensor> grad_out, const float* sig_out_ptr, int size, std::vector<int> shape) {
    auto result = std::make_shared<Tensor>(shape, std::string("cuda"));
    if (size == 0) return result;
    bool can_vec = (size >= 4) && is_aligned16(grad_out->fptr()) && is_aligned16(sig_out_ptr) && is_aligned16(result->fptr());
    if (can_vec) {
        int n4 = size / 4;
        int tail = size - n4 * 4;
        sigmoid_backward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(grad_out->fptr()),
            reinterpret_cast<const float4*>(sig_out_ptr),
            reinterpret_cast<float4*>(result->fptr()), n4);
        if (tail > 0) {
            sigmoid_backward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                grad_out->fptr() + n4*4, sig_out_ptr + n4*4, result->fptr() + n4*4, tail);
        }
    } else {
        sigmoid_backward_scalar_kernel<<<elem_blocks(size), 256>>>(grad_out->fptr(), sig_out_ptr, result->fptr(), size);
    }
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
    bool can_vec = (n >= 4) && is_aligned16(a->fptr()) && is_aligned16(result->fptr());
    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        tanh_forward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(a->fptr()),
            reinterpret_cast<float4*>(result->fptr()), n4);
        if (tail > 0) {
            tanh_forward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                a->fptr() + n4 * 4, result->fptr() + n4 * 4, tail);
        }
    } else {
        tanh_forward_scalar_kernel<<<elem_blocks(n), 256>>>(a->fptr(), result->fptr(), n);
    }
    return result;
}

std::shared_ptr<Tensor> run_cuda_tanh_backward(std::shared_ptr<Tensor> grad_out, const float* tanh_out_ptr, int size, std::vector<int> shape) {
    auto result = std::make_shared<Tensor>(shape, std::string("cuda"));
    if (size == 0) return result;
    
    bool can_vec = (size >= 4) && is_aligned16(grad_out->fptr())
                            && is_aligned16(tanh_out_ptr)
                            && is_aligned16(result->fptr());
    if (can_vec) {
        int n4 = size / 4;
        int tail = size - n4 * 4;
        tanh_backward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(grad_out->fptr()),
            reinterpret_cast<const float4*>(tanh_out_ptr),
            reinterpret_cast<float4*>(result->fptr()), n4);
        if (tail > 0) {
            tanh_backward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                grad_out->fptr() + n4 * 4, tanh_out_ptr + n4 * 4,
                result->fptr() + n4 * 4, tail);
        }
    } else {
        tanh_backward_scalar_kernel<<<elem_blocks(size), 256>>>(
            grad_out->fptr(), tanh_out_ptr, result->fptr(), size);
    }
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

    bool can_vec = (n >= 4) && is_aligned16(a->fptr()) && is_aligned16(result->fptr());

    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        leaky_relu_forward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(a->fptr()),
            reinterpret_cast<float4*>(result->fptr()), slope, n4);
        if (tail > 0) {
            leaky_relu_forward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                a->fptr() + n4 * 4, result->fptr() + n4 * 4, slope, tail);
        }
    } else {
        leaky_relu_forward_scalar_kernel<<<elem_blocks(n), 256>>>(a->fptr(), result->fptr(), slope, n);
    }
    return result;
}

std::shared_ptr<Tensor> run_cuda_leaky_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, float slope) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cuda"));
    int n = input->size;
    if (n == 0) return result;

    bool can_vec = (n >= 4) && is_aligned16(grad_out->fptr())
                            && is_aligned16(input->fptr())
                            && is_aligned16(result->fptr());

    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        leaky_relu_backward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(grad_out->fptr()),
            reinterpret_cast<const float4*>(input->fptr()),
            reinterpret_cast<float4*>(result->fptr()), slope, n4);
        if (tail > 0) {
            leaky_relu_backward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                grad_out->fptr() + n4 * 4, input->fptr() + n4 * 4,
                result->fptr() + n4 * 4, slope, tail);
        }
    } else {
        leaky_relu_backward_scalar_kernel<<<elem_blocks(n), 256>>>(
            grad_out->fptr(), input->fptr(), result->fptr(), slope, n);
    }
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

    bool can_vec = (n >= 4) && is_aligned16(a->fptr()) && is_aligned16(result->fptr());

    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        exp_forward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(a->fptr()),
            reinterpret_cast<float4*>(result->fptr()), n4);
        if (tail > 0) {
            exp_forward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                a->fptr() + n4 * 4, result->fptr() + n4 * 4, tail);
        }
    } else {
        exp_forward_scalar_kernel<<<elem_blocks(n), 256>>>(a->fptr(), result->fptr(), n);
    }
    return result;
}

std::shared_ptr<Tensor> run_cuda_exp_backward(std::shared_ptr<Tensor> grad_out, const float* exp_out_ptr, int size, std::vector<int> shape) {
    auto result = std::make_shared<Tensor>(shape, std::string("cuda"));
    if (size == 0) return result;

    bool can_vec = (size >= 4) && is_aligned16(grad_out->fptr())
                            && is_aligned16(exp_out_ptr)
                            && is_aligned16(result->fptr());

    if (can_vec) {
        int n4 = size / 4;
        int tail = size - n4 * 4;
        exp_backward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(grad_out->fptr()),
            reinterpret_cast<const float4*>(exp_out_ptr),
            reinterpret_cast<float4*>(result->fptr()), n4);
        if (tail > 0) {
            exp_backward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                grad_out->fptr() + n4 * 4, exp_out_ptr + n4 * 4,
                result->fptr() + n4 * 4, tail);
        }
    } else {
        exp_backward_scalar_kernel<<<elem_blocks(size), 256>>>(
            grad_out->fptr(), exp_out_ptr, result->fptr(), size);
    }
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

    bool can_vec = (n >= 4) && is_aligned16(a->fptr()) && is_aligned16(result->fptr());

    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        log_forward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(a->fptr()),
            reinterpret_cast<float4*>(result->fptr()), n4);
        if (tail > 0) {
            log_forward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                a->fptr() + n4 * 4, result->fptr() + n4 * 4, tail);
        }
    } else {
        log_forward_scalar_kernel<<<elem_blocks(n), 256>>>(a->fptr(), result->fptr(), n);
    }
    return result;
}

std::shared_ptr<Tensor> run_cuda_log_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cuda"));
    int n = input->size;
    if (n == 0) return result;

    bool can_vec = (n >= 4) && is_aligned16(grad_out->fptr())
                            && is_aligned16(input->fptr())
                            && is_aligned16(result->fptr());

    if (can_vec) {
        int n4 = n / 4;
        int tail = n - n4 * 4;
        log_backward_vec4_kernel<<<elem_blocks(n4), 256>>>(
            reinterpret_cast<const float4*>(grad_out->fptr()),
            reinterpret_cast<const float4*>(input->fptr()),
            reinterpret_cast<float4*>(result->fptr()), n4);
        if (tail > 0) {
            log_backward_scalar_kernel<<<elem_blocks(tail), 256>>>(
                grad_out->fptr() + n4 * 4, input->fptr() + n4 * 4,
                result->fptr() + n4 * 4, tail);
        }
    } else {
        log_backward_scalar_kernel<<<elem_blocks(n), 256>>>(
            grad_out->fptr(), input->fptr(), result->fptr(), n);
    }
    return result;
}

template <typename T>
struct alignas(16) Vec128 {
    T vals[16 / sizeof(T)];
};
struct AddOp {
    template <typename T>
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

struct SubOp {
    template <typename T>
    __device__ __forceinline__ T operator()(T a, T b) const { return a - b; }
};

struct MulOp {
    template <typename T>
    __device__ __forceinline__ T operator()(T a, T b) const { return a * b; }
};

struct DivOp {
    template <typename T>
    __device__ __forceinline__ T operator()(T a, T b) const { return a / b; }
};
// ---------------------------------------------------------
// 1. Optimized Device Kernels
// ---------------------------------------------------------

template <typename T>
__global__ void relu_kernel_typed(const T* __restrict__ in, T* __restrict__ out, int n) {
    constexpr int VEC_SIZE = 16 / sizeof(T);
    int n_vec = n / VEC_SIZE;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const auto* in_vec = reinterpret_cast<const Vec128<T>*>(in);
    auto* out_vec = reinterpret_cast<Vec128<T>*>(out);

    for (int i = idx; i < n_vec; i += stride) {
        Vec128<T> v = in_vec[i];
        Vec128<T> res;
        
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            res.vals[j] = v.vals[j] > T(0) ? v.vals[j] : T(0);
        }
        out_vec[i] = res;
    }

    int tail_start = n_vec * VEC_SIZE;
    for (int i = tail_start + idx; i < n; i += stride) {
        T v = in[i];
        out[i] = v > T(0) ? v : T(0);
    }
}

template <typename T>
__global__ void relu_backward_kernel_typed(const T* __restrict__ grad_out, const T* __restrict__ input,
                                           T* __restrict__ out, int n) {
    constexpr int VEC_SIZE = 16 / sizeof(T);
    int n_vec = n / VEC_SIZE;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const auto* go_vec = reinterpret_cast<const Vec128<T>*>(grad_out);
    const auto* in_vec = reinterpret_cast<const Vec128<T>*>(input);
    auto* out_vec      = reinterpret_cast<Vec128<T>*>(out);

    for (int i = idx; i < n_vec; i += stride) {
        Vec128<T> g = go_vec[i];
        Vec128<T> v = in_vec[i];
        Vec128<T> res;
        
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            res.vals[j] = v.vals[j] > T(0) ? g.vals[j] : T(0);
        }
        out_vec[i] = res;
    }

    int tail_start = n_vec * VEC_SIZE;
    for (int i = tail_start + idx; i < n; i += stride) {
        out[i] = input[i] > T(0) ? grad_out[i] : T(0);
    }
}

// ---------------------------------------------------------
// 2. Host Launch Functions (Required by bindings.cpp)
// ---------------------------------------------------------

template <typename T>
static std::shared_ptr<Tensor> run_cuda_relu_typed_impl(std::shared_ptr<Tensor> a) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"), a->dtype);
    int n = a->size;
    if (n == 0) return result;
    
    int threads = 256;
    constexpr int VEC_SIZE = 16 / sizeof(T);
    int n_vec = n / VEC_SIZE;
    int blocks = std::min((n_vec + threads - 1) / threads, 4096); 
    if (blocks == 0) blocks = 1; 

    relu_kernel_typed<T><<<blocks, threads>>>(
        static_cast<const T*>(a->data_ptr), static_cast<T*>(result->data_ptr), n);
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
    return result;
}

std::shared_ptr<Tensor> run_cuda_relu_typed(std::shared_ptr<Tensor> a) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cuda_relu_typed_impl<double>(a);
        case DType::INT32:   return run_cuda_relu_typed_impl<int32_t>(a);
        case DType::INT64:   return run_cuda_relu_typed_impl<int64_t>(a);
        default: throw std::runtime_error("relu(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}

template <typename T>
static std::shared_ptr<Tensor> run_cuda_relu_backward_typed_impl(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cuda"), input->dtype);
    int n = input->size;
    if (n == 0) return result;
    
    int threads = 256;
    constexpr int VEC_SIZE = 16 / sizeof(T);
    int n_vec = n / VEC_SIZE;
    int blocks = std::min((n_vec + threads - 1) / threads, 4096);
    if (blocks == 0) blocks = 1;

    relu_backward_kernel_typed<T><<<blocks, threads>>>(
        static_cast<const T*>(grad_out->data_ptr), static_cast<const T*>(input->data_ptr),
        static_cast<T*>(result->data_ptr), n);
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
    return result;
}

std::shared_ptr<Tensor> run_cuda_relu_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    switch (input->dtype) {
        case DType::FLOAT64: return run_cuda_relu_backward_typed_impl<double>(grad_out, input);
        case DType::INT32:   return run_cuda_relu_backward_typed_impl<int32_t>(grad_out, input);
        case DType::INT64:   return run_cuda_relu_backward_typed_impl<int64_t>(grad_out, input);
        default: throw std::runtime_error("relu() backward: unsupported dtype '" + dtype_name(input->dtype) + "'");
    }
}

template <typename T, typename Op>
__global__ void scalar_typed_kernel(const T* __restrict__ a, T s, T* __restrict__ c, int n, Op op) {
    constexpr int VEC_SIZE = 16 / sizeof(T);
    int n_vec = n / VEC_SIZE;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const auto* a_vec = reinterpret_cast<const Vec128<T>*>(a);
    auto* c_vec = reinterpret_cast<Vec128<T>*>(c);

    // Vectorized 128-bit grid stride loop
    for (int i = idx; i < n_vec; i += stride) {
        Vec128<T> v = a_vec[i];
        Vec128<T> res;
        
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            res.vals[j] = op(v.vals[j], s);
        }
        c_vec[i] = res;
    }

    // Scalar tail loop for remainders
    int tail_start = n_vec * VEC_SIZE;
    for (int i = tail_start + idx; i < n; i += stride) {
        c[i] = op(a[i], s);
    }
}

template <typename T, typename Op>
static std::shared_ptr<Tensor> run_cuda_scalar_typed_impl(std::shared_ptr<Tensor> a, double s, Op op) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"), a->dtype);
    T sv = static_cast<T>(s);
    int n = a->size;
    if (n == 0) return result;

    int threads = 256;
    constexpr int VEC_SIZE = 16 / sizeof(T);
    int n_vec = n / VEC_SIZE;
    
    // Cap blocks at 4096 to prevent scheduler overhead; the grid-stride loop handles the rest.
    int blocks = std::min((n_vec + threads - 1) / threads, 4096);
    if (blocks == 0) blocks = 1;

    scalar_typed_kernel<T, Op><<<blocks, threads>>>(
        static_cast<const T*>(a->data_ptr), sv, static_cast<T*>(result->data_ptr), n, op);
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
    
    return result;
}

// ---- Scalar-op functors (reuse the AddOp/SubOp/MulOp/DivOp convention
// already used elsewhere in this file for the tensor-tensor typed ops) ----
struct ScalarAddOp { template<typename T> __device__ T operator()(T a, T b) const { return a + b; } };
struct ScalarSubOp { template<typename T> __device__ T operator()(T a, T b) const { return a - b; } };
struct ScalarMulOp { template<typename T> __device__ T operator()(T a, T b) const { return a * b; } };
struct ScalarDivOp { template<typename T> __device__ T operator()(T a, T b) const { return a / b; } };

template <typename T, typename Op>
__global__ void scalar_op_scalar_kernel(const T* __restrict__ in, T* __restrict__ out, T s, int n, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) out[i] = op(__ldg(&in[i]), s);
}

template <typename T, typename Op>
static std::shared_ptr<Tensor> run_cuda_scalar_op_typed(std::shared_ptr<Tensor> a, double s, Op op) {
    if (!a->is_contiguous())
        throw std::invalid_argument("Scalar ops require a contiguous tensor for non-float32 dtypes. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"), a->dtype);
    int n = a->size;
    if (n == 0) return result;
    T sv = (T)s;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    scalar_op_scalar_kernel<T, Op><<<blocks, threads>>>(
        static_cast<const T*>(a->data_ptr), static_cast<T*>(result->data_ptr), sv, n, op);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

std::shared_ptr<Tensor> run_cuda_add_scalar_typed(std::shared_ptr<Tensor> a, double s) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cuda_scalar_op_typed<double>(a, s, ScalarAddOp{});
        case DType::INT32:   return run_cuda_scalar_op_typed<int32_t>(a, s, ScalarAddOp{});
        case DType::INT64:   return run_cuda_scalar_op_typed<int64_t>(a, s, ScalarAddOp{});
        default: throw std::runtime_error("add_scalar(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}
std::shared_ptr<Tensor> run_cuda_sub_scalar_typed(std::shared_ptr<Tensor> a, double s) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cuda_scalar_op_typed<double>(a, s, ScalarSubOp{});
        case DType::INT32:   return run_cuda_scalar_op_typed<int32_t>(a, s, ScalarSubOp{});
        case DType::INT64:   return run_cuda_scalar_op_typed<int64_t>(a, s, ScalarSubOp{});
        default: throw std::runtime_error("sub_scalar(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}
std::shared_ptr<Tensor> run_cuda_mul_scalar_typed(std::shared_ptr<Tensor> a, double s) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cuda_scalar_op_typed<double>(a, s, ScalarMulOp{});
        case DType::INT32:   return run_cuda_scalar_op_typed<int32_t>(a, s, ScalarMulOp{});
        case DType::INT64:   return run_cuda_scalar_op_typed<int64_t>(a, s, ScalarMulOp{});
        default: throw std::runtime_error("mul_scalar(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}
std::shared_ptr<Tensor> run_cuda_div_scalar_typed(std::shared_ptr<Tensor> a, double s) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cuda_scalar_op_typed<double>(a, s, ScalarDivOp{});
        case DType::INT32:   return run_cuda_scalar_op_typed<int32_t>(a, s, ScalarDivOp{});
        case DType::INT64:   return run_cuda_scalar_op_typed<int64_t>(a, s, ScalarDivOp{});
        default: throw std::runtime_error("div_scalar(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}

// ---- Typed leaky_relu (CUDA), vectorized per-type ----

template <typename T>
__global__ void leaky_relu_scalar_kernel_typed(const T* __restrict__ in, T* __restrict__ out, double slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        T v = __ldg(&in[i]);
        out[i] = v > T(0) ? v : (T)((double)v * slope);
    }
}

__global__ void leaky_relu_vec4_kernel_f32(const float4* __restrict__ in, float4* __restrict__ out, double slope, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 v = __ldg(&in[i]);
        float4 r;
        r.x = v.x > 0.0f ? v.x : (float)((double)v.x * slope);
        r.y = v.y > 0.0f ? v.y : (float)((double)v.y * slope);
        r.z = v.z > 0.0f ? v.z : (float)((double)v.z * slope);
        r.w = v.w > 0.0f ? v.w : (float)((double)v.w * slope);
        out[i] = r;
    }
}

__global__ void leaky_relu_vec2_kernel_f64(const double2* __restrict__ in, double2* __restrict__ out, double slope, int n2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n2; i += stride) {
        double2 v = __ldg(&in[i]);
        double2 r;
        r.x = v.x > 0.0 ? v.x : v.x * slope;
        r.y = v.y > 0.0 ? v.y : v.y * slope;
        out[i] = r;
    }
}

__global__ void leaky_relu_vec4_kernel_i32(const int4* __restrict__ in, int4* __restrict__ out, double slope, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        int4 v = __ldg(&in[i]);
        int4 r;
        r.x = v.x > 0 ? v.x : (int32_t)((double)v.x * slope);
        r.y = v.y > 0 ? v.y : (int32_t)((double)v.y * slope);
        r.z = v.z > 0 ? v.z : (int32_t)((double)v.z * slope);
        r.w = v.w > 0 ? v.w : (int32_t)((double)v.w * slope);
        out[i] = r;
    }
}

__global__ void leaky_relu_vec2_kernel_i64(const longlong2* __restrict__ in, longlong2* __restrict__ out, double slope, int n2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n2; i += stride) {
        longlong2 v = __ldg(&in[i]);
        longlong2 r;
        r.x = v.x > 0 ? v.x : (int64_t)((double)v.x * slope);
        r.y = v.y > 0 ? v.y : (int64_t)((double)v.y * slope);
        out[i] = r;
    }
}

template <typename T>
static std::shared_ptr<Tensor> run_cuda_leaky_relu_typed_impl(std::shared_ptr<Tensor> a, double slope) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"), a->dtype);
    int n = a->size;
    if (n == 0) return result;

    const T* in_ptr = static_cast<const T*>(a->data_ptr);
    T* out_ptr = static_cast<T*>(result->data_ptr);

    constexpr int lanes = (sizeof(T) == 4) ? 4 : 2;
    bool can_vec = (n >= lanes) && is_aligned16(in_ptr) && is_aligned16(out_ptr);

    if (can_vec) {
        int n_vec = n / lanes;
        int tail = n - n_vec * lanes;

        if constexpr (std::is_same<T, float>::value) {
            leaky_relu_vec4_kernel_f32<<<elem_blocks(n_vec), 256>>>(
                reinterpret_cast<const float4*>(in_ptr), reinterpret_cast<float4*>(out_ptr), slope, n_vec);
        } else if constexpr (std::is_same<T, double>::value) {
            leaky_relu_vec2_kernel_f64<<<elem_blocks(n_vec), 256>>>(
                reinterpret_cast<const double2*>(in_ptr), reinterpret_cast<double2*>(out_ptr), slope, n_vec);
        } else if constexpr (std::is_same<T, int32_t>::value) {
            leaky_relu_vec4_kernel_i32<<<elem_blocks(n_vec), 256>>>(
                reinterpret_cast<const int4*>(in_ptr), reinterpret_cast<int4*>(out_ptr), slope, n_vec);
        } else if constexpr (std::is_same<T, int64_t>::value) {
            leaky_relu_vec2_kernel_i64<<<elem_blocks(n_vec), 256>>>(
                reinterpret_cast<const longlong2*>(in_ptr), reinterpret_cast<longlong2*>(out_ptr), slope, n_vec);
        }

        if (tail > 0) {
            leaky_relu_scalar_kernel_typed<T><<<elem_blocks(tail), 256>>>(
                in_ptr + n_vec * lanes, out_ptr + n_vec * lanes, slope, tail);
        }
    } else {
        leaky_relu_scalar_kernel_typed<T><<<elem_blocks(n), 256>>>(in_ptr, out_ptr, slope, n);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

std::shared_ptr<Tensor> run_cuda_leaky_relu_typed(std::shared_ptr<Tensor> a, double slope) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cuda_leaky_relu_typed_impl<double>(a, slope);
        case DType::INT32:   return run_cuda_leaky_relu_typed_impl<int32_t>(a, slope);
        case DType::INT64:   return run_cuda_leaky_relu_typed_impl<int64_t>(a, slope);
        default: throw std::runtime_error("leaky_relu(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}

// ---- Typed leaky_relu_backward (CUDA) ----

template <typename T>
__global__ void leaky_relu_backward_scalar_kernel_typed(const T* __restrict__ grad_out, const T* __restrict__ input,
                                                         T* __restrict__ out, T slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        T x = __ldg(&input[i]);
        T g = __ldg(&grad_out[i]);
        out[i] = x > T(0) ? g : (T)(g * slope);
    }
}

__global__ void leaky_relu_backward_vec4_kernel_f32(const float4* __restrict__ grad_out, const float4* __restrict__ input,
                                                     float4* __restrict__ out, float slope, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 x = __ldg(&input[i]);
        float4 g = __ldg(&grad_out[i]);
        float4 r;
        r.x = x.x > 0.0f ? g.x : g.x * slope;
        r.y = x.y > 0.0f ? g.y : g.y * slope;
        r.z = x.z > 0.0f ? g.z : g.z * slope;
        r.w = x.w > 0.0f ? g.w : g.w * slope;
        out[i] = r;
    }
}

__global__ void leaky_relu_backward_vec2_kernel_f64(const double2* __restrict__ grad_out, const double2* __restrict__ input,
                                                     double2* __restrict__ out, double slope, int n2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n2; i += stride) {
        double2 x = __ldg(&input[i]);
        double2 g = __ldg(&grad_out[i]);
        double2 r;
        r.x = x.x > 0.0 ? g.x : g.x * slope;
        r.y = x.y > 0.0 ? g.y : g.y * slope;
        out[i] = r;
    }
}

__global__ void leaky_relu_backward_vec4_kernel_i32(const int4* __restrict__ grad_out, const int4* __restrict__ input,
                                                     int4* __restrict__ out, int32_t slope, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        int4 x = __ldg(&input[i]);
        int4 g = __ldg(&grad_out[i]);
        int4 r;
        r.x = x.x > 0 ? g.x : g.x * slope;
        r.y = x.y > 0 ? g.y : g.y * slope;
        r.z = x.z > 0 ? g.z : g.z * slope;
        r.w = x.w > 0 ? g.w : g.w * slope;
        out[i] = r;
    }
}

__global__ void leaky_relu_backward_vec2_kernel_i64(const longlong2* __restrict__ grad_out, const longlong2* __restrict__ input,
                                                     longlong2* __restrict__ out, int64_t slope, int n2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n2; i += stride) {
        longlong2 x = __ldg(&input[i]);
        longlong2 g = __ldg(&grad_out[i]);
        longlong2 r;
        r.x = x.x > 0 ? g.x : g.x * slope;
        r.y = x.y > 0 ? g.y : g.y * slope;
        out[i] = r;
    }
}

template <typename T>
static std::shared_ptr<Tensor> run_cuda_leaky_relu_backward_typed_impl(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, double slope) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cuda"), input->dtype);
    int n = input->size;
    if (n == 0) return result;

    const T* go_ptr = static_cast<const T*>(grad_out->data_ptr);
    const T* in_ptr = static_cast<const T*>(input->data_ptr);
    T* out_ptr = static_cast<T*>(result->data_ptr);
    T s = (T)slope;

    constexpr int lanes = (sizeof(T) == 4) ? 4 : 2;
    bool can_vec = (n >= lanes) && is_aligned16(go_ptr) && is_aligned16(in_ptr) && is_aligned16(out_ptr);

    if (can_vec) {
        int n_vec = n / lanes;
        int tail = n - n_vec * lanes;

        if constexpr (std::is_same<T, float>::value) {
            leaky_relu_backward_vec4_kernel_f32<<<elem_blocks(n_vec), 256>>>(
                reinterpret_cast<const float4*>(go_ptr), reinterpret_cast<const float4*>(in_ptr),
                reinterpret_cast<float4*>(out_ptr), s, n_vec);
        } else if constexpr (std::is_same<T, double>::value) {
            leaky_relu_backward_vec2_kernel_f64<<<elem_blocks(n_vec), 256>>>(
                reinterpret_cast<const double2*>(go_ptr), reinterpret_cast<const double2*>(in_ptr),
                reinterpret_cast<double2*>(out_ptr), s, n_vec);
        } else if constexpr (std::is_same<T, int32_t>::value) {
            leaky_relu_backward_vec4_kernel_i32<<<elem_blocks(n_vec), 256>>>(
                reinterpret_cast<const int4*>(go_ptr), reinterpret_cast<const int4*>(in_ptr),
                reinterpret_cast<int4*>(out_ptr), s, n_vec);
        } else if constexpr (std::is_same<T, int64_t>::value) {
            leaky_relu_backward_vec2_kernel_i64<<<elem_blocks(n_vec), 256>>>(
                reinterpret_cast<const longlong2*>(go_ptr), reinterpret_cast<const longlong2*>(in_ptr),
                reinterpret_cast<longlong2*>(out_ptr), s, n_vec);
        }

        if (tail > 0) {
            leaky_relu_backward_scalar_kernel_typed<T><<<elem_blocks(tail), 256>>>(
                go_ptr + n_vec * lanes, in_ptr + n_vec * lanes, out_ptr + n_vec * lanes, s, tail);
        }
    } else {
        leaky_relu_backward_scalar_kernel_typed<T><<<elem_blocks(n), 256>>>(go_ptr, in_ptr, out_ptr, s, n);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

std::shared_ptr<Tensor> run_cuda_leaky_relu_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, double slope) {
    switch (input->dtype) {
        case DType::FLOAT64: return run_cuda_leaky_relu_backward_typed_impl<double>(grad_out, input, slope);
        case DType::INT32:   return run_cuda_leaky_relu_backward_typed_impl<int32_t>(grad_out, input, slope);
        case DType::INT64:   return run_cuda_leaky_relu_backward_typed_impl<int64_t>(grad_out, input, slope);
        default: throw std::runtime_error("leaky_relu() backward: unsupported dtype '" + dtype_name(input->dtype) + "'");
    }
}

__global__ void sqrt_forward_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Vectorized 128-bit loads/stores
    int n4 = n / 4;
    const float4* in_vec = reinterpret_cast<const float4*>(in);
    float4* out_vec = reinterpret_cast<float4*>(out);
    
    for (int i = idx; i < n4; i += stride) {
        float4 val = __ldg(&in_vec[i]);
        float4 res;
        res.x = sqrtf(val.x);
        res.y = sqrtf(val.y);
        res.z = sqrtf(val.z);
        res.w = sqrtf(val.w);
        out_vec[i] = res;
    }
    
    // Tail handling
    for (int i = n4 * 4 + idx; i < n; i += stride) {
        out[i] = sqrtf(__ldg(&in[i]));
    }
}

__global__ void sqrt_backward_kernel(const float* __restrict__ grad_out, const float* __restrict__ sqrt_out,
                                      float* __restrict__ grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int n4 = n / 4;
    const float4* go_vec = reinterpret_cast<const float4*>(grad_out);
    const float4* so_vec = reinterpret_cast<const float4*>(sqrt_out);
    float4* gi_vec = reinterpret_cast<float4*>(grad_in);
    
    for (int i = idx; i < n4; i += stride) {
        float4 go = __ldg(&go_vec[i]);
        float4 so = __ldg(&so_vec[i]);
        float4 gi;
        gi.x = go.x * 0.5f / so.x;
        gi.y = go.y * 0.5f / so.y;
        gi.z = go.z * 0.5f / so.z;
        gi.w = go.w * 0.5f / so.w;
        gi_vec[i] = gi;
    }
    
    for (int i = n4 * 4 + idx; i < n; i += stride) {
        grad_in[i] = __ldg(&grad_out[i]) * 0.5f / __ldg(&sqrt_out[i]);
    }
}

std::shared_ptr<Tensor> run_cuda_sqrt(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) throw std::invalid_argument("sqrt requires a contiguous tensor. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"));
    int n = a->size;
    if (n == 0) return result;
    
    // Dividing by 4 for the primary loop blocks, standard blocks for tail
    sqrt_forward_kernel<<<elem_blocks(n), 256>>>(a->fptr(), result->fptr(), n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

std::shared_ptr<Tensor> run_cuda_sqrt_backward(std::shared_ptr<Tensor> grad_out, const float* sqrt_out_ptr, int size, std::vector<int> shape) {
    auto result = std::make_shared<Tensor>(shape, std::string("cuda"));
    if (size == 0) return result;
    sqrt_backward_kernel<<<elem_blocks(size), 256>>>(grad_out->fptr(), sqrt_out_ptr, result->fptr(), size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

__global__ void sqrt_f64_kernel(const double* __restrict__ in, double* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // double2 is 128-bit aligned
    int n2 = n / 2;
    const double2* in_vec = reinterpret_cast<const double2*>(in);
    double2* out_vec = reinterpret_cast<double2*>(out);
    
    for (int i = idx; i < n2; i += stride) {
        double2 val = __ldg(&in_vec[i]);
        double2 res;
        res.x = sqrt(val.x);
        res.y = sqrt(val.y);
        out_vec[i] = res;
    }
    
    for (int i = n2 * 2 + idx; i < n; i += stride) {
        out[i] = sqrt(__ldg(&in[i]));
    }
}

std::shared_ptr<Tensor> run_cuda_sqrt_f64(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) throw std::invalid_argument("sqrt requires a contiguous tensor. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"), DType::FLOAT64);
    int n = a->size;
    if (n == 0) return result;
    sqrt_f64_kernel<<<elem_blocks(n), 256>>>(
        static_cast<const double*>(a->data_ptr), static_cast<double*>(result->data_ptr), n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

// ---- abs: float32 native + typed float64/int32/int64 ----
__global__ void abs_forward_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int n4 = n / 4;
    const float4* in_vec = reinterpret_cast<const float4*>(in);
    float4* out_vec = reinterpret_cast<float4*>(out);
    
    for (int i = idx; i < n4; i += stride) {
        float4 val = __ldg(&in_vec[i]);
        float4 res;
        res.x = fabsf(val.x);
        res.y = fabsf(val.y);
        res.z = fabsf(val.z);
        res.w = fabsf(val.w);
        out_vec[i] = res;
    }
    
    for (int i = n4 * 4 + idx; i < n; i += stride) {
        out[i] = fabsf(__ldg(&in[i]));
    }
}

__global__ void abs_backward_kernel(const float* __restrict__ grad_out, const float* __restrict__ input,
                                     float* __restrict__ grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int n4 = n / 4;
    const float4* go_vec = reinterpret_cast<const float4*>(grad_out);
    const float4* in_vec = reinterpret_cast<const float4*>(input);
    float4* gi_vec = reinterpret_cast<float4*>(grad_in);
    
    for (int i = idx; i < n4; i += stride) {
        float4 go = __ldg(&go_vec[i]);
        float4 in_val = __ldg(&in_vec[i]);
        float4 gi;
        // Branchless sign extraction evaluates cleanly in PTX
        gi.x = go.x * ((in_val.x > 0.0f) - (in_val.x < 0.0f));
        gi.y = go.y * ((in_val.y > 0.0f) - (in_val.y < 0.0f));
        gi.z = go.z * ((in_val.z > 0.0f) - (in_val.z < 0.0f));
        gi.w = go.w * ((in_val.w > 0.0f) - (in_val.w < 0.0f));
        gi_vec[i] = gi;
    }
    
    for (int i = n4 * 4 + idx; i < n; i += stride) {
        float x = __ldg(&input[i]);
        grad_in[i] = __ldg(&grad_out[i]) * ((x > 0.0f) - (x < 0.0f));
    }
}

std::shared_ptr<Tensor> run_cuda_abs(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) throw std::invalid_argument("abs requires a contiguous tensor. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"));
    int n = a->size;
    if (n == 0) return result;
    abs_forward_kernel<<<elem_blocks(n), 256>>>(a->fptr(), result->fptr(), n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

std::shared_ptr<Tensor> run_cuda_abs_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cuda"));
    int n = input->size;
    if (n == 0) return result;
    abs_backward_kernel<<<elem_blocks(n), 256>>>(grad_out->fptr(), input->fptr(), result->fptr(), n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

template <typename T>
__global__ void abs_scalar_kernel_typed(const T* __restrict__ in, T* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Manual loop unrolling (ILP) to optimize throughput for generic types
    int i = idx;
    int stride4 = stride * 4;
    
    for (; i <= n - stride4; i += stride4) {
        T v0 = __ldg(&in[i]);
        T v1 = __ldg(&in[i + stride]);
        T v2 = __ldg(&in[i + stride * 2]);
        T v3 = __ldg(&in[i + stride * 3]);
        
        out[i] = v0 < T(0) ? -v0 : v0;
        out[i + stride] = v1 < T(0) ? -v1 : v1;
        out[i + stride * 2] = v2 < T(0) ? -v2 : v2;
        out[i + stride * 3] = v3 < T(0) ? -v3 : v3;
    }
    
    // Tail
    for (; i < n; i += stride) {
        T v = __ldg(&in[i]);
        out[i] = v < T(0) ? -v : v;
    }
}

template <typename T>
static std::shared_ptr<Tensor> run_cuda_abs_typed_impl(std::shared_ptr<Tensor> a) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"), a->dtype);
    int n = a->size;
    if (n == 0) return result;
    abs_scalar_kernel_typed<T><<<elem_blocks(n), 256>>>(
        static_cast<const T*>(a->data_ptr), static_cast<T*>(result->data_ptr), n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

std::shared_ptr<Tensor> run_cuda_abs_typed(std::shared_ptr<Tensor> a) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cuda_abs_typed_impl<double>(a);
        case DType::INT32:   return run_cuda_abs_typed_impl<int32_t>(a);
        case DType::INT64:   return run_cuda_abs_typed_impl<int64_t>(a);
        default: throw std::runtime_error("abs(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}

template <typename T>
__global__ void abs_backward_scalar_kernel_typed(const T* __restrict__ grad_out, const T* __restrict__ input,
                                                  T* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Unrolled x4 & Branchless evaluation
    int i = idx;
    int stride4 = stride * 4;
    
    for (; i <= n - stride4; i += stride4) {
        T i0 = __ldg(&input[i]);
        T i1 = __ldg(&input[i + stride]);
        T i2 = __ldg(&input[i + stride * 2]);
        T i3 = __ldg(&input[i + stride * 3]);

        out[i] = __ldg(&grad_out[i]) * ((i0 > T(0)) - (i0 < T(0)));
        out[i + stride] = __ldg(&grad_out[i + stride]) * ((i1 > T(0)) - (i1 < T(0)));
        out[i + stride * 2] = __ldg(&grad_out[i + stride * 2]) * ((i2 > T(0)) - (i2 < T(0)));
        out[i + stride * 3] = __ldg(&grad_out[i + stride * 3]) * ((i3 > T(0)) - (i3 < T(0)));
    }
    
    // Tail
    for (; i < n; i += stride) {
        T x = __ldg(&input[i]);
        out[i] = __ldg(&grad_out[i]) * ((x > T(0)) - (x < T(0)));
    }
}

template <typename T>
static std::shared_ptr<Tensor> run_cuda_abs_backward_typed_impl(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cuda"), input->dtype);
    int n = input->size;
    if (n == 0) return result;
    abs_backward_scalar_kernel_typed<T><<<elem_blocks(n), 256>>>(
        static_cast<const T*>(grad_out->data_ptr), static_cast<const T*>(input->data_ptr),
        static_cast<T*>(result->data_ptr), n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

std::shared_ptr<Tensor> run_cuda_abs_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    switch (input->dtype) {
        case DType::FLOAT64: return run_cuda_abs_backward_typed_impl<double>(grad_out, input);
        case DType::INT32:   return run_cuda_abs_backward_typed_impl<int32_t>(grad_out, input);
        case DType::INT64:   return run_cuda_abs_backward_typed_impl<int64_t>(grad_out, input);
        default: throw std::runtime_error("abs() backward: unsupported dtype '" + dtype_name(input->dtype) + "'");
    }
}