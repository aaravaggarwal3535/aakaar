#include <cuda_runtime.h>
#include <curand.h>
#include <cstdint>
#include <memory>
#include "tensor.h"
#include "rng.h"
#include <cstdint>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Helpers — matches the elem_blocks()/is_aligned16() convention already
// established in the elementwise kernel file, for consistency.
// ---------------------------------------------------------------------------

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

void run_curand_uniform(std::shared_ptr<Tensor> t, unsigned long long seed) {
    if (t->size == 0) return;  // nothing to generate; avoids a pointless curand call

    curandGenerator_t generator = CUDARandomManager::get_instance().get_generator(seed);
    curandStatus_t status;
    switch (t->dtype) {
        case DType::FLOAT32:
            status = curandGenerateUniform(generator, static_cast<float*>(t->data_ptr), t->size);
            break;
        case DType::FLOAT64:
            status = curandGenerateUniformDouble(generator, static_cast<double*>(t->data_ptr), t->size);
            break;
        default:
            throw std::runtime_error("rand() only supports float32/float64 dtypes. Use randint() for integer types.");
    }
    if (status != CURAND_STATUS_SUCCESS) throw std::runtime_error("cuRAND generation failed");
}

// ---------------------------------------------------------------------------
// Uniform-float -> integer-range conversion kernels
//
// Optimizations applied:
//   - __ldg() on the read-only uniform input (global memory caching).
//   - Grid-stride loop + capped block count (elem_blocks), matching the rest
//     of the codebase's convention and fixing the uncapped-blocks
//     inconsistency in the original version.
//   - float4/int4 vectorization for the INT32 path (4 uniforms -> 4 ints per
//     thread), alignment-guarded exactly like the elementwise kernels — a
//     "contiguous" tensor's stride can be fine while its starting byte
//     offset (e.g. a sliced view) isn't guaranteed 16-byte aligned, so a
//     runtime check gates the vec4 path with a scalar fallback.
//   - INT64 path is NOT vectorized: int4 vectorization doesn't apply cleanly
//     to 8-byte output elements without a 256-bit vector type, and CUDA has
//     no native int4-of-int64 equivalent — vectorizing here would need
//     manual struct packing that adds complexity for a type that's already
//     memory-bandwidth-light relative to the temp float buffer read. Not
//     applying it is a deliberate choice, not an oversight.
// ---------------------------------------------------------------------------

__global__ void float_to_randint32_scalar_kernel(const float* __restrict__ uniform, int32_t* __restrict__ out,
                                                  int n, int low, int range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float u = __ldg(&uniform[i]);
        int v = low + (int)(u * range);
        if (v >= low + range) v = low + range - 1;  // guard the rare uniform==1.0 edge case
        out[i] = v;
    }
}

__global__ void float_to_randint32_vec4_kernel(const float4* __restrict__ uniform, int4* __restrict__ out,
                                                int n4, int low, int range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 u = __ldg(&uniform[i]);
        int4 r;
        int hi = low + range - 1;
        r.x = low + (int)(u.x * range); if (r.x > hi) r.x = hi;
        r.y = low + (int)(u.y * range); if (r.y > hi) r.y = hi;
        r.z = low + (int)(u.z * range); if (r.z > hi) r.z = hi;
        r.w = low + (int)(u.w * range); if (r.w > hi) r.w = hi;
        out[i] = r;
    }
}

__global__ void float_to_randint64_kernel(const float* __restrict__ uniform, int64_t* __restrict__ out,
                                           int n, long long low, long long range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float u = __ldg(&uniform[i]);
        long long v = low + (long long)((double)u * (double)range);
        if (v >= low + range) v = low + range - 1;
        out[i] = v;
    }
}

void run_curand_randint(std::shared_ptr<Tensor> t, long long low, long long high, unsigned long long seed) {
    if (t->dtype != DType::INT32 && t->dtype != DType::INT64)
        throw std::runtime_error("randint() only supports int32/int64 dtypes.");
    if (high <= low)
        throw std::invalid_argument("randint(): high must be greater than low");

    int n = t->size;
    if (n == 0) return;  // nothing to generate; also avoids a zero-byte cudaMalloc

    curandGenerator_t generator = CUDARandomManager::get_instance().get_generator(seed);

    float* temp;
    cudaError_t merr = cudaMalloc(&temp, (size_t)n * sizeof(float));
    if (merr != cudaSuccess) throw std::runtime_error(std::string("Aakaar CUDA allocation failed: ") + cudaGetErrorString(merr));

    curandStatus_t status = curandGenerateUniform(generator, temp, n);
    if (status != CURAND_STATUS_SUCCESS) {
        cudaFree(temp);  // release before throwing — no leak on the error path
        throw std::runtime_error("cuRAND generation failed");
    }

    if (t->dtype == DType::INT32) {
        int range = (int)(high - low);
        int32_t* out_ptr = static_cast<int32_t*>(t->data_ptr);

        bool can_vec = (n >= 4) && is_aligned16(temp) && is_aligned16(out_ptr);
        if (can_vec) {
            int n4 = n / 4;
            int tail = n - n4 * 4;
            float_to_randint32_vec4_kernel<<<elem_blocks(n4), 256>>>(
                reinterpret_cast<const float4*>(temp),
                reinterpret_cast<int4*>(out_ptr), n4, (int)low, range);
            if (tail > 0) {
                float_to_randint32_scalar_kernel<<<elem_blocks(tail), 256>>>(
                    temp + n4 * 4, out_ptr + n4 * 4, tail, (int)low, range);
            }
        } else {
            float_to_randint32_scalar_kernel<<<elem_blocks(n), 256>>>(temp, out_ptr, n, (int)low, range);
        }
    } else {
        long long range = high - low;
        float_to_randint64_kernel<<<elem_blocks(n), 256>>>(
            temp, static_cast<int64_t*>(t->data_ptr), n, low, range);
    }

    cudaError_t launch_err = cudaGetLastError();
    cudaFree(temp);
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(std::string("randint kernel launch failed: ") + cudaGetErrorString(launch_err));
    }
}


// ---------------------------------------------------------------------------
// Op functors (unchanged)
// ---------------------------------------------------------------------------

struct AddOp { template<typename T> __device__ T operator()(T a, T b) const { return a + b; } };
struct SubOp { template<typename T> __device__ T operator()(T a, T b) const { return a - b; } };
struct MulOp { template<typename T> __device__ T operator()(T a, T b) const { return a * b; } };
struct DivOp { template<typename T> __device__ T operator()(T a, T b) const { return a / b; } };
// NOTE: for INT32/INT64, this is integer division — truncates toward zero,
// and division by zero is undefined behavior (no inf/nan fallback the way
// floating point div gets). This matches plain C++ `a / b` semantics on
// integers; it is not a bug, just a real behavioral difference from the
// float32 div kernel worth knowing about at the call site.

// ---------------------------------------------------------------------------
// Scalar kernel (used for the tail, and as the fallback when vectorization
// isn't safe for a given pointer alignment)
// ---------------------------------------------------------------------------

template <typename T, typename Op>
__global__ void elementwise_scalar_kernel(const T* __restrict__ a, const T* __restrict__ b,
                                           T* __restrict__ c, int n, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        c[i] = op(__ldg(&a[i]), __ldg(&b[i]));
    }
}

// ---------------------------------------------------------------------------
// Vectorized kernels — one specialization per type, using each type's
// natural 16-byte-wide CUDA vector type. Written as explicit overloads
// rather than a single generic template, since CUDA has no uniform
// "4-wide vector of T" alias across float/double/int32/int64 — each has
// its own native vector type with a different lane count.
// ---------------------------------------------------------------------------

template <typename Op>
__global__ void elementwise_vec4_kernel_f32(const float4* __restrict__ a, const float4* __restrict__ b,
                                             float4* __restrict__ c, int n4, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 va = __ldg(&a[i]);
        float4 vb = __ldg(&b[i]);
        float4 r;
        r.x = op(va.x, vb.x);
        r.y = op(va.y, vb.y);
        r.z = op(va.z, vb.z);
        r.w = op(va.w, vb.w);
        c[i] = r;
    }
}

template <typename Op>
__global__ void elementwise_vec2_kernel_f64(const double2* __restrict__ a, const double2* __restrict__ b,
                                             double2* __restrict__ c, int n2, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n2; i += stride) {
        double2 va = __ldg(&a[i]);
        double2 vb = __ldg(&b[i]);
        double2 r;
        r.x = op(va.x, vb.x);
        r.y = op(va.y, vb.y);
        c[i] = r;
    }
}

template <typename Op>
__global__ void elementwise_vec4_kernel_i32(const int4* __restrict__ a, const int4* __restrict__ b,
                                             int4* __restrict__ c, int n4, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n4; i += stride) {
        int4 va = __ldg(&a[i]);
        int4 vb = __ldg(&b[i]);
        int4 r;
        r.x = op(va.x, vb.x);
        r.y = op(va.y, vb.y);
        r.z = op(va.z, vb.z);
        r.w = op(va.w, vb.w);
        c[i] = r;
    }
}

template <typename Op>
__global__ void elementwise_vec2_kernel_i64(const longlong2* __restrict__ a, const longlong2* __restrict__ b,
                                             longlong2* __restrict__ c, int n2, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n2; i += stride) {
        longlong2 va = __ldg(&a[i]);
        longlong2 vb = __ldg(&b[i]);
        longlong2 r;
        r.x = op(va.x, vb.x);
        r.y = op(va.y, vb.y);
        c[i] = r;
    }
}

// ---------------------------------------------------------------------------
// Dispatch: picks the right vector width per type, guards alignment,
// handles the tail with the scalar kernel. One specialization per type
// rather than a fully generic template, since the vector width (4 lanes
// for 4-byte types, 2 lanes for 8-byte types) genuinely differs by type
// and forcing a single code path would either waste half a double2's
// width on int32 or require unsafe reinterpretation for int64.
// ---------------------------------------------------------------------------

template <typename T, typename Op>
static std::shared_ptr<Tensor> run_cuda_elementwise_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, Op op) {
    if (a->shape != b->shape)
        throw std::invalid_argument("Broadcasting is not yet supported for non-float32 dtypes.");

    auto result = std::make_shared<Tensor>(a->shape, std::string("cuda"), a->dtype);
    int n = a->size;
    if (n == 0) return result;

    const T* a_ptr = static_cast<const T*>(a->data_ptr);
    const T* b_ptr = static_cast<const T*>(b->data_ptr);
    T* c_ptr = static_cast<T*>(result->data_ptr);

    constexpr int lanes = (sizeof(T) == 4) ? 4 : 2;  // 4-byte types -> 4-wide; 8-byte types -> 2-wide (both = 16 bytes)
    bool can_vec = (n >= lanes) && is_aligned16(a_ptr) && is_aligned16(b_ptr) && is_aligned16(c_ptr);

    if (can_vec) {
        int n_vec = n / lanes;
        int tail = n - n_vec * lanes;

        if constexpr (sizeof(T) == 4 && std::is_same<T, float>::value) {
            elementwise_vec4_kernel_f32<<<elem_blocks(n_vec), 256>>>(
                reinterpret_cast<const float4*>(a_ptr), reinterpret_cast<const float4*>(b_ptr),
                reinterpret_cast<float4*>(c_ptr), n_vec, op);
        } else if constexpr (sizeof(T) == 8 && std::is_same<T, double>::value) {
            elementwise_vec2_kernel_f64<<<elem_blocks(n_vec), 256>>>(
                reinterpret_cast<const double2*>(a_ptr), reinterpret_cast<const double2*>(b_ptr),
                reinterpret_cast<double2*>(c_ptr), n_vec, op);
        } else if constexpr (sizeof(T) == 4 && std::is_same<T, int32_t>::value) {
            elementwise_vec4_kernel_i32<<<elem_blocks(n_vec), 256>>>(
                reinterpret_cast<const int4*>(a_ptr), reinterpret_cast<const int4*>(b_ptr),
                reinterpret_cast<int4*>(c_ptr), n_vec, op);
        } else if constexpr (sizeof(T) == 8 && std::is_same<T, int64_t>::value) {
            elementwise_vec2_kernel_i64<<<elem_blocks(n_vec), 256>>>(
                reinterpret_cast<const longlong2*>(a_ptr), reinterpret_cast<const longlong2*>(b_ptr),
                reinterpret_cast<longlong2*>(c_ptr), n_vec, op);
        }

        if (tail > 0) {
            elementwise_scalar_kernel<T, Op><<<elem_blocks(tail), 256>>>(
                a_ptr + n_vec * lanes, b_ptr + n_vec * lanes, c_ptr + n_vec * lanes, tail, op);
        }
    } else {
        elementwise_scalar_kernel<T, Op><<<elem_blocks(n), 256>>>(a_ptr, b_ptr, c_ptr, n, op);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Per-op dispatch — unchanged from the original, just now benefiting from
// the optimized run_cuda_elementwise_typed above.
// ---------------------------------------------------------------------------

std::shared_ptr<Tensor> run_cuda_add_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cuda_elementwise_typed<double>(a, b, AddOp{});
        case DType::INT32:   return run_cuda_elementwise_typed<int32_t>(a, b, AddOp{});
        case DType::INT64:   return run_cuda_elementwise_typed<int64_t>(a, b, AddOp{});
        default: throw std::runtime_error("add(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}
std::shared_ptr<Tensor> run_cuda_sub_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cuda_elementwise_typed<double>(a, b, SubOp{});
        case DType::INT32:   return run_cuda_elementwise_typed<int32_t>(a, b, SubOp{});
        case DType::INT64:   return run_cuda_elementwise_typed<int64_t>(a, b, SubOp{});
        default: throw std::runtime_error("sub(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}
std::shared_ptr<Tensor> run_cuda_mul_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cuda_elementwise_typed<double>(a, b, MulOp{});
        case DType::INT32:   return run_cuda_elementwise_typed<int32_t>(a, b, MulOp{});
        case DType::INT64:   return run_cuda_elementwise_typed<int64_t>(a, b, MulOp{});
        default: throw std::runtime_error("mul(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}
std::shared_ptr<Tensor> run_cuda_div_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cuda_elementwise_typed<double>(a, b, DivOp{});
        case DType::INT32:   return run_cuda_elementwise_typed<int32_t>(a, b, DivOp{});
        case DType::INT64:   return run_cuda_elementwise_typed<int64_t>(a, b, DivOp{});
        default: throw std::runtime_error("div(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}