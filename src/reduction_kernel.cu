#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <numeric>
#include "tensor.h"

// Naive but correct: one thread per output element, loop over the reduced axis.
// Not the fastest possible reduction (no tree/shared-memory optimization), but
// correct for any axis and any stride layout. Optimize later if profiling shows need.
__global__ void sum_axis_kernel(const float* __restrict__ in, float* __restrict__ out,
                                 int out_size, int reduce_size,
                                 int outer_stride, int reduce_stride, int inner_stride,
                                 int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;

    float acc = 0.0f;
    long long base = (long long)outer_idx * outer_stride + (long long)inner_idx * inner_stride;
    for (int r = 0; r < reduce_size; ++r) {
        acc += in[base + (long long)r * reduce_stride];
    }
    out[idx] = acc;
}

std::shared_ptr<Tensor> run_cuda_sum_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
    if (!a->is_contiguous())
        throw std::invalid_argument("sum() requires a contiguous tensor. Call .contiguous() first.");

    int ndim = (int)a->shape.size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::out_of_range("sum() dim out of range");

    int outer_size = 1, inner_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= a->shape[i];
    for (int i = dim + 1; i < ndim; ++i) inner_size *= a->shape[i];
    int reduce_size = a->shape[dim];
    int out_size = outer_size * inner_size;

    std::vector<int> out_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(a->shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    auto result = std::make_shared<Tensor>(out_shape, std::string("cuda"));

    int threads = 256;
    int blocks = (out_size + threads - 1) / threads;
    sum_axis_kernel<<<blocks, threads>>>(
        a->fptr(), result->fptr(), out_size, reduce_size,
        inner_size * reduce_size, inner_size, 1, inner_size
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

__global__ void sum_all_reduce_kernel(const float* __restrict__ in, float* __restrict__ partial_sums, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float sum = 0.0f;
    if (idx < n) sum += in[idx];
    if (idx + blockDim.x < n) sum += in[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

std::shared_ptr<Tensor> run_cuda_sum_all(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous())
        throw std::invalid_argument("sum() requires a contiguous tensor. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(std::vector<int>{1}, std::string("cuda"));
    int n = a->size;
    if (n == 0) { cudaMemset(result->fptr(), 0, sizeof(float)); return result; }

    int threads = 256;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    float* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));

    sum_all_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(a->fptr(), d_partial, n);

    // Reduce the (small) partials array with a second pass, or with a
    // final single-block call if it's already small enough.
    while (blocks > 1) {
        int n2 = blocks;
        int threads2 = 256;
        int blocks2 = (n2 + threads2 * 2 - 1) / (threads2 * 2);
        float* d_partial2;
        cudaMalloc(&d_partial2, blocks2 * sizeof(float));
        sum_all_reduce_kernel<<<blocks2, threads2, threads2 * sizeof(float)>>>(d_partial, d_partial2, n2);
        cudaFree(d_partial);
        d_partial = d_partial2;
        blocks = blocks2;
    }

    cudaMemcpy(result->fptr(), d_partial, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_partial);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

// Broadcast a reduced-size-1 axis back up to `target_size` by repeating values.
// This is sum's backward: gradient of a sum is broadcast to every summed element.
__global__ void broadcast_axis_kernel(const float* __restrict__ in, float* __restrict__ out,
                                       int out_size, int broadcast_size,
                                       int outer_stride_out, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;
    int outer_idx = idx / (broadcast_size * inner_size);
    int inner_idx = idx % inner_size;
    int in_idx = outer_idx * inner_size + inner_idx;
    out[idx] = in[in_idx];
}

std::shared_ptr<Tensor> run_cuda_broadcast_axis(std::shared_ptr<Tensor> a, int dim, int target_size) {
    if (!a->is_contiguous())
        throw std::invalid_argument("broadcast requires a contiguous tensor.");
    int ndim = (int)a->shape.size();
    if (dim < 0) dim += ndim;

    std::vector<int> out_shape = a->shape;
    out_shape[dim] = target_size;
    auto result = std::make_shared<Tensor>(out_shape, std::string("cuda"));

    int outer_size = 1, inner_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= a->shape[i];
    for (int i = dim + 1; i < ndim; ++i) inner_size *= a->shape[i];
    int out_size = outer_size * target_size * inner_size;

    int threads = 256;
    int blocks = (out_size + threads - 1) / threads;
    broadcast_axis_kernel<<<blocks, threads>>>(a->fptr(), result->fptr(), out_size, target_size, 0, inner_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}
// Append to reduction_kernel.cu

__global__ void max_axis_kernel(const float* __restrict__ in, float* __restrict__ out, int* __restrict__ argmax,
                                 int out_size, int reduce_size,
                                 int outer_stride, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;

    long long base = (long long)outer_idx * outer_stride + (long long)inner_idx;
    float best = in[base];
    int best_r = 0;
    for (int r = 1; r < reduce_size; ++r) {
        float v = in[base + (long long)r * inner_size];
        if (v > best) { best = v; best_r = r; }
    }
    out[idx] = best;
    argmax[idx] = best_r;
}

// Scatters grad_out into a zero tensor at the argmax positions only —
// this IS max's backward, no separate "unscatter" needed elsewhere.
__global__ void max_axis_backward_kernel(const float* __restrict__ grad_out, const int* __restrict__ argmax,
                                          float* __restrict__ grad_in,
                                          int out_size, int reduce_size, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    int r = argmax[idx];

    long long in_idx = (long long)outer_idx * reduce_size * inner_size + (long long)r * inner_size + inner_idx;
    grad_in[in_idx] = grad_out[idx];
}

// Returns {values, argmax_indices} as a pair — argmax stored in a plain int buffer,
// not wrapped as a Tensor since it's an internal implementation detail for backward,
// not user-facing data.
std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cuda_max_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
    if (!a->is_contiguous())
        throw std::invalid_argument("max() requires a contiguous tensor. Call .contiguous() first.");

    int ndim = (int)a->shape.size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::out_of_range("max() dim out of range");

    int outer_size = 1, inner_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= a->shape[i];
    for (int i = dim + 1; i < ndim; ++i) inner_size *= a->shape[i];
    int reduce_size = a->shape[dim];
    int out_size = outer_size * inner_size;

    std::vector<int> out_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(a->shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    auto result = std::make_shared<Tensor>(out_shape, std::string("cuda"));

    int* d_argmax;
    cudaMalloc(&d_argmax, out_size * sizeof(int));

    int threads = 256;
    int blocks = (out_size + threads - 1) / threads;
    max_axis_kernel<<<blocks, threads>>>(a->fptr(), result->fptr(), d_argmax,
                                          out_size, reduce_size, inner_size * reduce_size, inner_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(d_argmax); throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err)); }

    std::vector<int> h_argmax(out_size);
    cudaMemcpy(h_argmax.data(), d_argmax, out_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_argmax);

    return {result, h_argmax};
}

std::shared_ptr<Tensor> run_cuda_max_axis_backward(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                    std::vector<int> orig_shape, int dim, int reduce_size, int inner_size) {
    auto grad_in = std::make_shared<Tensor>(orig_shape, std::string("cuda"));
    grad_in->fill_zero();

    int out_size = (int)argmax.size();
    int* d_argmax;
    cudaMalloc(&d_argmax, out_size * sizeof(int));
    cudaMemcpy(d_argmax, argmax.data(), out_size * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (out_size + threads - 1) / threads;
    max_axis_backward_kernel<<<blocks, threads>>>(grad_out->fptr(), d_argmax, grad_in->fptr(),
                                                    out_size, reduce_size, inner_size);
    cudaFree(d_argmax);
    return grad_in;
}

template <typename InT, typename AccT>
__global__ void sum_axis_kernel_typed(const InT* __restrict__ in, AccT* __restrict__ out,
                                       int out_size, int reduce_size,
                                       int outer_stride, int inner_stride, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < out_size; i += stride) {
        int outer_idx = i / inner_size;
        int inner_idx = i % inner_size;
        long long base = (long long)outer_idx * outer_stride + (long long)inner_idx * inner_stride;

        AccT acc = AccT(0);
        int r = 0;
        
        // Loop unrolling for Instruction Level Parallelism (ILP)
        #pragma unroll
        for (; r <= reduce_size - 4; r += 4) {
            acc += (AccT)in[base + (long long)(r) * inner_size];
            acc += (AccT)in[base + (long long)(r + 1) * inner_size];
            acc += (AccT)in[base + (long long)(r + 2) * inner_size];
            acc += (AccT)in[base + (long long)(r + 3) * inner_size];
        }
        for (; r < reduce_size; ++r) {
            acc += (AccT)in[base + (long long)r * inner_size];
        }
        out[i] = acc;
    }
}

template <typename InT, typename AccT>
static std::shared_ptr<Tensor> run_cuda_sum_axis_typed_impl(std::shared_ptr<Tensor> a, int dim, bool keepdim, DType out_dtype) {
    if (!a->is_contiguous())
        throw std::invalid_argument("sum() requires a contiguous tensor. Call .contiguous() first.");

    int ndim = (int)a->shape.size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::out_of_range("sum() dim out of range");

    int outer_size = 1, inner_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= a->shape[i];
    for (int i = dim + 1; i < ndim; ++i) inner_size *= a->shape[i];
    int reduce_size = a->shape[dim];
    int out_size = outer_size * inner_size;

    std::vector<int> out_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(a->shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    auto result = std::make_shared<Tensor>(out_shape, std::string("cuda"), out_dtype);

    int threads = 256;
    int blocks = std::min((out_size + threads - 1) / threads, 4096);
    sum_axis_kernel_typed<InT, AccT><<<blocks, threads>>>(
        static_cast<const InT*>(a->data_ptr), static_cast<AccT*>(result->data_ptr),
        out_size, reduce_size, inner_size * reduce_size, inner_size, inner_size
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

std::shared_ptr<Tensor> run_cuda_sum_axis_typed(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cuda_sum_axis_typed_impl<double, double>(a, dim, keepdim, DType::FLOAT64);
        case DType::INT32:   return run_cuda_sum_axis_typed_impl<int32_t, int64_t>(a, dim, keepdim, DType::INT64);
        case DType::INT64:   return run_cuda_sum_axis_typed_impl<int64_t, int64_t>(a, dim, keepdim, DType::INT64);
        default: throw std::runtime_error("sum(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}

// Added GPU block-reduction kernel replacing the slow Host round-trip
// Helper function to safely handle atomicAdd for both double and int64_t
__device__ __forceinline__ void safeAtomicAdd(double* address, double val) {
    atomicAdd(address, val);
}

__device__ __forceinline__ void safeAtomicAdd(int64_t* address, int64_t val) {
    // Two's complement addition is identical for signed and unsigned.
    // Casting to unsigned long long int is perfectly safe and standard practice here.
    atomicAdd(reinterpret_cast<unsigned long long int*>(address), 
              static_cast<unsigned long long int>(val));
}

template <typename InT, typename AccT>
__global__ void sum_all_kernel_typed(const InT* __restrict__ in, AccT* __restrict__ out, int size) {
    extern __shared__ char smem[];
    AccT* sdata = reinterpret_cast<AccT*>(smem);

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    AccT mySum = 0;
    while (i < size) {
        mySum += (AccT)in[i];
        if (i + blockDim.x < size) {
            mySum += (AccT)in[i + blockDim.x];
        }
        i += gridSize;
    }
    sdata[tid] = mySum;
    __syncthreads();

    // Shared memory block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Replaced atomicAdd with our custom wrapper
        safeAtomicAdd(out, sdata[0]);
    }
}

template <typename InT, typename AccT>
static std::shared_ptr<Tensor> run_cuda_sum_all_typed_impl(std::shared_ptr<Tensor> a, DType out_dtype) {
    if (!a->is_contiguous())
        throw std::invalid_argument("sum() requires a contiguous tensor. Call .contiguous() first.");
    
    auto result = std::make_shared<Tensor>(std::vector<int>{1}, std::string("cuda"), out_dtype);
    int size = a->size;
    
    // Clear out atomic receiver
    cudaMemset(result->data_ptr, 0, sizeof(AccT));
    if (size == 0) return result;

    int threads = 256;
    int blocks = std::min(1024, (size + (threads * 2) - 1) / (threads * 2));
    size_t smem_size = threads * sizeof(AccT);

    sum_all_kernel_typed<InT, AccT><<<blocks, threads, smem_size>>>(
        static_cast<const InT*>(a->data_ptr), static_cast<AccT*>(result->data_ptr), size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    return result;
}

std::shared_ptr<Tensor> run_cuda_sum_all_typed(std::shared_ptr<Tensor> a) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cuda_sum_all_typed_impl<double, double>(a, DType::FLOAT64);
        case DType::INT32:   return run_cuda_sum_all_typed_impl<int32_t, int64_t>(a, DType::INT64);
        case DType::INT64:   return run_cuda_sum_all_typed_impl<int64_t, int64_t>(a, DType::INT64);
        default: throw std::runtime_error("sum(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}

template <typename T>
__global__ void max_axis_kernel_typed(const T* __restrict__ in, T* __restrict__ out, int* __restrict__ argmax,
                                       int out_size, int reduce_size, int outer_stride, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < out_size; i += stride) {
        int outer_idx = i / inner_size;
        int inner_idx = i % inner_size;

        long long base = (long long)outer_idx * outer_stride + (long long)inner_idx;
        T best = in[base];
        int best_r = 0;
        int r = 1;
        
        #pragma unroll
        for (; r <= reduce_size - 4; r += 4) {
            T v1 = in[base + (long long)(r) * inner_size];
            T v2 = in[base + (long long)(r + 1) * inner_size];
            T v3 = in[base + (long long)(r + 2) * inner_size];
            T v4 = in[base + (long long)(r + 3) * inner_size];

            if (v1 > best) { best = v1; best_r = r; }
            if (v2 > best) { best = v2; best_r = r + 1; }
            if (v3 > best) { best = v3; best_r = r + 2; }
            if (v4 > best) { best = v4; best_r = r + 3; }
        }
        for (; r < reduce_size; ++r) {
            T v = in[base + (long long)r * inner_size];
            if (v > best) { best = v; best_r = r; }
        }
        out[i] = best;
        argmax[i] = best_r;
    }
}

template <typename T>
__global__ void max_axis_backward_kernel_typed(const T* __restrict__ grad_out, const int* __restrict__ argmax,
                                                T* __restrict__ grad_in, int out_size, int reduce_size, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < out_size; i += stride) {
        int outer_idx = i / inner_size;
        int inner_idx = i % inner_size;
        int r = argmax[i];

        long long in_idx = (long long)outer_idx * reduce_size * inner_size + (long long)r * inner_size + inner_idx;
        grad_in[in_idx] = grad_out[i];
    }
}

template <typename T>
static std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cuda_max_axis_typed_impl(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
    if (!a->is_contiguous())
        throw std::invalid_argument("max() requires a contiguous tensor. Call .contiguous() first.");

    int ndim = (int)a->shape.size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::out_of_range("max() dim out of range");

    int outer_size = 1, inner_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= a->shape[i];
    for (int i = dim + 1; i < ndim; ++i) inner_size *= a->shape[i];
    int reduce_size = a->shape[dim];
    int out_size = outer_size * inner_size;

    std::vector<int> out_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(a->shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    auto result = std::make_shared<Tensor>(out_shape, std::string("cuda"), a->dtype);

    int* d_argmax;
    cudaMalloc(&d_argmax, out_size * sizeof(int));

    int threads = 256;
    int blocks = std::min((out_size + threads - 1) / threads, 4096);
    
    max_axis_kernel_typed<T><<<blocks, threads>>>(
        static_cast<const T*>(a->data_ptr), static_cast<T*>(result->data_ptr), d_argmax,
        out_size, reduce_size, inner_size * reduce_size, inner_size);
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { 
        cudaFree(d_argmax); 
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err)); 
    }

    std::vector<int> h_argmax(out_size);
    cudaMemcpy(h_argmax.data(), d_argmax, out_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_argmax);

    return {result, h_argmax};
}

std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cuda_max_axis_typed(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cuda_max_axis_typed_impl<double>(a, dim, keepdim);
        case DType::INT32:   return run_cuda_max_axis_typed_impl<int32_t>(a, dim, keepdim);
        case DType::INT64:   return run_cuda_max_axis_typed_impl<int64_t>(a, dim, keepdim);
        default: throw std::runtime_error("max(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}

template <typename T>
static std::shared_ptr<Tensor> run_cuda_max_axis_backward_typed_impl(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                                      std::vector<int> orig_shape, int dim, int reduce_size, int inner_size, DType dt) {
    auto grad_in = std::make_shared<Tensor>(orig_shape, std::string("cuda"), dt);
    
    // Fill grad_in with zeros first (max pooling backward logic)
    cudaMemset(grad_in->data_ptr, 0, grad_in->size * sizeof(T));

    int out_size = (int)argmax.size();
    if (out_size == 0) return grad_in;
    
    int* d_argmax;
    cudaMalloc(&d_argmax, out_size * sizeof(int));
    cudaMemcpy(d_argmax, argmax.data(), out_size * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = std::min((out_size + threads - 1) / threads, 4096);
    max_axis_backward_kernel_typed<T><<<blocks, threads>>>(
        static_cast<const T*>(grad_out->data_ptr), d_argmax, static_cast<T*>(grad_in->data_ptr),
        out_size, reduce_size, inner_size);
        
    cudaFree(d_argmax);
    return grad_in;
}

std::shared_ptr<Tensor> run_cuda_max_axis_backward_typed(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                          std::vector<int> orig_shape, int dim, int reduce_size, int inner_size) {
    switch (grad_out->dtype) {
        case DType::FLOAT64: return run_cuda_max_axis_backward_typed_impl<double>(grad_out, argmax, orig_shape, dim, reduce_size, inner_size, DType::FLOAT64);
        case DType::INT32:   return run_cuda_max_axis_backward_typed_impl<int32_t>(grad_out, argmax, orig_shape, dim, reduce_size, inner_size, DType::INT32);
        case DType::INT64:   return run_cuda_max_axis_backward_typed_impl<int64_t>(grad_out, argmax, orig_shape, dim, reduce_size, inner_size, DType::INT64);
        default: throw std::runtime_error("max() backward: unsupported dtype '" + dtype_name(grad_out->dtype) + "'");
    }
}