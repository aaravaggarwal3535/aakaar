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
        a->data_ptr, result->data_ptr, out_size, reduce_size,
        inner_size * reduce_size, inner_size, 1, inner_size
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    cudaDeviceSynchronize();
    return result;
}

std::shared_ptr<Tensor> run_cuda_sum_all(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous())
        throw std::invalid_argument("sum() requires a contiguous tensor. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(std::vector<int>{1}, std::string("cuda"));
    // Simple approach: copy to host, reduce, copy back. Fine for now; optimize with a
    // proper parallel reduction kernel later if this becomes a hot path.
    std::vector<float> buf(a->size);
    cudaMemcpy(buf.data(), a->data_ptr, a->size * sizeof(float), cudaMemcpyDeviceToHost);
    float acc = 0.0f;
    for (float v : buf) acc += v;
    cudaMemcpy(result->data_ptr, &acc, sizeof(float), cudaMemcpyHostToDevice);
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
    broadcast_axis_kernel<<<blocks, threads>>>(a->data_ptr, result->data_ptr, out_size, target_size, 0, inner_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    cudaDeviceSynchronize();
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
    max_axis_kernel<<<blocks, threads>>>(a->data_ptr, result->data_ptr, d_argmax,
                                          out_size, reduce_size, inner_size * reduce_size, inner_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(d_argmax); throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err)); }
    cudaDeviceSynchronize();

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
    max_axis_backward_kernel<<<blocks, threads>>>(grad_out->data_ptr, d_argmax, grad_in->data_ptr,
                                                    out_size, reduce_size, inner_size);
    cudaFree(d_argmax);
    cudaDeviceSynchronize();
    return grad_in;
}