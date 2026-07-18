#include <stdexcept>
#include <cuda_runtime.h>
#include <vector>

#define MAX_DIMS 8

struct ShapeInfo {
    int dims[MAX_DIMS];
    int strides[MAX_DIMS];
    int ndim;
};

__global__ void strided_gather_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                       ShapeInfo info, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int remaining = idx;
    int src_off = 0;
    for (int d = info.ndim - 1; d >= 0; --d) {
        int coord = remaining % info.dims[d];
        remaining /= info.dims[d];
        src_off += coord * info.strides[d];
    }
    dst[idx] = src[src_off];
}

void run_cuda_strided_gather(const float* src, float* dst, const std::vector<int>& shape,
                              const std::vector<int>& strides, int total_size) {
    ShapeInfo info;
    info.ndim = (int)shape.size();
    if (info.ndim > MAX_DIMS)
        throw std::runtime_error("strided_gather: tensor rank exceeds MAX_DIMS (8)");
    for (int i = 0; i < info.ndim; ++i) {
        info.dims[i] = shape[i];
        info.strides[i] = strides[i];
    }

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    strided_gather_kernel<<<blocks, threads>>>(src, dst, info, total_size);
}
