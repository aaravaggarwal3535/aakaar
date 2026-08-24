#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include "tensor.h"

template <typename T>
__global__ void im2col_1d_kernel(const T* __restrict__ x, T* __restrict__ col,
                                  int B, int C, int L_in, int K, int S, int P, int D, int L_out) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    long long total = (long long)B * C * K * L_out;

    for (; idx < total; idx += stride) {
        int o = idx % L_out;
        long long tmp = idx / L_out;
        int k = tmp % K; tmp /= K;
        int c = tmp % C;
        int b = tmp / C;

        int in_pos = o * S - P + k * D;
        T val = T(0);
        if (in_pos >= 0 && in_pos < L_in) val = x[((size_t)b * C + c) * L_in + in_pos];
        col[idx] = val;
    }
}

template <typename T>
__global__ void col2im_1d_kernel(const T* __restrict__ grad_col, T* __restrict__ grad_x,
                                  int B, int C, int L_in, int K, int S, int P, int D, int L_out) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    long long total = (long long)B * C * L_in;

    for (; idx < total; idx += stride) {
        int i = idx % L_in;
        long long tmp = idx / L_in;
        int c = tmp % C;
        int b = tmp / C;

        const T* gc_base = grad_col + ((size_t)(b * C + c) * K) * L_out;
        T acc = T(0);
        for (int k = 0; k < K; ++k) {
            int numer = i + P - k * D;
            if (numer < 0 || numer % S != 0) continue;
            int o = numer / S;
            if (o < 0 || o >= L_out) continue;
            acc += gc_base[(size_t)k * L_out + o];
        }
        grad_x[idx] = acc;
    }
}

static int conv_blocks(int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;
    return blocks;
}

void run_cuda_im2col_1d(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                         int K, int S, int P, int D, int L_out) {
    int B = x->shape[0], C = x->shape[1], L_in = x->shape[2];
    int total = B * C * K * L_out;
    if (total == 0) return;
    im2col_1d_kernel<float><<<conv_blocks(total), 256>>>(
        x->fptr(), col->fptr(), B, C, L_in, K, S, P, D, L_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
}

void run_cuda_col2im_1d(std::shared_ptr<Tensor> grad_col, std::shared_ptr<Tensor> grad_x,
                         int B, int C, int L_in, int K, int S, int P, int D, int L_out) {
    int total = B * C * L_in;
    if (total == 0) return;
    col2im_1d_kernel<float><<<conv_blocks(total), 256>>>(
        grad_col->fptr(), grad_x->fptr(), B, C, L_in, K, S, P, D, L_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
}

void run_cuda_im2col_1d_typed(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                               int K, int S, int P, int D, int L_out) {
    int B = x->shape[0], C = x->shape[1], L_in = x->shape[2];
    int total = B * C * K * L_out;
    if (total == 0) return;
    switch (x->dtype) {
        case DType::FLOAT64:
            im2col_1d_kernel<double><<<conv_blocks(total), 256>>>(
                static_cast<const double*>(x->data_ptr), static_cast<double*>(col->data_ptr), B, C, L_in, K, S, P, D, L_out);
            break;
        case DType::INT32:
            im2col_1d_kernel<int32_t><<<conv_blocks(total), 256>>>(
                static_cast<const int32_t*>(x->data_ptr), static_cast<int32_t*>(col->data_ptr), B, C, L_in, K, S, P, D, L_out);
            break;
        case DType::INT64:
            im2col_1d_kernel<int64_t><<<conv_blocks(total), 256>>>(
                static_cast<const int64_t*>(x->data_ptr), static_cast<int64_t*>(col->data_ptr), B, C, L_in, K, S, P, D, L_out);
            break;
        default:
            throw std::runtime_error("im2col_1d: unsupported dtype '" + dtype_name(x->dtype) + "'");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
}