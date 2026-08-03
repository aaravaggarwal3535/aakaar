#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include "tensor.h"

template <typename T>
__global__ void im2col_2d_kernel(const T* __restrict__ x, T* __restrict__ col,
                                  int B, int C, int H, int W, int KH, int KW,
                                  int SH, int SW, int PH, int PW, int DH, int DW,
                                  int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * C * KH * KW * OH * OW;
    if (idx >= total) return;

    int ow = idx % OW;
    long long tmp = idx / OW;
    int oh = tmp % OH; tmp /= OH;
    int kw = tmp % KW; tmp /= KW;
    int kh = tmp % KH; tmp /= KH;
    int c = tmp % C;
    int b = tmp / C;

    int in_h = oh * SH - PH + kh * DH;
    int in_w = ow * SW - PW + kw * DW;
    T val = T(0);
    if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W)
        val = x[(((size_t)b * C + c) * H + in_h) * W + in_w];
    col[idx] = val;
}

template <typename T>
__global__ void col2im_2d_kernel(const T* __restrict__ grad_col, T* __restrict__ grad_x,
                                  int B, int C, int H, int W, int KH, int KW,
                                  int SH, int SW, int PH, int PW, int DH, int DW,
                                  int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * C * H * W;
    if (idx >= total) return;

    int iw = idx % W;
    long long tmp = idx / W;
    int ih = tmp % H; tmp /= H;
    int c = tmp % C;
    int b = tmp / C;

    const T* gc_base = grad_col + ((size_t)(b * C + c)) * KH * KW * OH * OW;
    T acc = T(0);
    for (int kh = 0; kh < KH; ++kh) {
        int numer_h = ih + PH - kh * DH;
        if (numer_h < 0 || numer_h % SH != 0) continue;
        int oh = numer_h / SH;
        if (oh < 0 || oh >= OH) continue;
        for (int kw = 0; kw < KW; ++kw) {
            int numer_w = iw + PW - kw * DW;
            if (numer_w < 0 || numer_w % SW != 0) continue;
            int ow = numer_w / SW;
            if (ow < 0 || ow >= OW) continue;
            int k_idx = kh * KW + kw;
            acc += gc_base[(size_t)k_idx * OH * OW + oh * OW + ow];
        }
    }
    grad_x[idx] = acc;
}

static int conv2d_blocks(long long n) {
    int threads = 256;
    long long blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;
    return (int)blocks;
}

void run_cuda_im2col_2d(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                         int KH, int KW, int SH, int SW, int PH, int PW,
                         int DH, int DW, int OH, int OW) {
    int B = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    long long total = (long long)B * C * KH * KW * OH * OW;
    if (total == 0) return;
    im2col_2d_kernel<float><<<conv2d_blocks(total), 256>>>(
        x->fptr(), col->fptr(), B, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
}

void run_cuda_col2im_2d(std::shared_ptr<Tensor> grad_col, std::shared_ptr<Tensor> grad_x,
                         int B, int C, int H, int W, int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW, int OH, int OW) {
    long long total = (long long)B * C * H * W;
    if (total == 0) return;
    col2im_2d_kernel<float><<<conv2d_blocks(total), 256>>>(
        grad_col->fptr(), grad_x->fptr(), B, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
}

void run_cuda_im2col_2d_typed(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                               int KH, int KW, int SH, int SW, int PH, int PW,
                               int DH, int DW, int OH, int OW) {
    int B = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    long long total = (long long)B * C * KH * KW * OH * OW;
    if (total == 0) return;
    switch (x->dtype) {
        case DType::FLOAT64:
            im2col_2d_kernel<double><<<conv2d_blocks(total), 256>>>(
                static_cast<const double*>(x->data_ptr), static_cast<double*>(col->data_ptr),
                B, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);
            break;
        case DType::INT32:
            im2col_2d_kernel<int32_t><<<conv2d_blocks(total), 256>>>(
                static_cast<const int32_t*>(x->data_ptr), static_cast<int32_t*>(col->data_ptr),
                B, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);
            break;
        case DType::INT64:
            im2col_2d_kernel<int64_t><<<conv2d_blocks(total), 256>>>(
                static_cast<const int64_t*>(x->data_ptr), static_cast<int64_t*>(col->data_ptr),
                B, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);
            break;
        default:
            throw std::runtime_error("im2col_2d: unsupported dtype '" + dtype_name(x->dtype) + "'");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
}