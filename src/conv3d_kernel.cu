#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include "tensor.h"

__global__ void im2col_3d_kernel(const float* __restrict__ x, float* __restrict__ col,
                                  int B, int C, int D, int H, int W,
                                  int KD, int KH, int KW, int SD, int SH, int SW,
                                  int PD, int PH, int PW, int DD, int DH, int DW,
                                  int OD, int OH, int OW) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)B * C * KD * KH * KW * OD * OH * OW;
    if (idx >= total) return;

    long long tmp = idx;
    int ow = tmp % OW; tmp /= OW;
    int oh = tmp % OH; tmp /= OH;
    int od = tmp % OD; tmp /= OD;
    int kw = tmp % KW; tmp /= KW;
    int kh = tmp % KH; tmp /= KH;
    int kd = tmp % KD; tmp /= KD;
    int c = tmp % C;
    int b = tmp / C;

    int in_d = od * SD - PD + kd * DD;
    int in_h = oh * SH - PH + kh * DH;
    int in_w = ow * SW - PW + kw * DW;
    float val = 0.0f;
    if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W)
        val = x[((((size_t)b * C + c) * D + in_d) * H + in_h) * W + in_w];
    col[idx] = val;
}

__global__ void col2im_3d_kernel(const float* __restrict__ grad_col, float* __restrict__ grad_x,
                                  int B, int C, int D, int H, int W,
                                  int KD, int KH, int KW, int SD, int SH, int SW,
                                  int PD, int PH, int PW, int DD, int DH, int DW,
                                  int OD, int OH, int OW) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)B * C * D * H * W;
    if (idx >= total) return;

    long long tmp = idx;
    int iw = tmp % W; tmp /= W;
    int ih = tmp % H; tmp /= H;
    int id = tmp % D; tmp /= D;
    int c = tmp % C;
    int b = tmp / C;

    const float* gc_base = grad_col + ((size_t)(b * C + c)) * KD * KH * KW * OD * OH * OW;
    float acc = 0.0f;
    for (int kd = 0; kd < KD; ++kd) {
        int numer_d = id + PD - kd * DD;
        if (numer_d < 0 || numer_d % SD != 0) continue;
        int od = numer_d / SD;
        if (od < 0 || od >= OD) continue;
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
                int k_idx = (kd * KH + kh) * KW + kw;
                acc += gc_base[((size_t)k_idx * OD + od) * OH * OW + oh * OW + ow];
            }
        }
    }
    grad_x[idx] = acc;
}

static int conv3d_blocks(long long n) {
    int threads = 256;
    long long blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;
    return (int)blocks;
}

void run_cuda_im2col_3d(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                         int KD, int KH, int KW, int SD, int SH, int SW,
                         int PD, int PH, int PW, int DD, int DH, int DW,
                         int OD, int OH, int OW) {
    int B = x->shape[0], C = x->shape[1], D = x->shape[2], H = x->shape[3], W = x->shape[4];
    long long total = (long long)B * C * KD * KH * KW * OD * OH * OW;
    if (total == 0) return;
    im2col_3d_kernel<<<conv3d_blocks(total), 256>>>(
        x->fptr(), col->fptr(), B, C, D, H, W, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
}

void run_cuda_col2im_3d(std::shared_ptr<Tensor> grad_col, std::shared_ptr<Tensor> grad_x,
                         int B, int C, int D, int H, int W,
                         int KD, int KH, int KW, int SD, int SH, int SW,
                         int PD, int PH, int PW, int DD, int DH, int DW,
                         int OD, int OH, int OW) {
    long long total = (long long)B * C * D * H * W;
    if (total == 0) return;
    col2im_3d_kernel<<<conv3d_blocks(total), 256>>>(
        grad_col->fptr(), grad_x->fptr(), B, C, D, H, W, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
}