#include "tensor.h"
#include <omp.h>
#include <cstdint>
#include <stdexcept>

// im2col_2d: (B, C, H, W) -> (B, C*KH*KW, OH*OW)
void run_cpu_im2col_2d(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                        int KH, int KW, int SH, int SW, int PH, int PW,
                        int DH, int DW, int OH, int OW) {
    int B = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    const float* xp = x->fptr();
    float* cp = col->fptr();

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const float* x_chan = xp + ((size_t)b * C + c) * H * W;
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int k_idx = kh * KW + kw;
                    float* col_row = cp + (((size_t)b * C + c) * KH * KW + k_idx) * OH * OW;
                    for (int oh = 0; oh < OH; ++oh) {
                        int in_h = oh * SH - PH + kh * DH;
                        for (int ow = 0; ow < OW; ++ow) {
                            int in_w = ow * SW - PW + kw * DW;
                            float val = 0.0f;
                            if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W)
                                val = x_chan[in_h * W + in_w];
                            col_row[oh * OW + ow] = val;
                        }
                    }
                }
            }
        }
    }
}

void run_cpu_col2im_2d(std::shared_ptr<Tensor> grad_col, std::shared_ptr<Tensor> grad_x,
                        int B, int C, int H, int W, int KH, int KW, int SH, int SW,
                        int PH, int PW, int DH, int DW, int OH, int OW) {
    const float* gc = grad_col->fptr();
    float* gx = grad_x->fptr();

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            float* gx_chan = gx + ((size_t)b * C + c) * H * W;
            const float* gc_base = gc + ((size_t)b * C + c) * KH * KW * OH * OW;
            for (int ih = 0; ih < H; ++ih) {
                for (int iw = 0; iw < W; ++iw) {
                    float acc = 0.0f;
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
                    gx_chan[ih * W + iw] = acc;
                }
            }
        }
    }
}

template <typename T>
static void im2col_2d_typed_impl(const T* xp, T* cp, int B, int C, int H, int W,
                                  int KH, int KW, int SH, int SW, int PH, int PW,
                                  int DH, int DW, int OH, int OW) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* x_chan = xp + ((size_t)b * C + c) * H * W;
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int k_idx = kh * KW + kw;
                    T* col_row = cp + (((size_t)b * C + c) * KH * KW + k_idx) * OH * OW;
                    for (int oh = 0; oh < OH; ++oh) {
                        int in_h = oh * SH - PH + kh * DH;
                        for (int ow = 0; ow < OW; ++ow) {
                            int in_w = ow * SW - PW + kw * DW;
                            T val = T(0);
                            if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W)
                                val = x_chan[in_h * W + in_w];
                            col_row[oh * OW + ow] = val;
                        }
                    }
                }
            }
        }
    }
}

void run_cpu_im2col_2d_typed(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                              int KH, int KW, int SH, int SW, int PH, int PW,
                              int DH, int DW, int OH, int OW) {
    int B = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    switch (x->dtype) {
        case DType::FLOAT64:
            im2col_2d_typed_impl<double>(static_cast<const double*>(x->data_ptr),
                static_cast<double*>(col->data_ptr), B, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);
            break;
        case DType::INT32:
            im2col_2d_typed_impl<int32_t>(static_cast<const int32_t*>(x->data_ptr),
                static_cast<int32_t*>(col->data_ptr), B, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);
            break;
        case DType::INT64:
            im2col_2d_typed_impl<int64_t>(static_cast<const int64_t*>(x->data_ptr),
                static_cast<int64_t*>(col->data_ptr), B, C, H, W, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);
            break;
        default:
            throw std::runtime_error("im2col_2d: unsupported dtype '" + dtype_name(x->dtype) + "'");
    }
}