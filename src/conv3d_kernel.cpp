#include "tensor.h"
#include <omp.h>
#include <cstdint>
#include <stdexcept>

// im2col_3d: (B, C, D, H, W) -> (B, C*KD*KH*KW, OD*OH*OW). Float32 only —
// matches this codebase's convention elsewhere (Conv1d/Conv2d's typed
// int32/int64/float64 im2col variants exist but are forward-only/no-grad;
// omitted here for the initial Conv3d implementation since float32 is the
// only dtype with autograd support anyway).
void run_cpu_im2col_3d(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                        int KD, int KH, int KW, int SD, int SH, int SW,
                        int PD, int PH, int PW, int DD, int DH, int DW,
                        int OD, int OH, int OW) {
    int B = x->shape[0], C = x->shape[1], D = x->shape[2], H = x->shape[3], W = x->shape[4];
    const float* xp = x->fptr();
    float* cp = col->fptr();

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const float* x_chan = xp + ((size_t)b * C + c) * D * H * W;
            for (int kd = 0; kd < KD; ++kd) {
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
                        int k_idx = (kd * KH + kh) * KW + kw;
                        float* col_row = cp + (((size_t)b * C + c) * KD * KH * KW + k_idx) * OD * OH * OW;
                        for (int od = 0; od < OD; ++od) {
                            int in_d = od * SD - PD + kd * DD;
                            for (int oh = 0; oh < OH; ++oh) {
                                int in_h = oh * SH - PH + kh * DH;
                                for (int ow = 0; ow < OW; ++ow) {
                                    int in_w = ow * SW - PW + kw * DW;
                                    float val = 0.0f;
                                    if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W)
                                        val = x_chan[(in_d * H + in_h) * W + in_w];
                                    col_row[(od * OH + oh) * OW + ow] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void run_cpu_col2im_3d(std::shared_ptr<Tensor> grad_col, std::shared_ptr<Tensor> grad_x,
                        int B, int C, int D, int H, int W,
                        int KD, int KH, int KW, int SD, int SH, int SW,
                        int PD, int PH, int PW, int DD, int DH, int DW,
                        int OD, int OH, int OW) {
    const float* gc = grad_col->fptr();
    float* gx = grad_x->fptr();

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            float* gx_chan = gx + ((size_t)b * C + c) * D * H * W;
            const float* gc_base = gc + ((size_t)b * C + c) * KD * KH * KW * OD * OH * OW;
            for (int id = 0; id < D; ++id) {
                for (int ih = 0; ih < H; ++ih) {
                    for (int iw = 0; iw < W; ++iw) {
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
                        gx_chan[(id * H + ih) * W + iw] = acc;
                    }
                }
            }
        }
    }
}