#include "tensor.h"
#include <omp.h>
#include <cstdint>
#include <stdexcept>

// =============================================================================
// im2col_1d / col2im_1d — CPU
//
// col2im is implemented as a GATHER (each output/input element is written by
// exactly one thread, looping over the K positions that could have touched
// it) rather than a scatter-add. This avoids any need for locks/atomics and
// keeps the OpenMP parallelization trivially correct.
// =============================================================================

void run_cpu_im2col_1d(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                        int K, int S, int P, int D, int L_out) {
    int B = x->shape[0], C = x->shape[1], L_in = x->shape[2];
    const float* xp = x->fptr();
    float* cp = col->fptr();

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const float* x_row = xp + ((size_t)b * C + c) * L_in;
            for (int k = 0; k < K; ++k) {
                float* col_row = cp + (((size_t)b * C + c) * K + k) * L_out;
                for (int o = 0; o < L_out; ++o) {
                    int in_pos = o * S - P + k * D;
                    col_row[o] = (in_pos >= 0 && in_pos < L_in) ? x_row[in_pos] : 0.0f;
                }
            }
        }
    }
}

void run_cpu_col2im_1d(std::shared_ptr<Tensor> grad_col, std::shared_ptr<Tensor> grad_x,
                        int B, int C, int L_in, int K, int S, int P, int D, int L_out) {
    const float* gc = grad_col->fptr();
    float* gx = grad_x->fptr();

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            float* gx_row = gx + ((size_t)b * C + c) * L_in;
            const float* gc_base = gc + ((size_t)b * C + c) * K * L_out;
            for (int i = 0; i < L_in; ++i) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    int numer = i + P - k * D;
                    if (numer < 0 || numer % S != 0) continue;
                    int o = numer / S;
                    if (o < 0 || o >= L_out) continue;
                    acc += gc_base[(size_t)k * L_out + o];
                }
                gx_row[i] = acc;
            }
        }
    }
}

// ---- typed path: float64 / int32 / int64 (forward-only — no autograd) ----

template <typename T>
static void im2col_1d_typed_impl(const T* xp, T* cp, int B, int C, int L_in,
                                  int K, int S, int P, int D, int L_out) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* x_row = xp + ((size_t)b * C + c) * L_in;
            for (int k = 0; k < K; ++k) {
                T* col_row = cp + (((size_t)b * C + c) * K + k) * L_out;
                for (int o = 0; o < L_out; ++o) {
                    int in_pos = o * S - P + k * D;
                    col_row[o] = (in_pos >= 0 && in_pos < L_in) ? x_row[in_pos] : T(0);
                }
            }
        }
    }
}

void run_cpu_im2col_1d_typed(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                              int K, int S, int P, int D, int L_out) {
    int B = x->shape[0], C = x->shape[1], L_in = x->shape[2];
    switch (x->dtype) {
        case DType::FLOAT64:
            im2col_1d_typed_impl<double>(static_cast<const double*>(x->data_ptr),
                                          static_cast<double*>(col->data_ptr), B, C, L_in, K, S, P, D, L_out);
            break;
        case DType::INT32:
            im2col_1d_typed_impl<int32_t>(static_cast<const int32_t*>(x->data_ptr),
                                           static_cast<int32_t*>(col->data_ptr), B, C, L_in, K, S, P, D, L_out);
            break;
        case DType::INT64:
            im2col_1d_typed_impl<int64_t>(static_cast<const int64_t*>(x->data_ptr),
                                           static_cast<int64_t*>(col->data_ptr), B, C, L_in, K, S, P, D, L_out);
            break;
        default:
            throw std::runtime_error("im2col_1d: unsupported dtype '" + dtype_name(x->dtype) + "'");
    }
}