#include <cmath>
#include <random>
#include <memory>
#include <stdexcept>
#include "tensor.h"
#include <omp.h>
#include <cblas.h>

extern "C" char* openblas_get_config(void);
extern "C" char* openblas_get_corename(void);

std::string get_openblas_diagnostic() {
    return std::string("Config: ") + openblas_get_config() + " | Core: " + openblas_get_corename();
}

void fill_cpu_random(std::shared_ptr<Tensor> t, unsigned long long seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < t->size; ++i) {
        t->data_ptr[i] = dis(gen);
    }
}

static bool compute_broadcast_plan_cpu(const std::vector<int>& sa, const std::vector<int>& sta,
                                        const std::vector<int>& sb, const std::vector<int>& stb,
                                        std::vector<int>& out_shape,
                                        std::vector<int>& pa_shape, std::vector<int>& pa_strides,
                                        std::vector<int>& pb_shape, std::vector<int>& pb_strides) {
    int nd = std::max(sa.size(), sb.size());
    out_shape.resize(nd); pa_shape.resize(nd); pa_strides.resize(nd);
    pb_shape.resize(nd); pb_strides.resize(nd);
    for (int i = 0; i < nd; ++i) {
        int ai = (int)sa.size() - nd + i;
        int bi = (int)sb.size() - nd + i;
        int da = ai >= 0 ? sa[ai] : 1;
        int db = bi >= 0 ? sb[bi] : 1;
        if (da != db && da != 1 && db != 1) return false;
        out_shape[i] = std::max(da, db);
        pa_shape[i] = da; pa_strides[i] = ai >= 0 ? sta[ai] : 0;
        pb_shape[i] = db; pb_strides[i] = bi >= 0 ? stb[bi] : 0;
    }
    return true;
}

std::shared_ptr<Tensor> run_cpu_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (!a->is_contiguous() || !b->is_contiguous())
        throw std::invalid_argument("matmul requires contiguous tensors. Call .contiguous() first.");

    auto batch_dims = [](const std::vector<int>& s) { return std::vector<int>(s.begin(), s.end()-2); };
    int M = a->shape[a->shape.size()-2], K = a->shape.back();
    int K2 = b->shape[b->shape.size()-2], N = b->shape.back();
    if (K != K2) throw std::invalid_argument("Shape mismatch: inner dimensions must match");

    auto ba = batch_dims(a->shape), bb = batch_dims(b->shape);
    int ndb = std::max(ba.size(), bb.size());
    std::vector<int> out_batch(ndb);
    for (int i = 0; i < ndb; ++i) {
        int da = i < (int)ba.size() ? ba[ba.size()-1-i] : 1;
        int db = i < (int)bb.size() ? bb[bb.size()-1-i] : 1;
        if (da != db && da != 1 && db != 1)
            throw std::invalid_argument("Batch dimensions are not broadcastable for matmul");
        out_batch[ndb-1-i] = std::max(da, db);
    }
    int total_batch = 1;
    for (int d : out_batch) total_batch *= d;

    std::vector<int> out_shape = out_batch;
    out_shape.push_back(M); out_shape.push_back(N);
    auto result = std::make_shared<Tensor>(out_shape, std::string("cpu"));

    auto compute_offset = [&](int flat_idx, const std::vector<int>& in_shape, int mat_size) {
        auto in_batch = batch_dims(in_shape);
        int nd_in = (int)in_batch.size();
        std::vector<int> idx(ndb);
        int rem = flat_idx;
        for (int i = ndb - 1; i >= 0; --i) { idx[i] = rem % out_batch[i]; rem /= out_batch[i]; }
        long long off = 0, stride = mat_size;
        for (int i = ndb - 1; i >= 0; --i) {
            int in_i = i - (ndb - nd_in);
            int in_dim = in_i >= 0 ? in_batch[in_i] : 1;
            int use_idx = (in_dim == 1) ? 0 : idx[i];
            if (in_i >= 0) { off += (long long)use_idx * stride; stride *= in_dim; }
        }
        return (int)off;
    };

    for (int bi = 0; bi < total_batch; ++bi) {
        const float* ap = a->data_ptr + compute_offset(bi, a->shape, M*K);
        const float* bp = b->data_ptr + compute_offset(bi, b->shape, K*N);
        float* cp = result->data_ptr + bi * M * N;

        // Row-major C = A * B, single BLAS call replaces the old triple-nested loop.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0f, ap, K,
                    bp, N,
                    0.0f, cp, N);
    }
    return result;
}

template <typename F>
static std::shared_ptr<Tensor> run_cpu_broadcast_elementwise(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, F op) {
    std::vector<int> out_shape, pa_shape, pa_strides, pb_shape, pb_strides;
    if (!compute_broadcast_plan_cpu(a->shape, a->strides, b->shape, b->strides,
                                     out_shape, pa_shape, pa_strides, pb_shape, pb_strides))
        throw std::invalid_argument("Shapes are not broadcastable for elementwise op: " +
                                     a->shape_str() + " vs " + b->shape_str());

    auto result = std::make_shared<Tensor>(out_shape, std::string("cpu"));
    int ndim = (int)out_shape.size();
    std::vector<int> idx(ndim, 0);
    for (int flat = 0; flat < result->size; ++flat) {
        int a_off = 0, b_off = 0;
        for (int d = 0; d < ndim; ++d) {
            int ai = (pa_shape[d] == 1) ? 0 : idx[d];
            int bi = (pb_shape[d] == 1) ? 0 : idx[d];
            a_off += ai * pa_strides[d];
            b_off += bi * pb_strides[d];
        }
        result->data_ptr[flat] = op(a->data_ptr[a_off], b->data_ptr[b_off]);
        for (int d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape != b->shape) return run_cpu_broadcast_elementwise(a, b, [](float x, float y){ return x+y; });
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    if (a->is_contiguous() && b->is_contiguous()) {
        for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->data_ptr[i] + b->data_ptr[i];
    } else {
        for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->get_scalar_flat(i) + b->get_scalar_flat(i);
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape != b->shape) return run_cpu_broadcast_elementwise(a, b, [](float x, float y){ return x-y; });
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    if (a->is_contiguous() && b->is_contiguous()) {
        for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->data_ptr[i] - b->data_ptr[i];
    } else {
        for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->get_scalar_flat(i) - b->get_scalar_flat(i);
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape != b->shape) return run_cpu_broadcast_elementwise(a, b, [](float x, float y){ return x*y; });
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    if (a->is_contiguous() && b->is_contiguous()) {
        for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->data_ptr[i] * b->data_ptr[i];
    } else {
        for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->get_scalar_flat(i) * b->get_scalar_flat(i);
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape != b->shape) return run_cpu_broadcast_elementwise(a, b, [](float x, float y){ return x/y; });
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    if (a->is_contiguous() && b->is_contiguous()) {
        for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->data_ptr[i] / b->data_ptr[i];
    } else {
        for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->get_scalar_flat(i) / b->get_scalar_flat(i);
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_add_scalar(std::shared_ptr<Tensor> a, float s) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->get_scalar_flat(i) + s;
    return result;
}

std::shared_ptr<Tensor> run_cpu_sub_scalar(std::shared_ptr<Tensor> a, float s) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->get_scalar_flat(i) - s;
    return result;
}

std::shared_ptr<Tensor> run_cpu_mul_scalar(std::shared_ptr<Tensor> a, float s) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->get_scalar_flat(i) * s;
    return result;
}

std::shared_ptr<Tensor> run_cpu_div_scalar(std::shared_ptr<Tensor> a, float s) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->data_ptr[i] = a->get_scalar_flat(i) / s;
    return result;
}

std::shared_ptr<Tensor> run_cpu_sum_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
    if (!a->is_contiguous())
        throw std::invalid_argument("sum() requires a contiguous tensor. Call .contiguous() first.");

    int ndim = (int)a->shape.size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::out_of_range("sum() dim out of range");

    int outer_size = 1, inner_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= a->shape[i];
    for (int i = dim + 1; i < ndim; ++i) inner_size *= a->shape[i];
    int reduce_size = a->shape[dim];

    std::vector<int> out_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(a->shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    auto result = std::make_shared<Tensor>(out_shape, std::string("cpu"));

    for (int o = 0; o < outer_size; ++o) {
        for (int i = 0; i < inner_size; ++i) {
            float acc = 0.0f;
            for (int r = 0; r < reduce_size; ++r) {
                acc += a->data_ptr[(o * reduce_size + r) * inner_size + i];
            }
            result->data_ptr[o * inner_size + i] = acc;
        }
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_sum_all(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous())
        throw std::invalid_argument("sum() requires a contiguous tensor. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(std::vector<int>{1}, std::string("cpu"));
    float acc = 0.0f;
    for (int i = 0; i < a->size; ++i) acc += a->data_ptr[i];
    result->data_ptr[0] = acc;
    return result;
}

std::shared_ptr<Tensor> run_cpu_broadcast_axis(std::shared_ptr<Tensor> a, int dim, int target_size) {
    if (!a->is_contiguous())
        throw std::invalid_argument("broadcast requires a contiguous tensor.");
    int ndim = (int)a->shape.size();
    if (dim < 0) dim += ndim;

    std::vector<int> out_shape = a->shape;
    out_shape[dim] = target_size;
    auto result = std::make_shared<Tensor>(out_shape, std::string("cpu"));

    int outer_size = 1, inner_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= a->shape[i];
    for (int i = dim + 1; i < ndim; ++i) inner_size *= a->shape[i];

    for (int o = 0; o < outer_size; ++o) {
        for (int t = 0; t < target_size; ++t) {
            for (int i = 0; i < inner_size; ++i) {
                result->data_ptr[(o * target_size + t) * inner_size + i] =
                    a->data_ptr[o * inner_size + i];
            }
        }
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_relu(std::shared_ptr<Tensor> a) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) {
        float v = a->get_scalar_flat(i);
        result->data_ptr[i] = v > 0.0f ? v : 0.0f;
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cpu"));
    for (int i = 0; i < input->size; ++i) {
        result->data_ptr[i] = input->get_scalar_flat(i) > 0.0f ? grad_out->data_ptr[i] : 0.0f;
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_sigmoid(std::shared_ptr<Tensor> a) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) {
        result->data_ptr[i] = 1.0f / (1.0f + std::exp(-a->get_scalar_flat(i)));
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_sigmoid_backward(std::shared_ptr<Tensor> grad_out, const float* sig_out_ptr, int size, std::vector<int> shape) {
    auto result = std::make_shared<Tensor>(shape, std::string("cpu"));
    for (int i = 0; i < size; ++i) {
        float s = sig_out_ptr[i];
        result->data_ptr[i] = grad_out->data_ptr[i] * s * (1.0f - s);
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_tanh(std::shared_ptr<Tensor> a) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) {
        result->data_ptr[i] = std::tanh(a->get_scalar_flat(i));
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_tanh_backward(std::shared_ptr<Tensor> grad_out, const float* tanh_out_ptr, int size, std::vector<int> shape) {
    auto result = std::make_shared<Tensor>(shape, std::string("cpu"));
    for (int i = 0; i < size; ++i) {
        float t = tanh_out_ptr[i];
        result->data_ptr[i] = grad_out->data_ptr[i] * (1.0f - t * t);
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_leaky_relu(std::shared_ptr<Tensor> a, float slope) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) {
        float v = a->get_scalar_flat(i);
        result->data_ptr[i] = v > 0.0f ? v : v * slope;
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_leaky_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, float slope) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cpu"));
    for (int i = 0; i < input->size; ++i) {
        float x = input->get_scalar_flat(i);
        result->data_ptr[i] = x > 0.0f ? grad_out->data_ptr[i] : grad_out->data_ptr[i] * slope;
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_exp(std::shared_ptr<Tensor> a) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->data_ptr[i] = std::exp(a->get_scalar_flat(i));
    return result;
}
std::shared_ptr<Tensor> run_cpu_exp_backward(std::shared_ptr<Tensor> grad_out, const float* exp_out_ptr, int size, std::vector<int> shape) {
    auto result = std::make_shared<Tensor>(shape, std::string("cpu"));
    for (int i = 0; i < size; ++i) {
        result->data_ptr[i] = grad_out->data_ptr[i] * exp_out_ptr[i];
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_log(std::shared_ptr<Tensor> a) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->data_ptr[i] = std::log(a->get_scalar_flat(i));
    return result;
}
std::shared_ptr<Tensor> run_cpu_log_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cpu"));
    for (int i = 0; i < input->size; ++i) result->data_ptr[i] = grad_out->data_ptr[i] / input->get_scalar_flat(i);
    return result;
}
std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cpu_max_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
    if (!a->is_contiguous())
        throw std::invalid_argument("max() requires a contiguous tensor. Call .contiguous() first.");

    int ndim = (int)a->shape.size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::out_of_range("max() dim out of range");

    int outer_size = 1, inner_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= a->shape[i];
    for (int i = dim + 1; i < ndim; ++i) inner_size *= a->shape[i];
    int reduce_size = a->shape[dim];

    std::vector<int> out_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i == dim) { if (keepdim) out_shape.push_back(1); }
        else out_shape.push_back(a->shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    auto result = std::make_shared<Tensor>(out_shape, std::string("cpu"));
    std::vector<int> argmax(outer_size * inner_size);

    for (int o = 0; o < outer_size; ++o) {
        for (int i = 0; i < inner_size; ++i) {
            float best = a->data_ptr[(o * reduce_size + 0) * inner_size + i];
            int best_r = 0;
            for (int r = 1; r < reduce_size; ++r) {
                float v = a->data_ptr[(o * reduce_size + r) * inner_size + i];
                if (v > best) { best = v; best_r = r; }
            }
            result->data_ptr[o * inner_size + i] = best;
            argmax[o * inner_size + i] = best_r;
        }
    }
    return {result, argmax};
}

std::shared_ptr<Tensor> run_cpu_max_axis_backward(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                   std::vector<int> orig_shape, int dim, int reduce_size, int inner_size) {
    auto grad_in = std::make_shared<Tensor>(orig_shape, std::string("cpu"));
    grad_in->fill_zero();

    int out_size = (int)argmax.size();
    int outer_size = out_size / inner_size;
    for (int o = 0; o < outer_size; ++o) {
        for (int i = 0; i < inner_size; ++i) {
            int idx = o * inner_size + i;
            int r = argmax[idx];
            grad_in->data_ptr[(o * reduce_size + r) * inner_size + i] = grad_out->data_ptr[idx];
        }
    }
    return grad_in;
}
