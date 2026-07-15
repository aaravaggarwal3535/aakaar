#include <cmath>
#include <random>
#include <memory>
#include <stdexcept>
#include "tensor.h"
#include <omp.h>
#include <cblas.h>

#ifndef _WIN32
#include <dlfcn.h>
#else
extern "C" char* openblas_get_config(void);
extern "C" char* openblas_get_corename(void);
#endif

std::string get_openblas_diagnostic() {
#ifdef _WIN32
    return std::string("Config: ") + openblas_get_config() + " | Core: " + openblas_get_corename();
#else
    void* self = dlopen(nullptr, RTLD_NOW);
    std::string result = "Config: ";
    if (self) {
        typedef char* (*config_fn)(void);
        auto get_config = (config_fn)dlsym(self, "openblas_get_config");
        auto get_corename = (config_fn)dlsym(self, "openblas_get_corename");
        result += get_config ? get_config() : "unavailable";
        result += " | Core: ";
        result += get_corename ? get_corename() : "unavailable";
        dlclose(self);
    } else {
        result += "unavailable | Core: unavailable";
    }
    return result;
#endif
}

void fill_cpu_random(std::shared_ptr<Tensor> t, unsigned long long seed) {
    std::mt19937 gen(seed);
    switch (t->dtype) {
        case DType::FLOAT32: {
            std::uniform_real_distribution<float> dis(0.0f, 1.0f);
            float* ptr = static_cast<float*>(t->data_ptr);
            for (int i = 0; i < t->size; ++i) ptr[i] = dis(gen);
            break;
        }
        case DType::FLOAT64: {
            std::uniform_real_distribution<double> dis(0.0, 1.0);
            double* ptr = static_cast<double*>(t->data_ptr);
            for (int i = 0; i < t->size; ++i) ptr[i] = dis(gen);
            break;
        }
        default:
            throw std::runtime_error("rand() only supports float32/float64 dtypes. Use randint() for integer types.");
    }
}

void fill_cpu_randint(std::shared_ptr<Tensor> t, long long low, long long high, unsigned long long seed) {
    std::mt19937 gen(seed);
    switch (t->dtype) {
        case DType::INT32: {
            std::uniform_int_distribution<int32_t> dis((int32_t)low, (int32_t)(high - 1));
            int32_t* ptr = static_cast<int32_t*>(t->data_ptr);
            for (int i = 0; i < t->size; ++i) ptr[i] = dis(gen);
            break;
        }
        case DType::INT64: {
            std::uniform_int_distribution<int64_t> dis(low, high - 1);
            int64_t* ptr = static_cast<int64_t*>(t->data_ptr);
            for (int i = 0; i < t->size; ++i) ptr[i] = dis(gen);
            break;
        }
        default:
            throw std::runtime_error("randint() only supports int32/int64 dtypes.");
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
        const float* ap = a->fptr() + compute_offset(bi, a->shape, M*K);
        const float* bp = b->fptr() + compute_offset(bi, b->shape, K*N);
        float* cp = result->fptr() + bi * M * N;

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
        result->fptr()[flat] = op(a->fptr()[a_off], b->fptr()[b_off]);
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
        for (int i = 0; i < a->size; ++i) result->fptr()[i] = a->fptr()[i] + b->fptr()[i];
    } else {
        for (int i = 0; i < a->size; ++i) result->fptr()[i] = a->get_scalar_flat(i) + b->get_scalar_flat(i);
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape != b->shape) return run_cpu_broadcast_elementwise(a, b, [](float x, float y){ return x-y; });
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    if (a->is_contiguous() && b->is_contiguous()) {
        for (int i = 0; i < a->size; ++i) result->fptr()[i] = a->fptr()[i] - b->fptr()[i];
    } else {
        for (int i = 0; i < a->size; ++i) result->fptr()[i] = a->get_scalar_flat(i) - b->get_scalar_flat(i);
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape != b->shape) return run_cpu_broadcast_elementwise(a, b, [](float x, float y){ return x*y; });
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    if (a->is_contiguous() && b->is_contiguous()) {
        for (int i = 0; i < a->size; ++i) result->fptr()[i] = a->fptr()[i] * b->fptr()[i];
    } else {
        for (int i = 0; i < a->size; ++i) result->fptr()[i] = a->get_scalar_flat(i) * b->get_scalar_flat(i);
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape != b->shape) return run_cpu_broadcast_elementwise(a, b, [](float x, float y){ return x/y; });
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    if (a->is_contiguous() && b->is_contiguous()) {
        for (int i = 0; i < a->size; ++i) result->fptr()[i] = a->fptr()[i] / b->fptr()[i];
    } else {
        for (int i = 0; i < a->size; ++i) result->fptr()[i] = a->get_scalar_flat(i) / b->get_scalar_flat(i);
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_add_scalar(std::shared_ptr<Tensor> a, float s) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->fptr()[i] = a->get_scalar_flat(i) + s;
    return result;
}

std::shared_ptr<Tensor> run_cpu_sub_scalar(std::shared_ptr<Tensor> a, float s) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->fptr()[i] = a->get_scalar_flat(i) - s;
    return result;
}

std::shared_ptr<Tensor> run_cpu_mul_scalar(std::shared_ptr<Tensor> a, float s) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->fptr()[i] = a->get_scalar_flat(i) * s;
    return result;
}

std::shared_ptr<Tensor> run_cpu_div_scalar(std::shared_ptr<Tensor> a, float s) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->fptr()[i] = a->get_scalar_flat(i) / s;
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
                acc += a->fptr()[(o * reduce_size + r) * inner_size + i];
            }
            result->fptr()[o * inner_size + i] = acc;
        }
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_sum_all(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous())
        throw std::invalid_argument("sum() requires a contiguous tensor. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(std::vector<int>{1}, std::string("cpu"));
    float acc = 0.0f;
    for (int i = 0; i < a->size; ++i) acc += a->fptr()[i];
    result->fptr()[0] = acc;
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
                result->fptr()[(o * target_size + t) * inner_size + i] =
                    a->fptr()[o * inner_size + i];
            }
        }
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_relu(std::shared_ptr<Tensor> a) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) {
        float v = a->get_scalar_flat(i);
        result->fptr()[i] = v > 0.0f ? v : 0.0f;
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cpu"));
    for (int i = 0; i < input->size; ++i) {
        result->fptr()[i] = input->get_scalar_flat(i) > 0.0f ? grad_out->fptr()[i] : 0.0f;
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_sigmoid(std::shared_ptr<Tensor> a) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) {
        result->fptr()[i] = 1.0f / (1.0f + std::exp(-a->get_scalar_flat(i)));
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_sigmoid_backward(std::shared_ptr<Tensor> grad_out, const float* sig_out_ptr, int size, std::vector<int> shape) {
    auto result = std::make_shared<Tensor>(shape, std::string("cpu"));
    for (int i = 0; i < size; ++i) {
        float s = sig_out_ptr[i];
        result->fptr()[i] = grad_out->fptr()[i] * s * (1.0f - s);
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_tanh(std::shared_ptr<Tensor> a) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) {
        result->fptr()[i] = std::tanh(a->get_scalar_flat(i));
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_tanh_backward(std::shared_ptr<Tensor> grad_out, const float* tanh_out_ptr, int size, std::vector<int> shape) {
    auto result = std::make_shared<Tensor>(shape, std::string("cpu"));
    for (int i = 0; i < size; ++i) {
        float t = tanh_out_ptr[i];
        result->fptr()[i] = grad_out->fptr()[i] * (1.0f - t * t);
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_leaky_relu(std::shared_ptr<Tensor> a, float slope) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) {
        float v = a->get_scalar_flat(i);
        result->fptr()[i] = v > 0.0f ? v : v * slope;
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_leaky_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, float slope) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cpu"));
    for (int i = 0; i < input->size; ++i) {
        float x = input->get_scalar_flat(i);
        result->fptr()[i] = x > 0.0f ? grad_out->fptr()[i] : grad_out->fptr()[i] * slope;
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_exp(std::shared_ptr<Tensor> a) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->fptr()[i] = std::exp(a->get_scalar_flat(i));
    return result;
}
std::shared_ptr<Tensor> run_cpu_exp_backward(std::shared_ptr<Tensor> grad_out, const float* exp_out_ptr, int size, std::vector<int> shape) {
    auto result = std::make_shared<Tensor>(shape, std::string("cpu"));
    for (int i = 0; i < size; ++i) {
        result->fptr()[i] = grad_out->fptr()[i] * exp_out_ptr[i];
    }
    return result;
}
std::shared_ptr<Tensor> run_cpu_log(std::shared_ptr<Tensor> a) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"));
    for (int i = 0; i < a->size; ++i) result->fptr()[i] = std::log(a->get_scalar_flat(i));
    return result;
}
std::shared_ptr<Tensor> run_cpu_log_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cpu"));
    for (int i = 0; i < input->size; ++i) result->fptr()[i] = grad_out->fptr()[i] / input->get_scalar_flat(i);
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
            float best = a->fptr()[(o * reduce_size + 0) * inner_size + i];
            int best_r = 0;
            for (int r = 1; r < reduce_size; ++r) {
                float v = a->fptr()[(o * reduce_size + r) * inner_size + i];
                if (v > best) { best = v; best_r = r; }
            }
            result->fptr()[o * inner_size + i] = best;
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
            grad_in->fptr()[(o * reduce_size + r) * inner_size + i] = grad_out->fptr()[idx];
        }
    }
    return grad_in;
}

template <typename T, typename F>
static std::shared_ptr<Tensor> run_cpu_elementwise_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, F op) {
    if (a->shape != b->shape)
        throw std::invalid_argument("Broadcasting is not yet supported for non-float32 dtypes.");
    auto result = std::make_shared<Tensor>(a->shape, a->device, a->dtype);
    T* ap = static_cast<T*>(a->data_ptr);
    T* bp = static_cast<T*>(b->data_ptr);
    T* rp = static_cast<T*>(result->data_ptr);
    for (int i = 0; i < a->size; ++i) rp[i] = op(ap[i], bp[i]);
    return result;
}

std::shared_ptr<Tensor> run_cpu_add_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cpu_elementwise_typed<double>(a, b, [](double x, double y){ return x+y; });
        case DType::INT32:   return run_cpu_elementwise_typed<int32_t>(a, b, [](int32_t x, int32_t y){ return x+y; });
        case DType::INT64:   return run_cpu_elementwise_typed<int64_t>(a, b, [](int64_t x, int64_t y){ return x+y; });
        default: throw std::runtime_error("add(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}
std::shared_ptr<Tensor> run_cpu_sub_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cpu_elementwise_typed<double>(a, b, [](double x, double y){ return x-y; });
        case DType::INT32:   return run_cpu_elementwise_typed<int32_t>(a, b, [](int32_t x, int32_t y){ return x-y; });
        case DType::INT64:   return run_cpu_elementwise_typed<int64_t>(a, b, [](int64_t x, int64_t y){ return x-y; });
        default: throw std::runtime_error("sub(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}
std::shared_ptr<Tensor> run_cpu_mul_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cpu_elementwise_typed<double>(a, b, [](double x, double y){ return x*y; });
        case DType::INT32:   return run_cpu_elementwise_typed<int32_t>(a, b, [](int32_t x, int32_t y){ return x*y; });
        case DType::INT64:   return run_cpu_elementwise_typed<int64_t>(a, b, [](int64_t x, int64_t y){ return x*y; });
        default: throw std::runtime_error("mul(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}
std::shared_ptr<Tensor> run_cpu_div_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cpu_elementwise_typed<double>(a, b, [](double x, double y){ return x/y; });
        case DType::INT32:   return run_cpu_elementwise_typed<int32_t>(a, b, [](int32_t x, int32_t y){ return x/y; });
        case DType::INT64:   return run_cpu_elementwise_typed<int64_t>(a, b, [](int64_t x, int64_t y){ return x/y; });
        default: throw std::runtime_error("div(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}

// ---- Typed sum_axis / sum_all ----
// torch-style promotion: int32/int64 always accumulate into int64 to reduce
// overflow risk (matches torch.sum's default integer promotion). float32/
// float64 keep their own width — no promotion for floats.

static DType sum_result_dtype(DType in) {
    switch (in) {
        case DType::INT32:
        case DType::INT64:
            return DType::INT64;
        case DType::FLOAT32:
            return DType::FLOAT32;
        case DType::FLOAT64:
            return DType::FLOAT64;
    }
    throw std::runtime_error("sum(): unknown dtype");
}

template <typename InT, typename AccT>
static std::shared_ptr<Tensor> run_cpu_sum_axis_typed_impl(std::shared_ptr<Tensor> a, int dim, bool keepdim, DType out_dtype) {
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

    auto result = std::make_shared<Tensor>(out_shape, std::string("cpu"), out_dtype);
    const InT* ap = static_cast<const InT*>(a->data_ptr);
    AccT* rp = static_cast<AccT*>(result->data_ptr);

    for (int o = 0; o < outer_size; ++o) {
        for (int i = 0; i < inner_size; ++i) {
            AccT acc = AccT(0);
            for (int r = 0; r < reduce_size; ++r) {
                acc += (AccT)ap[(o * reduce_size + r) * inner_size + i];
            }
            rp[o * inner_size + i] = acc;
        }
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_sum_axis_typed(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
    DType out_dtype = sum_result_dtype(a->dtype);
    switch (a->dtype) {
        case DType::FLOAT64: return run_cpu_sum_axis_typed_impl<double, double>(a, dim, keepdim, out_dtype);
        case DType::INT32:   return run_cpu_sum_axis_typed_impl<int32_t, int64_t>(a, dim, keepdim, out_dtype);
        case DType::INT64:   return run_cpu_sum_axis_typed_impl<int64_t, int64_t>(a, dim, keepdim, out_dtype);
        default: throw std::runtime_error("sum(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}

template <typename InT, typename AccT>
static std::shared_ptr<Tensor> run_cpu_sum_all_typed_impl(std::shared_ptr<Tensor> a, DType out_dtype) {
    if (!a->is_contiguous())
        throw std::invalid_argument("sum() requires a contiguous tensor. Call .contiguous() first.");
    auto result = std::make_shared<Tensor>(std::vector<int>{1}, std::string("cpu"), out_dtype);
    const InT* ap = static_cast<const InT*>(a->data_ptr);
    AccT acc = AccT(0);
    for (int i = 0; i < a->size; ++i) acc += (AccT)ap[i];
    static_cast<AccT*>(result->data_ptr)[0] = acc;
    return result;
}

std::shared_ptr<Tensor> run_cpu_sum_all_typed(std::shared_ptr<Tensor> a) {
    DType out_dtype = sum_result_dtype(a->dtype);
    switch (a->dtype) {
        case DType::FLOAT64: return run_cpu_sum_all_typed_impl<double, double>(a, out_dtype);
        case DType::INT32:   return run_cpu_sum_all_typed_impl<int32_t, int64_t>(a, out_dtype);
        case DType::INT64:   return run_cpu_sum_all_typed_impl<int64_t, int64_t>(a, out_dtype);
        default: throw std::runtime_error("sum(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}

template <typename T>
static std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cpu_max_axis_typed_impl(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
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

    auto result = std::make_shared<Tensor>(out_shape, std::string("cpu"), a->dtype);
    const T* ap = static_cast<const T*>(a->data_ptr);
    T* rp = static_cast<T*>(result->data_ptr);
    std::vector<int> argmax(outer_size * inner_size);

    for (int o = 0; o < outer_size; ++o) {
        for (int i = 0; i < inner_size; ++i) {
            T best = ap[(o * reduce_size + 0) * inner_size + i];
            int best_r = 0;
            for (int r = 1; r < reduce_size; ++r) {
                T v = ap[(o * reduce_size + r) * inner_size + i];
                if (v > best) { best = v; best_r = r; }
            }
            rp[o * inner_size + i] = best;
            argmax[o * inner_size + i] = best_r;
        }
    }
    return {result, argmax};
}

std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cpu_max_axis_typed(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cpu_max_axis_typed_impl<double>(a, dim, keepdim);
        case DType::INT32:   return run_cpu_max_axis_typed_impl<int32_t>(a, dim, keepdim);
        case DType::INT64:   return run_cpu_max_axis_typed_impl<int64_t>(a, dim, keepdim);
        default: throw std::runtime_error("max(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}

template <typename T>
static std::shared_ptr<Tensor> run_cpu_max_axis_backward_typed_impl(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                                     std::vector<int> orig_shape, int dim, int reduce_size, int inner_size, DType dt) {
    auto grad_in = std::make_shared<Tensor>(orig_shape, std::string("cpu"), dt);
    grad_in->fill_zero_typed(dt);

    const T* gop = static_cast<const T*>(grad_out->data_ptr);
    T* gip = static_cast<T*>(grad_in->data_ptr);

    int out_size = (int)argmax.size();
    int outer_size = out_size / inner_size;
    for (int o = 0; o < outer_size; ++o) {
        for (int i = 0; i < inner_size; ++i) {
            int idx = o * inner_size + i;
            int r = argmax[idx];
            gip[(o * reduce_size + r) * inner_size + i] = gop[idx];
        }
    }
    return grad_in;
}

std::shared_ptr<Tensor> run_cpu_max_axis_backward_typed(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                         std::vector<int> orig_shape, int dim, int reduce_size, int inner_size) {
    switch (grad_out->dtype) {
        case DType::FLOAT64: return run_cpu_max_axis_backward_typed_impl<double>(grad_out, argmax, orig_shape, dim, reduce_size, inner_size, DType::FLOAT64);
        case DType::INT32:   return run_cpu_max_axis_backward_typed_impl<int32_t>(grad_out, argmax, orig_shape, dim, reduce_size, inner_size, DType::INT32);
        case DType::INT64:   return run_cpu_max_axis_backward_typed_impl<int64_t>(grad_out, argmax, orig_shape, dim, reduce_size, inner_size, DType::INT64);
        default: throw std::runtime_error("max() backward: unsupported dtype '" + dtype_name(grad_out->dtype) + "'");
    }
}

// ---- Typed relu / relu_backward ----
// Well-defined on integers (pure comparison + select), so int32/int64 are
// supported here, unlike sigmoid/tanh/exp/log which require float math.

template <typename T>
static std::shared_ptr<Tensor> run_cpu_relu_typed_impl(std::shared_ptr<Tensor> a) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"), a->dtype);
    const T* ap = static_cast<const T*>(a->data_ptr);
    T* rp = static_cast<T*>(result->data_ptr);
    for (int i = 0; i < a->size; ++i) {
        T v = ap[i];
        rp[i] = v > T(0) ? v : T(0);
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_relu_typed(std::shared_ptr<Tensor> a) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cpu_relu_typed_impl<double>(a);
        case DType::INT32:   return run_cpu_relu_typed_impl<int32_t>(a);
        case DType::INT64:   return run_cpu_relu_typed_impl<int64_t>(a);
        default: throw std::runtime_error("relu(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}

template <typename T>
static std::shared_ptr<Tensor> run_cpu_relu_backward_typed_impl(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cpu"), input->dtype);
    const T* gop = static_cast<const T*>(grad_out->data_ptr);
    const T* ip = static_cast<const T*>(input->data_ptr);
    T* rp = static_cast<T*>(result->data_ptr);
    for (int i = 0; i < input->size; ++i) {
        rp[i] = ip[i] > T(0) ? gop[i] : T(0);
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_relu_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input) {
    switch (input->dtype) {
        case DType::FLOAT64: return run_cpu_relu_backward_typed_impl<double>(grad_out, input);
        case DType::INT32:   return run_cpu_relu_backward_typed_impl<int32_t>(grad_out, input);
        case DType::INT64:   return run_cpu_relu_backward_typed_impl<int64_t>(grad_out, input);
        default: throw std::runtime_error("relu() backward: unsupported dtype '" + dtype_name(input->dtype) + "'");
    }
}

template <typename T, typename F>
static std::shared_ptr<Tensor> run_cpu_scalar_typed_impl(std::shared_ptr<Tensor> a, double s, F op) {
    auto result = std::make_shared<Tensor>(a->shape, a->device, a->dtype);
    const T* ap = static_cast<const T*>(a->data_ptr);
    T* rp = static_cast<T*>(result->data_ptr);
    T sv = static_cast<T>(s);
    for (int i = 0; i < a->size; ++i) rp[i] = op(ap[i], sv);
    return result;
}

std::shared_ptr<Tensor> run_cpu_add_scalar_typed(std::shared_ptr<Tensor> a, double s) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cpu_scalar_typed_impl<double>(a, s, [](double x, double y){ return x+y; });
        case DType::INT32:   return run_cpu_scalar_typed_impl<int32_t>(a, s, [](int32_t x, int32_t y){ return x+y; });
        case DType::INT64:   return run_cpu_scalar_typed_impl<int64_t>(a, s, [](int64_t x, int64_t y){ return x+y; });
        default: throw std::runtime_error("add(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}
std::shared_ptr<Tensor> run_cpu_sub_scalar_typed(std::shared_ptr<Tensor> a, double s) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cpu_scalar_typed_impl<double>(a, s, [](double x, double y){ return x-y; });
        case DType::INT32:   return run_cpu_scalar_typed_impl<int32_t>(a, s, [](int32_t x, int32_t y){ return x-y; });
        case DType::INT64:   return run_cpu_scalar_typed_impl<int64_t>(a, s, [](int64_t x, int64_t y){ return x-y; });
        default: throw std::runtime_error("sub(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}
std::shared_ptr<Tensor> run_cpu_mul_scalar_typed(std::shared_ptr<Tensor> a, double s) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cpu_scalar_typed_impl<double>(a, s, [](double x, double y){ return x*y; });
        case DType::INT32:   return run_cpu_scalar_typed_impl<int32_t>(a, s, [](int32_t x, int32_t y){ return x*y; });
        case DType::INT64:   return run_cpu_scalar_typed_impl<int64_t>(a, s, [](int64_t x, int64_t y){ return x*y; });
        default: throw std::runtime_error("mul(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}
std::shared_ptr<Tensor> run_cpu_div_scalar_typed(std::shared_ptr<Tensor> a, double s) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cpu_scalar_typed_impl<double>(a, s, [](double x, double y){ return x/y; });
        case DType::INT32:   return run_cpu_scalar_typed_impl<int32_t>(a, s, [](int32_t x, int32_t y){ return x/y; });
        case DType::INT64:   return run_cpu_scalar_typed_impl<int64_t>(a, s, [](int64_t x, int64_t y){ return x/y; });
        default: throw std::runtime_error("div(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}

// ---- Typed scalar ops: add/sub/mul/div by a scalar ----
template <typename T, typename F>
static std::shared_ptr<Tensor> run_cpu_scalar_op_typed(std::shared_ptr<Tensor> a, double s, F op) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"), a->dtype);
    T sv = (T)s;
    if (a->is_contiguous()) {
        const T* ap = static_cast<const T*>(a->data_ptr);
        T* rp = static_cast<T*>(result->data_ptr);
        for (int i = 0; i < a->size; ++i) rp[i] = op(ap[i], sv);
    } else {
        // Non-contiguous: gather via manual stride offsets (get_scalar_flat
        // is float32-only, so this is done here directly instead).
        T* rp = static_cast<T*>(result->data_ptr);
        std::vector<int> idx(a->shape.size(), 0);
        const T* ap = static_cast<const T*>(a->data_ptr);
        for (int flat = 0; flat < a->size; ++flat) {
            int off = 0;
            for (size_t d = 0; d < a->shape.size(); ++d) off += idx[d] * a->strides[d];
            rp[flat] = op(ap[off], sv);
            for (int d = (int)a->shape.size() - 1; d >= 0; --d) {
                if (++idx[d] < a->shape[d]) break;
                idx[d] = 0;
            }
        }
    }
    return result;
}

template <typename T>
static std::shared_ptr<Tensor> run_cpu_leaky_relu_typed_impl(std::shared_ptr<Tensor> a, double slope) {
    auto result = std::make_shared<Tensor>(a->shape, std::string("cpu"), a->dtype);
    const T* ap = static_cast<const T*>(a->data_ptr);
    T* rp = static_cast<T*>(result->data_ptr);
    for (int i = 0; i < a->size; ++i) {
        T v = ap[i];
        rp[i] = v > T(0) ? v : (T)((double)v * slope);
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_leaky_relu_typed(std::shared_ptr<Tensor> a, double slope) {
    switch (a->dtype) {
        case DType::FLOAT64: return run_cpu_leaky_relu_typed_impl<double>(a, slope);
        case DType::INT32:   return run_cpu_leaky_relu_typed_impl<int32_t>(a, slope);
        case DType::INT64:   return run_cpu_leaky_relu_typed_impl<int64_t>(a, slope);
        default: throw std::runtime_error("leaky_relu(): unsupported dtype '" + dtype_name(a->dtype) + "'");
    }
}

template <typename T>
static std::shared_ptr<Tensor> run_cpu_leaky_relu_backward_typed_impl(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, double slope) {
    auto result = std::make_shared<Tensor>(input->shape, std::string("cpu"), input->dtype);
    const T* gop = static_cast<const T*>(grad_out->data_ptr);
    const T* ip = static_cast<const T*>(input->data_ptr);
    T* rp = static_cast<T*>(result->data_ptr);
    for (int i = 0; i < input->size; ++i) {
        T x = ip[i];
        T g = gop[i];
        rp[i] = x > T(0) ? g : (T)((double)g * slope);
    }
    return result;
}

std::shared_ptr<Tensor> run_cpu_leaky_relu_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, double slope) {
    switch (input->dtype) {
        case DType::FLOAT64: return run_cpu_leaky_relu_backward_typed_impl<double>(grad_out, input, slope);
        case DType::INT32:   return run_cpu_leaky_relu_backward_typed_impl<int32_t>(grad_out, input, slope);
        case DType::INT64:   return run_cpu_leaky_relu_backward_typed_impl<int64_t>(grad_out, input, slope);
        default: throw std::runtime_error("leaky_relu() backward: unsupported dtype '" + dtype_name(input->dtype) + "'");
    }
}