#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "dtype.h"

#ifndef AAKAAR_NO_CUDA
#include <cuda_runtime.h>
#include "allocator.h"
#endif

namespace py = pybind11;

class Tensor;

// --- conv1d (im2col/col2im) ---
void run_cpu_im2col_1d(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                        int K, int S, int P, int D, int L_out);
void run_cpu_col2im_1d(std::shared_ptr<Tensor> grad_col, std::shared_ptr<Tensor> grad_x,
                        int B, int C, int L_in, int K, int S, int P, int D, int L_out);
void run_cpu_im2col_1d_typed(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                              int K, int S, int P, int D, int L_out);
#ifndef AAKAAR_NO_CUDA
void run_cuda_im2col_1d(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                         int K, int S, int P, int D, int L_out);
void run_cuda_col2im_1d(std::shared_ptr<Tensor> grad_col, std::shared_ptr<Tensor> grad_x,
                         int B, int C, int L_in, int K, int S, int P, int D, int L_out);
void run_cuda_im2col_1d_typed(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                               int K, int S, int P, int D, int L_out);
#endif

struct Node {
    std::function<std::vector<std::shared_ptr<Tensor>>(std::shared_ptr<Tensor>)> backward_fn;
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::string op_name;
    bool freed = false;
};

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    void* data_ptr;                // raw buffer; interpret according to `dtype`
    DType dtype;
    int size;
    std::vector<int> shape;
    std::vector<int> strides;      // in elements, not bytes
    std::string device;

    bool requires_grad = false;
    std::shared_ptr<Tensor> grad;
    std::shared_ptr<Node> grad_fn;

    std::shared_ptr<Tensor> base;
    void* owned_ptr = nullptr;

    static std::vector<int> contiguous_strides(const std::vector<int>& shp) {
        std::vector<int> st(shp.size());
        int acc = 1;
        for (int i = (int)shp.size() - 1; i >= 0; --i) {
            st[i] = acc;
            acc *= shp[i];
        }
        return st;
    }

    // Fresh allocation. `dt` defaults to FLOAT32 to keep every existing call
    // site (Tensor(shape, device)) working unchanged during this transition.
    Tensor(std::vector<int> shp, std::string dev, DType dt = DType::FLOAT32)
        : shape(shp), device(dev), dtype(dt), base(nullptr) {
        size = 1;
        for (int d : shape) {
            if (d <= 0) throw std::invalid_argument("Tensor dimensions must be positive");
            size *= d;
        }
        strides = contiguous_strides(shape);
        size_t bytes = (size_t)size * dtype_size(dtype);

        if (device == "cuda") {
#ifdef AAKAAR_NO_CUDA
            throw std::runtime_error("This build of aakaar was compiled without CUDA support. Use device='cpu' instead.");
#else
            // NOTE: CachingAllocator currently allocates/tracks in units of
            // float (4 bytes). For FLOAT32 this is exactly right. For other
            // dtypes in Phase 1 we bypass the cache and go straight to
            // cudaMalloc/cudaFree — correct but not yet cache-accelerated.
            // Revisiting the allocator to be byte-based is a natural
            // follow-up once dtype support is otherwise complete.
            if (dtype == DType::FLOAT32) {
                owned_ptr = CachingAllocator::get_instance().allocate(size);
            } else {
                void* ptr;
                cudaError_t err = cudaMalloc(&ptr, bytes);
                if (err != cudaSuccess) throw std::runtime_error("Aakaar out of memory!");
                owned_ptr = ptr;
            }
#endif
        } else {
            owned_ptr = ::operator new(bytes);
        }
        data_ptr = owned_ptr;
    }

    // View constructor: shares memory of `parent` at element offset `off`.
    Tensor(std::shared_ptr<Tensor> parent, int off, std::vector<int> shp, std::vector<int> strd)
        : shape(shp), strides(strd), device(parent->device), dtype(parent->dtype), base(parent) {
        size = 1;
        for (int d : shape) size *= d;
        data_ptr = static_cast<char*>(parent->data_ptr) + (size_t)off * dtype_size(dtype);
    }

    ~Tensor() {
        if (base == nullptr) {
            if (device == "cuda") {
#ifndef AAKAAR_NO_CUDA
                if (dtype == DType::FLOAT32) {
                    CachingAllocator::get_instance().free(static_cast<float*>(owned_ptr), size);
                } else {
                    cudaFree(owned_ptr);
                }
#endif
            } else {
                ::operator delete(owned_ptr);
            }
        }
    }

    bool is_contiguous() const {
        return strides == contiguous_strides(shape);
    }

    // --- FLOAT32-only accessors for now; every other dtype throws clearly. ---

    float* fptr() {
        require_float32(dtype, "this operation");
        return static_cast<float*>(data_ptr);
    }

    float get_scalar(const std::vector<int>& idx) {
        require_float32(dtype, "get_scalar");
        int off = 0;
        for (size_t i = 0; i < idx.size(); ++i) off += idx[i] * strides[i];
        float value;
#ifndef AAKAAR_NO_CUDA
        if (device == "cuda") {
            cudaMemcpy(&value, static_cast<float*>(data_ptr) + off, sizeof(float), cudaMemcpyDeviceToHost);
            return value;
        }
#endif
        return static_cast<float*>(data_ptr)[off];
    }

    void set_scalar(const std::vector<int>& idx, float value) {
        require_float32(dtype, "set_scalar");
        int off = 0;
        for (size_t i = 0; i < idx.size(); ++i) off += idx[i] * strides[i];
#ifndef AAKAAR_NO_CUDA
        if (device == "cuda") {
            cudaMemcpy(static_cast<float*>(data_ptr) + off, &value, sizeof(float), cudaMemcpyHostToDevice);
            return;
        }
#endif
        static_cast<float*>(data_ptr)[off] = value;
    }

    template <typename T>
    T get_scalar_typed(const std::vector<int>& idx) {
        int off = 0;
        for (size_t i = 0; i < idx.size(); ++i) off += idx[i] * strides[i];
        T value;
    #ifndef AAKAAR_NO_CUDA
        if (device == "cuda") {
            cudaMemcpy(&value, static_cast<T*>(data_ptr) + off, sizeof(T), cudaMemcpyDeviceToHost);
            return value;
        }
    #endif
        return static_cast<T*>(data_ptr)[off];
    }

    template <typename T>
    void set_scalar_typed(const std::vector<int>& idx, T value) {
        int off = 0;
        for (size_t i = 0; i < idx.size(); ++i) off += idx[i] * strides[i];
    #ifndef AAKAAR_NO_CUDA
        if (device == "cuda") {
            cudaMemcpy(static_cast<T*>(data_ptr) + off, &value, sizeof(T), cudaMemcpyHostToDevice);
            return;
        }
    #endif
        static_cast<T*>(data_ptr)[off] = value;
    }

    float get_scalar_flat(int flat_idx) {
        require_float32(dtype, "get_scalar_flat");
        std::vector<int> idx(shape.size());
        int remaining = flat_idx;
        for (int d = (int)shape.size() - 1; d >= 0; --d) {
            idx[d] = remaining % shape[d];
            remaining /= shape[d];
        }
        return get_scalar(idx);
    }

    void fill_zero() {
        require_float32(dtype, "fill_zero");
        size_t bytes = (size_t)size * dtype_size(dtype);
#ifndef AAKAAR_NO_CUDA
        if (device == "cuda") {
            cudaMemset(data_ptr, 0, bytes);
            return;
        }
#endif
        std::memset(data_ptr, 0, bytes);
    }
    void fill_zero_typed(DType dt) {
    size_t bytes = (size_t)size * dtype_size(dt);
#ifndef AAKAAR_NO_CUDA
    if (device == "cuda") {
        cudaMemset(data_ptr, 0, bytes);
        return;
    }
#endif
    std::memset(data_ptr, 0, bytes);
}

    float item() {
        require_float32(dtype, "item");
        if (size != 1)
            throw std::runtime_error("item() only works on tensors with exactly one element, got size=" + std::to_string(size));
        std::vector<int> idx(shape.size(), 0);
        return get_scalar(idx);
    }

    std::shared_ptr<Tensor> transpose(int dim0, int dim1) {
        int ndim = (int)shape.size();
        if (dim0 < 0) dim0 += ndim;
        if (dim1 < 0) dim1 += ndim;
        if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim)
            throw std::out_of_range("transpose dim out of range");
        std::vector<int> new_shape = shape;
        std::vector<int> new_strides = strides;
        std::swap(new_shape[dim0], new_shape[dim1]);
        std::swap(new_strides[dim0], new_strides[dim1]);
        return std::make_shared<Tensor>(shared_from_this(), 0, new_shape, new_strides);
    }

    std::shared_ptr<Tensor> transpose_all() {
        int ndim = (int)shape.size();
        std::vector<int> new_shape(ndim), new_strides(ndim);
        for (int i = 0; i < ndim; ++i) {
            new_shape[i] = shape[ndim - 1 - i];
            new_strides[i] = strides[ndim - 1 - i];
        }
        return std::make_shared<Tensor>(shared_from_this(), 0, new_shape, new_strides);
    }

    std::shared_ptr<Tensor> transpose2d() {
        if (shape.size() != 2)
            throw std::invalid_argument("transpose2d() requires a 2D tensor; use transpose(dim0, dim1) for N-D");
        return transpose(0, 1);
    }

    void zero_grad() { grad = nullptr; }

    std::shared_ptr<Tensor> view(std::vector<int> new_shape) {
        int new_size = 1;
        int infer_dim = -1;
        for (size_t i = 0; i < new_shape.size(); ++i) {
            if (new_shape[i] == -1) {
                if (infer_dim != -1) throw std::invalid_argument("Only one dimension can be inferred (-1) in view()");
                infer_dim = (int)i;
            } else {
                if (new_shape[i] <= 0) throw std::invalid_argument("view() dimensions must be positive (or -1 to infer)");
                new_size *= new_shape[i];
            }
        }
        if (infer_dim != -1) {
            if (size % new_size != 0) throw std::invalid_argument("view(): inferred dimension does not evenly divide total size");
            new_shape[infer_dim] = size / new_size;
            new_size = size;
        }
        if (new_size != size)
            throw std::invalid_argument("view(): new shape's total size (" + std::to_string(new_size) +
                                         ") must match tensor's size (" + std::to_string(size) + ")");
        if (!is_contiguous())
            throw std::invalid_argument("view() requires a contiguous tensor. Use reshape() instead.");

        return std::make_shared<Tensor>(shared_from_this(), 0, new_shape, contiguous_strides(new_shape));
    }
    // Builds a fresh tensor and copies `count` floats from `src` into it.
    // `src` is always a host (CPU) pointer; if `dev` is "cuda", the data is
    // copied host->device during construction.
    static std::shared_ptr<Tensor> from_buffer(const float* src, std::vector<int> shp, std::string dev) {
        auto result = std::make_shared<Tensor>(shp, dev, DType::FLOAT32);
    #ifndef AAKAAR_NO_CUDA
        if (dev == "cuda") {
            cudaMemcpy(result->data_ptr, src, (size_t)result->size * sizeof(float), cudaMemcpyHostToDevice);
            return result;
        }
    #endif
        std::memcpy(result->data_ptr, src, (size_t)result->size * sizeof(float));
        return result;
    }
    std::shared_ptr<Tensor> reshape(std::vector<int> new_shape) {
        if (is_contiguous()) return view(new_shape);
        return contiguous()->view(new_shape);
    }

    std::shared_ptr<Tensor> contiguous() {
    if (is_contiguous()) return shared_from_this();

    if (dtype != DType::FLOAT32) {
        // Generic byte-level strided gather: works for any dtype since it
        // doesn't interpret the data, just moves bytes according to strides
        // (computed in elements, scaled by dtype_size here).
        auto result = std::make_shared<Tensor>(shape, device, dtype);
        size_t elem_size = dtype_size(dtype);

        if (device == "cuda") {
#ifndef AAKAAR_NO_CUDA
            // Fall back to a host round-trip for non-float32 CUDA contiguous()
            // for now — a dedicated typed strided-gather CUDA kernel (like the
            // float32 one) is a further optimization, not needed for correctness.
            std::vector<char> host_buf((size_t)size * elem_size);
            std::vector<int> idx(shape.size(), 0);
            for (int flat = 0; flat < size; ++flat) {
                int off = 0;
                for (size_t d = 0; d < shape.size(); ++d) off += idx[d] * strides[d];
                cudaMemcpy(host_buf.data() + (size_t)flat * elem_size,
                           static_cast<char*>(data_ptr) + (size_t)off * elem_size,
                           elem_size, cudaMemcpyDeviceToHost);
                for (int d = (int)shape.size() - 1; d >= 0; --d) {
                    if (++idx[d] < shape[d]) break;
                    idx[d] = 0;
                }
            }
            cudaMemcpy(result->data_ptr, host_buf.data(), (size_t)size * elem_size, cudaMemcpyHostToDevice);
            return result;
#endif
        }

        // CPU path: direct byte copy per element according to strides.
        std::vector<int> idx(shape.size(), 0);
        for (int flat = 0; flat < size; ++flat) {
            int off = 0;
            for (size_t d = 0; d < shape.size(); ++d) off += idx[d] * strides[d];
            std::memcpy(static_cast<char*>(result->data_ptr) + (size_t)flat * elem_size,
                        static_cast<char*>(data_ptr) + (size_t)off * elem_size,
                        elem_size);
            for (int d = (int)shape.size() - 1; d >= 0; --d) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        return result;
    }

    // Existing float32 fast path (GPU strided-gather kernel), unchanged.
    auto result = std::make_shared<Tensor>(shape, device, dtype);
#ifndef AAKAAR_NO_CUDA
    if (device == "cuda") {
        extern void run_cuda_strided_gather(const float*, float*, const std::vector<int>&,
                                             const std::vector<int>&, int);
        run_cuda_strided_gather(static_cast<float*>(data_ptr), static_cast<float*>(result->data_ptr), shape, strides, size);
        return result;
    }
#endif
    std::vector<int> idx(shape.size(), 0);
    for (int flat = 0; flat < size; ++flat) {
        static_cast<float*>(result->data_ptr)[flat] = get_scalar(idx);
        for (int d = (int)shape.size() - 1; d >= 0; --d) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
    return result;
}

    void copy_(std::shared_ptr<Tensor> other) {
    if (shape != other->shape)
        throw std::invalid_argument("copy_(): shape mismatch");
    if (dtype != other->dtype)
        throw std::invalid_argument("copy_(): dtype mismatch (" + dtype_name(dtype) + " vs " + dtype_name(other->dtype) + ")");

    size_t elem_size = dtype_size(dtype);

    if (is_contiguous() && other->is_contiguous() && device == other->device) {
        size_t bytes = (size_t)size * elem_size;
#ifndef AAKAAR_NO_CUDA
        if (device == "cuda") {
            cudaMemcpy(data_ptr, other->data_ptr, bytes, cudaMemcpyDeviceToDevice);
            return;
        }
#endif
        std::memcpy(data_ptr, other->data_ptr, bytes);
        return;
    }

    if (dtype != DType::FLOAT32)
        throw std::runtime_error("copy_() between non-contiguous tensors is not yet supported for dtype '" +
                                  dtype_name(dtype) + "'. Call .contiguous() on both tensors first.");

    std::vector<int> idx(shape.size(), 0);
    for (int flat = 0; flat < size; ++flat) {
        set_scalar(idx, other->get_scalar(idx));
        for (int d = (int)shape.size() - 1; d >= 0; --d) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
}

    std::shared_ptr<Tensor> to_device(std::string target_device) {
        if (target_device == device) return shared_from_this();
    #ifdef AAKAAR_NO_CUDA
        if (target_device == "cuda") throw std::runtime_error("This build of aakaar has no CUDA support.");
    #endif
        auto result = std::make_shared<Tensor>(shape, target_device, dtype);
        size_t elem_size = dtype_size(dtype);

        if (is_contiguous()) {
            size_t bytes = (size_t)size * elem_size;
    #ifndef AAKAAR_NO_CUDA
            if (device == "cuda" && target_device == "cuda") {
                cudaMemcpy(result->data_ptr, data_ptr, bytes, cudaMemcpyDeviceToDevice);
            } else if (device == "cuda" && target_device == "cpu") {
                cudaMemcpy(result->data_ptr, data_ptr, bytes, cudaMemcpyDeviceToHost);
            } else if (device == "cpu" && target_device == "cuda") {
                cudaMemcpy(result->data_ptr, data_ptr, bytes, cudaMemcpyHostToDevice);
            } else
    #endif
            {
                std::memcpy(result->data_ptr, data_ptr, bytes);
            }
            return result;
        }

        // Non-contiguous fallback: only float32 has a working strided element-by-element
        // path today (get_scalar/set_scalar are float32-only per Step 1's design).
        // Other dtypes must be made contiguous first before crossing devices.
        require_float32(dtype, "to_device (on a non-contiguous tensor)");

        std::vector<float> host_buf(size);
        std::vector<int> idx(shape.size(), 0);
        for (int flat = 0; flat < size; ++flat) {
            host_buf[flat] = get_scalar(idx);
            for (int d = (int)shape.size() - 1; d >= 0; --d) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
    #ifndef AAKAAR_NO_CUDA
        if (target_device == "cuda") {
            cudaMemcpy(result->data_ptr, host_buf.data(), size * sizeof(float), cudaMemcpyHostToDevice);
            return result;
        }
    #endif
        std::memcpy(result->data_ptr, host_buf.data(), size * sizeof(float));
        return result;
    }

    std::shared_ptr<Tensor> detach() {
        auto result = std::make_shared<Tensor>(shared_from_this(), 0, shape, strides);
        result->requires_grad = false;
        return result;
    }

    template <typename T>
    py::array_t<T> to_numpy_typed() {
        py::array_t<T> result(shape);
        T* out = result.mutable_data();

        std::vector<T> host_buf;
        const T* src;

        if (device == "cuda") {
    #ifndef AAKAAR_NO_CUDA
            int max_off = 0;
            for (size_t i = 0; i < shape.size(); ++i)
                if (shape[i] > 0) max_off += (shape[i] - 1) * strides[i];
            host_buf.resize(max_off + 1);
            cudaMemcpy(host_buf.data(), data_ptr, (size_t)(max_off + 1) * sizeof(T), cudaMemcpyDeviceToHost);
            src = host_buf.data();
    #else
            throw std::runtime_error("CUDA tensor on a CPU-only build");
    #endif
        } else {
            src = static_cast<T*>(data_ptr);
        }

        std::vector<int> idx(shape.size(), 0);
        for (int flat = 0; flat < size; ++flat) {
            int off = 0;
            for (size_t d = 0; d < shape.size(); ++d) off += idx[d] * strides[d];
            out[flat] = src[off];
            for (int d = (int)shape.size() - 1; d >= 0; --d) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        return result;
    }

    py::object to_numpy() {
        switch (dtype) {
            case DType::FLOAT32: return to_numpy_typed<float>();
            case DType::FLOAT64: return to_numpy_typed<double>();
            case DType::INT32:   return to_numpy_typed<int32_t>();
            case DType::INT64:   return to_numpy_typed<int64_t>();
        }
        throw std::runtime_error("Unknown dtype");
    }

    std::string shape_str() const {
        std::string s = "(";
        for (size_t i = 0; i < shape.size(); ++i) {
            s += std::to_string(shape[i]);
            if (i < shape.size() - 1) s += ", ";
        }
        s += shape.size() == 1 ? ",)" : ")";
        return s;
    }

    std::string repr() {
        if (dtype != DType::FLOAT32) {
            return "aakaar.Tensor(dtype='" + dtype_name(dtype) + "', device='" + device +
                   "', shape=" + shape_str() + ") [preview unavailable for this dtype yet]";
        }
        std::string out = "aakaar.Tensor([";
        int preview = size < 6 ? size : 6;
        std::vector<int> idx(shape.size(), 0);
        for (int i = 0; i < preview; ++i) {
            out += std::to_string(get_scalar(idx));
            if (i < preview - 1) out += ", ";
            for (int d = (int)shape.size() - 1; d >= 0; --d) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        if (size > preview) out += ", ...";
        out += "], device='" + device + "', shape=" + shape_str() + ")";
        return out;
    }
};