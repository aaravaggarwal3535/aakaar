#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifndef AAKAAR_NO_CUDA
#include <cuda_runtime.h>
#include "allocator.h"
#endif

namespace py = pybind11;
// Add near the top, after namespace py = pybind11;
class Tensor;

struct Node {
    std::function<std::vector<std::shared_ptr<Tensor>>(std::shared_ptr<Tensor>)> backward_fn;
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::string op_name;
    bool freed = false;  // set true only when a backward() call with retain_graph=False
                          // has consumed this node; a later backward() attempt through
                          // this node then raises a clear error instead of silently
                          // treating it as a leaf.
};

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    float* data_ptr;               // points at the first element THIS tensor/view sees
    int size;                      // element count of this view
    std::vector<int> shape;
    std::vector<int> strides;      // in elements
    std::string device;
    // Add inside the Tensor class, alongside your other public fields
    bool requires_grad = false;
    std::shared_ptr<Tensor> grad;       // populated after .backward()
    std::shared_ptr<Node> grad_fn;      // null for leaf tensors

    std::shared_ptr<Tensor> base;  // non-null => this is a view; keeps owner alive
    float* owned_ptr = nullptr;    // the actually-allocated buffer (only if base == nullptr)

    static std::vector<int> contiguous_strides(const std::vector<int>& shp) {
        std::vector<int> st(shp.size());
        int acc = 1;
        for (int i = (int)shp.size() - 1; i >= 0; --i) {
            st[i] = acc;
            acc *= shp[i];
        }
        return st;
    }

    // Fresh allocation (used by rand(), matmul() results, etc.)
    Tensor(std::vector<int> shp, std::string dev) : shape(shp), device(dev), base(nullptr) {
        size = 1;
        for (int d : shape) {
            if (d <= 0) throw std::invalid_argument("Tensor dimensions must be positive");
            size *= d;
        }
        strides = contiguous_strides(shape);
        if (device == "cuda") {
#ifdef AAKAAR_NO_CUDA
            throw std::runtime_error("This build of aakaar was compiled without CUDA support. Use device='cpu' instead.");
#else
            owned_ptr = CachingAllocator::get_instance().allocate(size);
#endif
        } else {
            owned_ptr = new float[size];
        }
        data_ptr = owned_ptr;
    }

    // View constructor: shares memory of `parent` at element offset `off`
    Tensor(std::shared_ptr<Tensor> parent, int off, std::vector<int> shp, std::vector<int> strd)
        : shape(shp), strides(strd), device(parent->device), base(parent) {
        size = 1;
        for (int d : shape) size *= d;
        data_ptr = parent->data_ptr + off;
    }

    ~Tensor() {
        if (base == nullptr) {   // only owners free memory; views just drop their shared_ptr ref
            if (device == "cuda") {
#ifndef AAKAAR_NO_CUDA
                CachingAllocator::get_instance().free(owned_ptr, size);
#endif
            } else {
                delete[] owned_ptr;
            }
        }
    }

    bool is_contiguous() const {
        return strides == contiguous_strides(shape);
    }

    // Read a single scalar value at a multi-dimensional index, respecting strides.
    float get_scalar(const std::vector<int>& idx) {
        int off = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            off += idx[i] * strides[i];
        }
        float value;
#ifndef AAKAAR_NO_CUDA
        if (device == "cuda") {
            cudaMemcpy(&value, data_ptr + off, sizeof(float), cudaMemcpyDeviceToHost);
            return value;
        }
#endif
        return data_ptr[off];
    }

// Add inside the class, near contiguous()
std::shared_ptr<Tensor> detach() {
    auto result = std::make_shared<Tensor>(shared_from_this(), 0, shape, strides);
    result->requires_grad = false;
    // grad_fn is already null on a freshly constructed Tensor
    return result;
}
// Swap two arbitrary axes (like torch's .transpose(dim0, dim1))
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

    // Full axis reversal, matching torch's .T semantics for any ndim
    std::shared_ptr<Tensor> transpose_all() {
        int ndim = (int)shape.size();
        std::vector<int> new_shape(ndim), new_strides(ndim);
        for (int i = 0; i < ndim; ++i) {
            new_shape[i] = shape[ndim - 1 - i];
            new_strides[i] = strides[ndim - 1 - i];
        }
        return std::make_shared<Tensor>(shared_from_this(), 0, new_shape, new_strides);
    }

    // Kept for backward compatibility with existing code that calls transpose2d()
    std::shared_ptr<Tensor> transpose2d() {
        if (shape.size() != 2)
            throw std::invalid_argument("transpose2d() requires a 2D tensor; use transpose(dim0, dim1) for N-D");
        return transpose(0, 1);
    }

void zero_grad() {
    grad = nullptr;
}

// Add inside the Tensor class, near contiguous()

// Strict zero-copy reshape. Fails if the tensor isn't contiguous, matching torch's .view().
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
        throw std::invalid_argument("view() requires a contiguous tensor (non-contiguous strides "
                                     "cannot be reinterpreted as a new shape without copying). "
                                     "Use reshape() instead, which handles this automatically.");

    return std::make_shared<Tensor>(shared_from_this(), 0, new_shape, contiguous_strides(new_shape));
}

// Flexible reshape: zero-copy when possible, otherwise makes a contiguous copy first.
std::shared_ptr<Tensor> reshape(std::vector<int> new_shape) {
    if (is_contiguous()) {
        return view(new_shape);
    }
    return contiguous()->view(new_shape);
}

    std::shared_ptr<Tensor> contiguous() {
        if (is_contiguous()) {
            return shared_from_this();  // already contiguous, return self reference
        }
        auto result = std::make_shared<Tensor>(shape, device);
        std::vector<int> idx(shape.size(), 0);
        for (int flat = 0; flat < size; ++flat) {
            float val = get_scalar(idx);
            if (device == "cuda") {
#ifndef AAKAAR_NO_CUDA
                cudaMemcpy(result->data_ptr + flat, &val, sizeof(float), cudaMemcpyHostToDevice);
#endif
            } else {
                result->data_ptr[flat] = val;
            }
            for (int d = (int)shape.size() - 1; d >= 0; --d) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        return result;
    }

    std::shared_ptr<Tensor> to_device(std::string target_device) {
        if (target_device == device) {
            return shared_from_this();
        }

#ifdef AAKAAR_NO_CUDA
        if (target_device == "cuda") {
            throw std::runtime_error("This build of aakaar has no CUDA support.");
        }
#endif

        auto result = std::make_shared<Tensor>(shape, target_device);

        if (is_contiguous()) {
#ifndef AAKAAR_NO_CUDA
            if (device == "cuda" && target_device == "cuda") {
                cudaMemcpy(result->data_ptr, data_ptr, size * sizeof(float), cudaMemcpyDeviceToDevice);
            } else if (device == "cuda" && target_device == "cpu") {
                cudaMemcpy(result->data_ptr, data_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
            } else if (device == "cpu" && target_device == "cuda") {
                cudaMemcpy(result->data_ptr, data_ptr, size * sizeof(float), cudaMemcpyHostToDevice);
            } else
#endif
            {
                std::memcpy(result->data_ptr, data_ptr, size * sizeof(float));
            }
            return result;
        }

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

    // Access element at flat logical index i, respecting strides (handles views correctly)
    float get_scalar_flat(int flat_idx) {
        std::vector<int> idx(shape.size());
        int remaining = flat_idx;
        for (int d = (int)shape.size() - 1; d >= 0; --d) {
            idx[d] = remaining % shape[d];
            remaining /= shape[d];
        }
        return get_scalar(idx);
    }
// ... existing get_scalar and get_scalar_flat ...

    // Write a single scalar value at a multi-dimensional index, respecting strides.
    void set_scalar(const std::vector<int>& idx, float value) {
        int off = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            off += idx[i] * strides[i];
        }
#ifndef AAKAAR_NO_CUDA
        if (device == "cuda") {
            cudaMemcpy(data_ptr + off, &value, sizeof(float), cudaMemcpyHostToDevice);
            return;
        }
#endif
        data_ptr[off] = value;
    }

    // Zero out this tensor's entire buffer.
    void fill_zero() {
#ifndef AAKAAR_NO_CUDA
        if (device == "cuda") {
            cudaMemset(data_ptr, 0, size * sizeof(float));
            return;
        }
#endif
        std::memset(data_ptr, 0, size * sizeof(float));
    }

    // Unwrap a single-element tensor to a plain float.
    float item() {
        if (size != 1)
            throw std::runtime_error(
                "item() only works on tensors with exactly one element, got size=" +
                std::to_string(size) + ". Use to_numpy() for multi-element tensors.");
        std::vector<int> idx(shape.size(), 0);
        return get_scalar(idx);
    }

    // ... existing to_numpy() method ...
    // Correctly handles non-contiguous strided views on both CPU and CUDA
    py::array_t<float> to_numpy() {
        py::array_t<float> result(shape);
        float* out = result.mutable_data();

        std::vector<float> host_buf;
        const float* src;

        if (device == "cuda") {
#ifndef AAKAAR_NO_CUDA
            int max_off = 0;
            for (size_t i = 0; i < shape.size(); ++i)
                if (shape[i] > 0) max_off += (shape[i] - 1) * strides[i];
            host_buf.resize(max_off + 1);
            cudaMemcpy(host_buf.data(), data_ptr, (max_off + 1) * sizeof(float), cudaMemcpyDeviceToHost);
            src = host_buf.data();
#else
            throw std::runtime_error("CUDA tensor on a CPU-only build");
#endif
        } else {
            src = data_ptr;
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
