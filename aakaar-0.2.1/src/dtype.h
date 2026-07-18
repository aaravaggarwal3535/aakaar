#pragma once
#include <string>
#include <stdexcept>
#include <cstdint>

enum class DType { FLOAT32, FLOAT64, INT32, INT64 };

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::FLOAT32: return 4;
        case DType::FLOAT64: return 8;
        case DType::INT32:   return 4;
        case DType::INT64:   return 8;
    }
    throw std::runtime_error("Unknown dtype");
}

inline std::string dtype_name(DType dt) {
    switch (dt) {
        case DType::FLOAT32: return "float32";
        case DType::FLOAT64: return "float64";
        case DType::INT32:   return "int32";
        case DType::INT64:   return "int64";
    }
    throw std::runtime_error("Unknown dtype");
}

inline DType dtype_from_string(const std::string& s) {
    if (s == "float32" || s == "fp32") return DType::FLOAT32;
    if (s == "float64" || s == "fp64") return DType::FLOAT64;
    if (s == "int32")   return DType::INT32;
    if (s == "int64")   return DType::INT64;
    throw std::invalid_argument("Unknown dtype string: " + s);
}

// Helper used throughout Phase 1: most kernels/methods only support FLOAT32
// so far. Centralizing this check gives one consistent, clear error message
// everywhere else in the codebase, rather than a different ad-hoc message
// per call site.
inline void require_float32(DType dt, const char* op_name) {
    if (dt != DType::FLOAT32) {
        throw std::runtime_error(
            std::string(op_name) + "() does not yet support dtype '" + dtype_name(dt) +
            "'. Only float32 is currently fully implemented for this operation; "
            "float64/int32/int64 support is being added incrementally."
        );
    }
}