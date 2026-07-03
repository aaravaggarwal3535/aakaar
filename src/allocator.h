#pragma once
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <iostream>

class CachingAllocator {
private:
    // Maps the size of a tensor to a list of available GPU pointers
    std::unordered_map<int, std::vector<float*>> free_blocks;

    // Private constructor (Singleton pattern)
    CachingAllocator() {}

public:
    // Get the global instance of the allocator
    static CachingAllocator& get_instance() {
        static CachingAllocator instance;
        return instance;
    }

    // Allocate memory (or reuse an old block)
    float* allocate(int size) {
        // 1. Check if we already have a block of this size in the pool
        if (free_blocks.find(size) != free_blocks.end() && !free_blocks[size].empty()) {
            // Pop a recycled pointer off the back of the list and return it instantly
            float* ptr = free_blocks[size].back();
            free_blocks[size].pop_back();
            return ptr;
        }

        // 2. If no recycled block exists, we MUST ask the OS (cudaMalloc)
        float* ptr;
        cudaError_t err = cudaMalloc((void**)&ptr, size * sizeof(float));
        if (err != cudaSuccess) {
            throw std::runtime_error("Aakaar out of memory!");
        }
        return ptr;
    }

    // Free memory (put it in the pool, DON'T give it to the OS)
    void free(float* ptr, int size) {
        free_blocks[size].push_back(ptr);
    }

    // A utility to actually clear the cache and give memory back to the GPU
    void empty_cache() {
        for (auto& pair : free_blocks) {
            for (float* ptr : pair.second) {
                cudaFree(ptr); // ACTUALLY free the memory
            }
        }
        free_blocks.clear();
    }
};