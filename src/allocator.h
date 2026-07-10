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
    long long cache_hits = 0;
    long long cache_misses = 0;

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
        if (free_blocks.find(size) != free_blocks.end() && !free_blocks[size].empty()) {
            float* ptr = free_blocks[size].back();
            free_blocks[size].pop_back();
            cache_hits++;
            return ptr;
        }
        cache_misses++;
        float* ptr;
        cudaError_t err = cudaMalloc((void**)&ptr, size * sizeof(float));
        if (err != cudaSuccess) {
            throw std::runtime_error("Aakaar out of memory!");
        }
        return ptr;
    }
    void free(float* ptr, int size) {
        free_blocks[size].push_back(ptr);
    }
    void empty_cache() {
        for (auto& pair : free_blocks) {
            for (float* ptr : pair.second) {
                cudaFree(ptr);
            }
        }
        free_blocks.clear();
    }
    std::pair<long long,long long> get_stats() { return {cache_hits, cache_misses}; }
};