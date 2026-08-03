#pragma once
#ifndef AAKAAR_NO_CUDA
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <string>

// Host-side counterpart to CachingAllocator: pools page-locked (pinned)
// staging buffers keyed by byte size, so from_numpy() doesn't pay for a
// fresh cudaHostAlloc + cudaFreeHost on every single call. Each buffer
// carries a cudaEvent marking when its most recent async H2D copy
// completed; a buffer is only handed back out once that event has been
// reached (cudaEventSynchronize is near-instant if the copy already
// finished, which it almost always has by reuse time — many CPU-side
// operations happen between two from_numpy() calls of the same shape).
class PinnedAllocator {
public:
    static PinnedAllocator& get_instance() {
        static PinnedAllocator instance;
        return instance;
    }

    struct Buffer {
        void* ptr;
        cudaEvent_t ready_event;
    };

    Buffer acquire(size_t bytes) {
        auto& pool = free_pool[bytes];
        if (!pool.empty()) {
            Buffer buf = pool.back();
            pool.pop_back();
            cudaEventSynchronize(buf.ready_event);
            return buf;
        }
        Buffer buf;
        cudaError_t err = cudaHostAlloc(&buf.ptr, bytes, cudaHostAllocDefault);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("Aakaar pinned host allocation failed: ") + cudaGetErrorString(err) +
                                      " (requested " + std::to_string(bytes) + " bytes)");
        cudaEventCreateWithFlags(&buf.ready_event, cudaEventDisableTiming);
        return buf;
    }

    // stream: whichever stream the async copy was launched on — the event
    // is recorded there so the next acquire() of this same-size buffer
    // waits for exactly that copy, not an unrelated one.
    void release(size_t bytes, Buffer buf, cudaStream_t stream) {
        cudaEventRecord(buf.ready_event, stream);
        free_pool[bytes].push_back(buf);
    }

private:
    PinnedAllocator() {}
    std::unordered_map<size_t, std::vector<Buffer>> free_pool;
};
#endif // AAKAAR_NO_CUDA