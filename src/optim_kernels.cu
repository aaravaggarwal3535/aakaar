#include <cuda_runtime.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include "tensor.h"

static int opt_blocks(int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;
    return blocks;
}

// ---------------------------------------------------------------------------
// SGD (momentum / dampening / nesterov / weight_decay), single fused pass.
// velocity_initialized mirrors the python "self._velocity[i] is None" check:
// when 0, velocity is seeded directly from grad (not from an EMA against a
// zero buffer) on this call, matching the original semantics exactly.
// ---------------------------------------------------------------------------
__global__ void sgd_step_kernel(float* __restrict__ p, const float* __restrict__ grad,
                                 float* __restrict__ velocity,
                                 float lr, float momentum, float weight_decay, float dampening,
                                 int nesterov, int has_momentum, int velocity_initialized, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float g = grad[i];
        if (weight_decay != 0.0f) g += p[i] * weight_decay;

        float update;
        if (has_momentum) {
            float v = velocity_initialized ? (velocity[i] * momentum + g * (1.0f - dampening)) : g;
            velocity[i] = v;
            update = nesterov ? (g + v * momentum) : v;
        } else {
            update = g;
        }
        p[i] = p[i] - update * lr;
    }
}

void run_cuda_sgd_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                        std::shared_ptr<Tensor> velocity,
                        float lr, float momentum, float weight_decay, float dampening,
                        int nesterov, int has_momentum, int velocity_initialized) {
    int n = p->size;
    if (n == 0) return;
    sgd_step_kernel<<<opt_blocks(n), 256>>>(
        p->fptr(), grad->fptr(), velocity->fptr(),
        lr, momentum, weight_decay, dampening, nesterov, has_momentum, velocity_initialized, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed (sgd_step_fused): ") + cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// AdamW — decoupled weight decay (applied to p directly, not folded into grad).
// ---------------------------------------------------------------------------
__global__ void adamw_step_kernel(float* __restrict__ p, const float* __restrict__ grad,
                                   float* __restrict__ m, float* __restrict__ v,
                                   float lr, float beta1, float beta2, float eps, float weight_decay,
                                   float bc1_inv, float bc2_inv, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float g = grad[i];
        float m_new = m[i] * beta1 + g * (1.0f - beta1);
        float v_new = v[i] * beta2 + g * g * (1.0f - beta2);
        m[i] = m_new; v[i] = v_new;
        float m_hat = m_new * bc1_inv;
        float v_hat = v_new * bc2_inv;
        p[i] = p[i] - p[i] * (lr * weight_decay) - lr * (m_hat / (sqrtf(v_hat) + eps));
    }
}

void run_cuda_adamw_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                          std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> v,
                          float lr, float beta1, float beta2, float eps, float weight_decay, int t) {
    int n = p->size;
    if (n == 0) return;
    float bc1_inv = 1.0f / (1.0f - powf(beta1, (float)t));
    float bc2_inv = 1.0f / (1.0f - powf(beta2, (float)t));
    adamw_step_kernel<<<opt_blocks(n), 256>>>(
        p->fptr(), grad->fptr(), m->fptr(), v->fptr(),
        lr, beta1, beta2, eps, weight_decay, bc1_inv, bc2_inv, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed (adamw_step_fused): ") + cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// Adamax — infinity-norm second moment. Uses fmaxf directly (a raw kernel
// has no need for the relu-identity trick the tensor-op-level code used to
// build max() out of relu — that hack only existed because no elementwise
// max(tensor,tensor) primitive existed at that layer).
// ---------------------------------------------------------------------------
__global__ void adamax_step_kernel(float* __restrict__ p, const float* __restrict__ grad,
                                    float* __restrict__ m, float* __restrict__ u,
                                    float lr, float beta1, float beta2, float eps, float weight_decay,
                                    float bc1_inv, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float g = grad[i];
        if (weight_decay != 0.0f) g += p[i] * weight_decay;
        float m_new = m[i] * beta1 + g * (1.0f - beta1);
        float u_new = fmaxf(u[i] * beta2, fabsf(g));
        m[i] = m_new; u[i] = u_new;
        p[i] = p[i] - (m_new * bc1_inv / (u_new + eps)) * lr;
    }
}

void run_cuda_adamax_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                           std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> u,
                           float lr, float beta1, float beta2, float eps, float weight_decay, int t) {
    int n = p->size;
    if (n == 0) return;
    float bc1_inv = 1.0f / (1.0f - powf(beta1, (float)t));
    adamax_step_kernel<<<opt_blocks(n), 256>>>(
        p->fptr(), grad->fptr(), m->fptr(), u->fptr(),
        lr, beta1, beta2, eps, weight_decay, bc1_inv, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed (adamax_step_fused): ") + cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// NAdam — mu_t/mu_t1/mu_product/mu_product_next are pure scalars (same for
// every parameter within one step() call), computed once on the host in
// python exactly as the unfused path already did; the kernel only does the
// per-element part.
// ---------------------------------------------------------------------------
__global__ void nadam_step_kernel(float* __restrict__ p, const float* __restrict__ grad,
                                   float* __restrict__ m, float* __restrict__ v,
                                   float lr, float beta1, float beta2, float eps, float weight_decay,
                                   float mu_t, float mu_t1, float mu_product, float mu_product_next,
                                   float bc2_inv, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float g = grad[i];
        if (weight_decay != 0.0f) g += p[i] * weight_decay;
        float m_new = m[i] * beta1 + g * (1.0f - beta1);
        float v_new = v[i] * beta2 + g * g * (1.0f - beta2);
        m[i] = m_new; v[i] = v_new;

        float m_hat = (mu_t1 * m_new) / (1.0f - mu_product_next) + ((1.0f - mu_t) * g) / (1.0f - mu_product);
        float v_hat = v_new * bc2_inv;
        p[i] = p[i] - (m_hat / (sqrtf(v_hat) + eps)) * lr;
    }
}

void run_cuda_nadam_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                          std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> v,
                          float lr, float beta1, float beta2, float eps, float weight_decay,
                          float mu_t, float mu_t1, float mu_product, float mu_product_next, int t) {
    int n = p->size;
    if (n == 0) return;
    float bc2_inv = 1.0f / (1.0f - powf(beta2, (float)t));
    nadam_step_kernel<<<opt_blocks(n), 256>>>(
        p->fptr(), grad->fptr(), m->fptr(), v->fptr(),
        lr, beta1, beta2, eps, weight_decay, mu_t, mu_t1, mu_product, mu_product_next, bc2_inv, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed (nadam_step_fused): ") + cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// RAdam — rho_t/r_t/use_adaptive are scalars computed once per step() call
// on the host (they don't depend on any per-element data), exactly matching
// the unfused path's per-call (not per-parameter) rho_t computation.
// ---------------------------------------------------------------------------
__global__ void radam_step_kernel(float* __restrict__ p, const float* __restrict__ grad,
                                   float* __restrict__ m, float* __restrict__ v,
                                   float lr, float beta1, float beta2, float eps, float weight_decay,
                                   int use_adaptive, float r_t, float bc1_inv, float bc2_inv, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float g = grad[i];
        if (weight_decay != 0.0f) g += p[i] * weight_decay;
        float m_new = m[i] * beta1 + g * (1.0f - beta1);
        float v_new = v[i] * beta2 + g * g * (1.0f - beta2);
        m[i] = m_new; v[i] = v_new;

        float m_hat = m_new * bc1_inv;
        float v_hat = v_new * bc2_inv;
        float denom = use_adaptive ? (sqrtf(v_hat) + eps) : 1.0f;
        float scale = use_adaptive ? r_t : 1.0f;
        p[i] = p[i] - lr * scale * m_hat / denom;
    }
}

void run_cuda_radam_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                          std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> v,
                          float lr, float beta1, float beta2, float eps, float weight_decay,
                          int use_adaptive, float r_t, int t) {
    int n = p->size;
    if (n == 0) return;
    float bc1_inv = 1.0f / (1.0f - powf(beta1, (float)t));
    float bc2_inv = 1.0f / (1.0f - powf(beta2, (float)t));
    radam_step_kernel<<<opt_blocks(n), 256>>>(
        p->fptr(), grad->fptr(), m->fptr(), v->fptr(),
        lr, beta1, beta2, eps, weight_decay, use_adaptive, r_t, bc1_inv, bc2_inv, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed (radam_step_fused): ") + cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// RMSprop (with optional plain momentum on top of the RMS-scaled update).
// buf_initialized mirrors "self._buf[i] is None" — first use seeds buf
// directly from `update` rather than an EMA against zero.
// ---------------------------------------------------------------------------
__global__ void rmsprop_step_kernel(float* __restrict__ p, const float* __restrict__ grad,
                                     float* __restrict__ sq_avg, float* __restrict__ buf,
                                     float lr, float alpha, float eps, float weight_decay, float momentum,
                                     int has_momentum, int buf_initialized, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float g = grad[i];
        if (weight_decay != 0.0f) g += p[i] * weight_decay;
        float sq_new = sq_avg[i] * alpha + g * g * (1.0f - alpha);
        sq_avg[i] = sq_new;
        float update = g / (sqrtf(sq_new) + eps);

        if (has_momentum) {
            float b = buf_initialized ? (buf[i] * momentum + update) : update;
            buf[i] = b;
            update = b;
        }
        p[i] = p[i] - update * lr;
    }
}

void run_cuda_rmsprop_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                           std::shared_ptr<Tensor> sq_avg, std::shared_ptr<Tensor> buf,
                           float lr, float alpha, float eps, float weight_decay, float momentum,
                           int has_momentum, int buf_initialized) {
    int n = p->size;
    if (n == 0) return;
    rmsprop_step_kernel<<<opt_blocks(n), 256>>>(
        p->fptr(), grad->fptr(), sq_avg->fptr(), buf->fptr(),
        lr, alpha, eps, weight_decay, momentum, has_momentum, buf_initialized, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed (rmsprop_step_fused): ") + cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// Adadelta — no first-step special case needed: sq_avg/acc_delta both start
// at exactly zero, which is what the recurrence naturally assumes.
// ---------------------------------------------------------------------------
__global__ void adadelta_step_kernel(float* __restrict__ p, const float* __restrict__ grad,
                                      float* __restrict__ sq_avg, float* __restrict__ acc_delta,
                                      float lr, float rho, float eps, float weight_decay, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float g = grad[i];
        if (weight_decay != 0.0f) g += p[i] * weight_decay;
        float sq_new = sq_avg[i] * rho + g * g * (1.0f - rho);
        sq_avg[i] = sq_new;
        float delta = sqrtf(acc_delta[i] + eps) / sqrtf(sq_new + eps) * g;
        acc_delta[i] = acc_delta[i] * rho + delta * delta * (1.0f - rho);
        p[i] = p[i] - delta * lr;
    }
}

void run_cuda_adadelta_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                             std::shared_ptr<Tensor> sq_avg, std::shared_ptr<Tensor> acc_delta,
                             float lr, float rho, float eps, float weight_decay) {
    int n = p->size;
    if (n == 0) return;
    adadelta_step_kernel<<<opt_blocks(n), 256>>>(
        p->fptr(), grad->fptr(), sq_avg->fptr(), acc_delta->fptr(),
        lr, rho, eps, weight_decay, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed (adadelta_step_fused): ") + cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// Rprop — resilient backprop. Per-element data-dependent branches here are
// fine (a handful of comparisons, not a memory-bound loop), unlike the
// tensor-op-level code which needed to avoid data-dependent branching by
// building masks out of sign()/relu() since it had no per-element control
// flow available at all.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float rprop_clip(float x, float lo, float hi) {
    float raised = lo + fmaxf(x - lo, 0.0f);
    return hi - fmaxf(hi - raised, 0.0f);
}

__global__ void rprop_step_kernel(float* __restrict__ p, const float* __restrict__ grad,
                                   float* __restrict__ prev_grad, float* __restrict__ step_size,
                                   float lr, float eta_minus, float eta_plus,
                                   float step_min, float step_max, int first_step, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float g = grad[i];
        float ss;
        if (first_step) {
            ss = lr;
        } else {
            float sign_agree_val = g * prev_grad[i];
            float sign_agree = (sign_agree_val > 0.0f) ? 1.0f : (sign_agree_val < 0.0f ? -1.0f : 0.0f);
            float factor = (sign_agree > 0.0f) ? eta_plus : (sign_agree < 0.0f ? eta_minus : 1.0f);
            ss = rprop_clip(step_size[i] * factor, step_min, step_max);
            if (sign_agree < 0.0f) g = 0.0f;  // zero grad where sign flipped, torch's convention
        }
        step_size[i] = ss;
        prev_grad[i] = g;
        float sign_g = (g > 0.0f) ? 1.0f : (g < 0.0f ? -1.0f : 0.0f);
        p[i] = p[i] - sign_g * ss;
    }
}

void run_cuda_rprop_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                          std::shared_ptr<Tensor> prev_grad, std::shared_ptr<Tensor> step_size,
                          float lr, float eta_minus, float eta_plus,
                          float step_min, float step_max, int first_step) {
    int n = p->size;
    if (n == 0) return;
    rprop_step_kernel<<<opt_blocks(n), 256>>>(
        p->fptr(), grad->fptr(), prev_grad->fptr(), step_size->fptr(),
        lr, eta_minus, eta_plus, step_min, step_max, first_step, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed (rprop_step_fused): ") + cudaGetErrorString(err));
}

// append to optim_kernels.cu
__global__ void add_channel_bias_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                                         float* __restrict__ out, int C, int HW, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int c = (i / HW) % C;
        out[i] = x[i] + bias[c];
    }
}

void run_cuda_add_channel_bias(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> bias,
                                std::shared_ptr<Tensor> out, int C, int HW, int total) {
    if (total == 0) return;
    add_channel_bias_kernel<<<opt_blocks(total), 256>>>(x->fptr(), bias->fptr(), out->fptr(), C, HW, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed (add_channel_bias): ") + cudaGetErrorString(err));
}

// Single-pass reduction for conv's bias gradient: (B,C,H,W) -> (1,C,1,1).
// Replaces reduce_grad_to_shape's generic 3-launch path (one sum_axis per
// collapsed dimension, each re-reading the previous pass's materialized
// intermediate) with one pass that reads the full gradient tensor exactly
// once. C is small (channel counts), so atomicAdd contention per output
// element is low relative to the memory-traffic savings.
__global__ void channel_bias_grad_kernel(const float* __restrict__ grad_out, float* __restrict__ grad_bias,
                                          int C, int HW, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int c = (i / HW) % C;
        atomicAdd(&grad_bias[c], grad_out[i]);
    }
}

void run_cuda_channel_bias_grad(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> grad_bias, int C, int HW) {
    grad_bias->fill_zero();
    int total = grad_out->size;
    if (total == 0) return;
    channel_bias_grad_kernel<<<opt_blocks(total), 256>>>(grad_out->fptr(), grad_bias->fptr(), C, HW, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed (channel_bias_grad): ") + cudaGetErrorString(err));
}

// Single-pass channel-bias-gradient reduction: (B,C,D,H,W) or (B,C,H,W)
// or (B,C,L) -> (1,C,1,...). One block per channel — each block reduces
// its own B*spatial elements via a shared-memory tree, then writes its
// single result directly. No atomics anywhere: every block owns a
// distinct output element, so there is no contention to create, unlike
// a naive one-thread-per-input-element + atomicAdd approach (which was
// tried and reverted for Conv2d after causing severe contention when
// millions of threads fought over a handful of addresses).
__global__ void channel_bias_grad_blockreduce_kernel(const float* __restrict__ grad_out,
                                                       float* __restrict__ grad_bias,
                                                       int B, int C, int spatial) {
    extern __shared__ float sdata[];
    int c = blockIdx.x;
    int tid = threadIdx.x;

    double local = 0.0;
    // Grid-stride over every (b, spatial_idx) pair that belongs to this channel.
    for (int b = 0; b < B; ++b) {
        const float* row = grad_out + ((long long)b * C + c) * spatial;
        for (int i = tid; i < spatial; i += blockDim.x) {
            local += (double)row[i];
        }
    }
    sdata[tid] = (float)local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) grad_bias[c] = sdata[0];
}

void run_cuda_channel_bias_grad_nd(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> grad_bias,
                                    int B, int C, int spatial) {
    int threads = 256;
    channel_bias_grad_blockreduce_kernel<<<C, threads, threads * sizeof(float)>>>(
        grad_out->fptr(), grad_bias->fptr(), B, C, spatial);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed (channel_bias_grad_nd): ") + cudaGetErrorString(err));
}