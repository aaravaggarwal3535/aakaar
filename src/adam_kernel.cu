#include <cuda_runtime.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include "tensor.h"

// Fully fused Adam parameter update. Does weight decay, both moment
// updates, bias correction, and the parameter update in ONE kernel launch
// per tensor (vs. ~13 separate elementwise-op launches in the generic
// python composition). Writes p/m/v in place; never touches autograd —
// this is only ever called from Optimizer.step(), which must not be
// tracked by the graph regardless.
__global__ void adam_step_kernel(float* __restrict__ p, const float* __restrict__ grad,
                                  float* __restrict__ m, float* __restrict__ v,
                                  float lr, float beta1, float beta2, float eps,
                                  float weight_decay,
                                  float bias_correction1_inv, float bias_correction2_inv,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float g = grad[i];
        if (weight_decay != 0.0f) g += p[i] * weight_decay;

        float m_new = m[i] * beta1 + g * (1.0f - beta1);
        float v_new = v[i] * beta2 + g * g * (1.0f - beta2);
        m[i] = m_new;
        v[i] = v_new;

        float m_hat = m_new * bias_correction1_inv;
        float v_hat = v_new * bias_correction2_inv;
        p[i] = p[i] - lr * (m_hat / (sqrtf(v_hat) + eps));
    }
}

static int adam_blocks(int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;
    return blocks;
}

// t is the 1-indexed step count (same convention as the existing python
// Adam: self._t += 1 happens before this is called).
void run_cuda_adam_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                         std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> v,
                         float lr, float beta1, float beta2, float eps,
                         float weight_decay, int t) {
    if (!p->is_contiguous() || !grad->is_contiguous() || !m->is_contiguous() || !v->is_contiguous())
        throw std::invalid_argument("adam_step_fused: all tensors must be contiguous.");
    if (p->size != grad->size || p->size != m->size || p->size != v->size)
        throw std::invalid_argument("adam_step_fused: p/grad/m/v must have matching sizes.");

    int n = p->size;
    if (n == 0) return;

    float bias_correction1_inv = 1.0f / (1.0f - powf(beta1, (float)t));
    float bias_correction2_inv = 1.0f / (1.0f - powf(beta2, (float)t));

    adam_step_kernel<<<adam_blocks(n), 256>>>(
        p->fptr(), grad->fptr(), m->fptr(), v->fptr(),
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1_inv, bias_correction2_inv, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel launch failed (adam_step_fused): ") + cudaGetErrorString(err));
}