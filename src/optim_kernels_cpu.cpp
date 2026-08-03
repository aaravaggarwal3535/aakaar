#include <cmath>
#include <memory>
#include <stdexcept>
#include <omp.h>
#include "tensor.h"

void run_cpu_sgd_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                       std::shared_ptr<Tensor> velocity,
                       float lr, float momentum, float weight_decay, float dampening,
                       int nesterov, int has_momentum, int velocity_initialized) {
    int n = p->size;
    if (n == 0) return;
    float* pp = p->fptr(); const float* gp = grad->fptr(); float* vp = velocity->fptr();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float g = gp[i];
        if (weight_decay != 0.0f) g += pp[i] * weight_decay;
        float update;
        if (has_momentum) {
            float v = velocity_initialized ? (vp[i] * momentum + g * (1.0f - dampening)) : g;
            vp[i] = v;
            update = nesterov ? (g + v * momentum) : v;
        } else {
            update = g;
        }
        pp[i] = pp[i] - update * lr;
    }
}

void run_cpu_adamw_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                         std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> v,
                         float lr, float beta1, float beta2, float eps, float weight_decay, int t) {
    int n = p->size;
    if (n == 0) return;
    float bc1_inv = 1.0f / (1.0f - std::pow(beta1, (float)t));
    float bc2_inv = 1.0f / (1.0f - std::pow(beta2, (float)t));
    float* pp = p->fptr(); const float* gp = grad->fptr(); float* mp = m->fptr(); float* vp = v->fptr();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float g = gp[i];
        float m_new = mp[i] * beta1 + g * (1.0f - beta1);
        float v_new = vp[i] * beta2 + g * g * (1.0f - beta2);
        mp[i] = m_new; vp[i] = v_new;
        float m_hat = m_new * bc1_inv;
        float v_hat = v_new * bc2_inv;
        pp[i] = pp[i] - pp[i] * (lr * weight_decay) - lr * (m_hat / (std::sqrt(v_hat) + eps));
    }
}

void run_cpu_adamax_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                          std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> u,
                          float lr, float beta1, float beta2, float eps, float weight_decay, int t) {
    int n = p->size;
    if (n == 0) return;
    float bc1_inv = 1.0f / (1.0f - std::pow(beta1, (float)t));
    float* pp = p->fptr(); const float* gp = grad->fptr(); float* mp = m->fptr(); float* up = u->fptr();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float g = gp[i];
        if (weight_decay != 0.0f) g += pp[i] * weight_decay;
        float m_new = mp[i] * beta1 + g * (1.0f - beta1);
        float u_new = std::max(up[i] * beta2, std::fabs(g));
        mp[i] = m_new; up[i] = u_new;
        pp[i] = pp[i] - (m_new * bc1_inv / (u_new + eps)) * lr;
    }
}

void run_cpu_nadam_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                         std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> v,
                         float lr, float beta1, float beta2, float eps, float weight_decay,
                         float mu_t, float mu_t1, float mu_product, float mu_product_next, int t) {
    int n = p->size;
    if (n == 0) return;
    float bc2_inv = 1.0f / (1.0f - std::pow(beta2, (float)t));
    float* pp = p->fptr(); const float* gp = grad->fptr(); float* mp = m->fptr(); float* vp = v->fptr();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float g = gp[i];
        if (weight_decay != 0.0f) g += pp[i] * weight_decay;
        float m_new = mp[i] * beta1 + g * (1.0f - beta1);
        float v_new = vp[i] * beta2 + g * g * (1.0f - beta2);
        mp[i] = m_new; vp[i] = v_new;
        float m_hat = (mu_t1 * m_new) / (1.0f - mu_product_next) + ((1.0f - mu_t) * g) / (1.0f - mu_product);
        float v_hat = v_new * bc2_inv;
        pp[i] = pp[i] - (m_hat / (std::sqrt(v_hat) + eps)) * lr;
    }
}

void run_cpu_radam_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                         std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> v,
                         float lr, float beta1, float beta2, float eps, float weight_decay,
                         int use_adaptive, float r_t, int t) {
    int n = p->size;
    if (n == 0) return;
    float bc1_inv = 1.0f / (1.0f - std::pow(beta1, (float)t));
    float bc2_inv = 1.0f / (1.0f - std::pow(beta2, (float)t));
    float* pp = p->fptr(); const float* gp = grad->fptr(); float* mp = m->fptr(); float* vp = v->fptr();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float g = gp[i];
        if (weight_decay != 0.0f) g += pp[i] * weight_decay;
        float m_new = mp[i] * beta1 + g * (1.0f - beta1);
        float v_new = vp[i] * beta2 + g * g * (1.0f - beta2);
        mp[i] = m_new; vp[i] = v_new;
        float m_hat = m_new * bc1_inv;
        float v_hat = v_new * bc2_inv;
        float denom = use_adaptive ? (std::sqrt(v_hat) + eps) : 1.0f;
        float scale = use_adaptive ? r_t : 1.0f;
        pp[i] = pp[i] - lr * scale * m_hat / denom;
    }
}

void run_cpu_rmsprop_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                          std::shared_ptr<Tensor> sq_avg, std::shared_ptr<Tensor> buf,
                          float lr, float alpha, float eps, float weight_decay, float momentum,
                          int has_momentum, int buf_initialized) {
    int n = p->size;
    if (n == 0) return;
    float* pp = p->fptr(); const float* gp = grad->fptr(); float* sp = sq_avg->fptr(); float* bp = buf->fptr();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float g = gp[i];
        if (weight_decay != 0.0f) g += pp[i] * weight_decay;
        float sq_new = sp[i] * alpha + g * g * (1.0f - alpha);
        sp[i] = sq_new;
        float update = g / (std::sqrt(sq_new) + eps);
        if (has_momentum) {
            float b = buf_initialized ? (bp[i] * momentum + update) : update;
            bp[i] = b;
            update = b;
        }
        pp[i] = pp[i] - update * lr;
    }
}

void run_cpu_adadelta_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                           std::shared_ptr<Tensor> sq_avg, std::shared_ptr<Tensor> acc_delta,
                           float lr, float rho, float eps, float weight_decay) {
    int n = p->size;
    if (n == 0) return;
    float* pp = p->fptr(); const float* gp = grad->fptr(); float* sp = sq_avg->fptr(); float* ap = acc_delta->fptr();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float g = gp[i];
        if (weight_decay != 0.0f) g += pp[i] * weight_decay;
        float sq_new = sp[i] * rho + g * g * (1.0f - rho);
        sp[i] = sq_new;
        float delta = std::sqrt(ap[i] + eps) / std::sqrt(sq_new + eps) * g;
        ap[i] = ap[i] * rho + delta * delta * (1.0f - rho);
        pp[i] = pp[i] - delta * lr;
    }
}

static float rprop_clip_cpu(float x, float lo, float hi) {
    float raised = lo + std::max(x - lo, 0.0f);
    return hi - std::max(hi - raised, 0.0f);
}

void run_cpu_rprop_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                        std::shared_ptr<Tensor> prev_grad, std::shared_ptr<Tensor> step_size,
                        float lr, float eta_minus, float eta_plus,
                        float step_min, float step_max, int first_step) {
    int n = p->size;
    if (n == 0) return;
    float* pp = p->fptr(); const float* gp = grad->fptr();
    float* pgp = prev_grad->fptr(); float* ssp = step_size->fptr();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float g = gp[i];
        float ss;
        if (first_step) {
            ss = lr;
        } else {
            float sign_agree_val = g * pgp[i];
            float sign_agree = (sign_agree_val > 0.0f) ? 1.0f : (sign_agree_val < 0.0f ? -1.0f : 0.0f);
            float factor = (sign_agree > 0.0f) ? eta_plus : (sign_agree < 0.0f ? eta_minus : 1.0f);
            ss = rprop_clip_cpu(ssp[i] * factor, step_min, step_max);
            if (sign_agree < 0.0f) g = 0.0f;
        }
        ssp[i] = ss;
        pgp[i] = g;
        float sign_g = (g > 0.0f) ? 1.0f : (g < 0.0f ? -1.0f : 0.0f);
        pp[i] = pp[i] - sign_g * ss;
    }
}

// append to optim_kernels_cpu.cpp
void run_cpu_add_channel_bias(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> bias,
                               std::shared_ptr<Tensor> out, int C, int HW, int total) {
    const float* xp = x->fptr(); const float* bp = bias->fptr(); float* op = out->fptr();
    #pragma omp parallel for
    for (int i = 0; i < total; ++i) {
        int c = (i / HW) % C;
        op[i] = xp[i] + bp[c];
    }
}

void run_cpu_channel_bias_grad(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> grad_bias, int C, int HW) {
    int total = grad_out->size;
    int B = total / (C * HW);
    float* gb = grad_bias->fptr();
    const float* go = grad_out->fptr();
    #pragma omp parallel for
    for (int c = 0; c < C; ++c) {
        double acc = 0.0;
        for (int b = 0; b < B; ++b) {
            const float* row = go + ((long long)b * C + c) * HW;
            for (int hw = 0; hw < HW; ++hw) acc += row[hw];
        }
        gb[c] = (float)acc;
    }
}

void run_cpu_channel_bias_grad_nd(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> grad_bias,
                                   int B, int C, int spatial) {
    const float* go = grad_out->fptr();
    float* gb = grad_bias->fptr();
    #pragma omp parallel for
    for (int c = 0; c < C; ++c) {
        double acc = 0.0;
        for (int b = 0; b < B; ++b) {
            const float* row = go + ((long long)b * C + c) * spatial;
            for (int i = 0; i < spatial; ++i) acc += row[i];
        }
        gb[c] = (float)acc;
    }
}