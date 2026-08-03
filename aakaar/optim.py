from . import no_grad
import numpy as np
import aakaar

class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0,
                 dampening=0.0, nesterov=False):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening == 0")
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self._velocity = [None] * len(self.parameters)

    def step(self):
        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                use_fused = p.dtype == "float32" and grad.dtype == "float32" and p.is_contiguous()

                if use_fused:
                    initialized = self._velocity[i] is not None
                    if self._velocity[i] is None:
                        self._velocity[i] = grad * 0.0
                    aakaar._C._sgd_step_fused(
                        p, grad, self._velocity[i],
                        self.lr, self.momentum, self.weight_decay, self.dampening,
                        int(self.nesterov), int(self.momentum != 0), int(initialized),
                    )
                else:
                    if self.weight_decay != 0:
                        grad = grad + p * self.weight_decay
                    if self.momentum != 0:
                        if self._velocity[i] is None:
                            self._velocity[i] = grad
                        else:
                            self._velocity[i] = self._velocity[i] * self.momentum + grad * (1 - self.dampening)
                        grad = grad + self._velocity[i] * self.momentum if self.nesterov else self._velocity[i]
                    p.copy_(p - grad * self.lr)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class Adam:
    # unchanged from the earlier fix — kept here for completeness
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [None] * len(self.parameters)
        self._v = [None] * len(self.parameters)
        self._t = 0

    def step(self):
        self._t += 1
        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                use_fused = p.dtype == "float32" and grad.dtype == "float32" and p.is_contiguous()
                if self._m[i] is None:
                    self._m[i] = grad * 0.0
                    self._v[i] = grad * 0.0
                if use_fused:
                    aakaar._C._adam_step_fused(
                        p, grad, self._m[i], self._v[i],
                        self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, self._t,
                    )
                else:
                    if self.weight_decay != 0:
                        grad = grad + p * self.weight_decay
                    self._m[i] = self._m[i] * self.beta1 + grad * (1 - self.beta1)
                    self._v[i] = self._v[i] * self.beta2 + (grad * grad) * (1 - self.beta2)
                    m_hat = self._m[i] / (1 - self.beta1 ** self._t)
                    v_hat = self._v[i] / (1 - self.beta2 ** self._t)
                    p.copy_(p - (m_hat / (v_hat.sqrt() + self.eps)) * self.lr)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class AdamW:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [None] * len(self.parameters)
        self._v = [None] * len(self.parameters)
        self._t = 0

    def step(self):
        self._t += 1
        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                use_fused = p.dtype == "float32" and grad.dtype == "float32" and p.is_contiguous()
                if self._m[i] is None:
                    self._m[i] = grad * 0.0
                    self._v[i] = grad * 0.0
                if use_fused:
                    aakaar._C._adamw_step_fused(
                        p, grad, self._m[i], self._v[i],
                        self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, self._t,
                    )
                else:
                    self._m[i] = self._m[i] * self.beta1 + grad * (1 - self.beta1)
                    self._v[i] = self._v[i] * self.beta2 + (grad * grad) * (1 - self.beta2)
                    m_hat = self._m[i] / (1 - self.beta1 ** self._t)
                    v_hat = self._v[i] / (1 - self.beta2 ** self._t)
                    p.copy_(p - p * (self.lr * self.weight_decay) - (m_hat / (v_hat.sqrt() + self.eps)) * self.lr)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class Adamax:
    def __init__(self, parameters, lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [None] * len(self.parameters)
        self._u = [None] * len(self.parameters)
        self._t = 0

    def step(self):
        self._t += 1
        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                use_fused = p.dtype == "float32" and grad.dtype == "float32" and p.is_contiguous()
                if self._m[i] is None:
                    self._m[i] = grad * 0.0
                    self._u[i] = grad * 0.0
                if use_fused:
                    aakaar._C._adamax_step_fused(
                        p, grad, self._m[i], self._u[i],
                        self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, self._t,
                    )
                else:
                    if self.weight_decay != 0:
                        grad = grad + p * self.weight_decay
                    self._m[i] = self._m[i] * self.beta1 + grad * (1 - self.beta1)
                    abs_grad = grad.abs()
                    scaled_u = self._u[i] * self.beta2
                    self._u[i] = scaled_u + (abs_grad - scaled_u).relu()
                    bias_correction = 1 - self.beta1 ** self._t
                    p.copy_(p - (self._m[i] / bias_correction / (self._u[i] + self.eps)) * self.lr)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class NAdam:
    def __init__(self, parameters, lr=0.002, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, momentum_decay=0.004):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum_decay = momentum_decay
        self._m = [None] * len(self.parameters)
        self._v = [None] * len(self.parameters)
        self._mu_product = 1.0
        self._t = 0

    def step(self):
        self._t += 1
        mu_t = self.beta1 * (1 - 0.5 * 0.96 ** (self._t * self.momentum_decay))
        mu_t1 = self.beta1 * (1 - 0.5 * 0.96 ** ((self._t + 1) * self.momentum_decay))
        mu_product_before = self._mu_product
        self._mu_product *= mu_t
        mu_product_next = self._mu_product * mu_t1

        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                use_fused = p.dtype == "float32" and grad.dtype == "float32" and p.is_contiguous()
                if self._m[i] is None:
                    self._m[i] = grad * 0.0
                    self._v[i] = grad * 0.0
                if use_fused:
                    aakaar._C._nadam_step_fused(
                        p, grad, self._m[i], self._v[i],
                        self.lr, self.beta1, self.beta2, self.eps, self.weight_decay,
                        mu_t, mu_t1, mu_product_before, mu_product_next, self._t,
                    )
                else:
                    if self.weight_decay != 0:
                        grad = grad + p * self.weight_decay
                    self._m[i] = self._m[i] * self.beta1 + grad * (1 - self.beta1)
                    self._v[i] = self._v[i] * self.beta2 + (grad * grad) * (1 - self.beta2)
                    m_hat = (mu_t1 * self._m[i]) / (1 - mu_product_next) + \
                            ((1 - mu_t) * grad) / (1 - mu_product_before)
                    v_hat = self._v[i] / (1 - self.beta2 ** self._t)
                    p.copy_(p - (m_hat / (v_hat.sqrt() + self.eps)) * self.lr)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class RAdam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [None] * len(self.parameters)
        self._v = [None] * len(self.parameters)
        self._t = 0
        self._rho_inf = 2.0 / (1 - self.beta2) - 1

    def step(self):
        self._t += 1
        rho_t = self._rho_inf - 2 * self._t * (self.beta2 ** self._t) / (1 - self.beta2 ** self._t)
        use_adaptive = rho_t > 4
        r_t = 0.0
        if use_adaptive:
            r_t = (((rho_t - 4) * (rho_t - 2) * self._rho_inf) /
                   ((self._rho_inf - 4) * (self._rho_inf - 2) * rho_t)) ** 0.5

        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                use_fused = p.dtype == "float32" and grad.dtype == "float32" and p.is_contiguous()
                if self._m[i] is None:
                    self._m[i] = grad * 0.0
                    self._v[i] = grad * 0.0
                if use_fused:
                    aakaar._C._radam_step_fused(
                        p, grad, self._m[i], self._v[i],
                        self.lr, self.beta1, self.beta2, self.eps, self.weight_decay,
                        int(use_adaptive), r_t, self._t,
                    )
                else:
                    if self.weight_decay != 0:
                        grad = grad + p * self.weight_decay
                    self._m[i] = self._m[i] * self.beta1 + grad * (1 - self.beta1)
                    self._v[i] = self._v[i] * self.beta2 + (grad * grad) * (1 - self.beta2)
                    m_hat = self._m[i] / (1 - self.beta1 ** self._t)
                    if use_adaptive:
                        v_hat = self._v[i] / (1 - self.beta2 ** self._t)
                        p.copy_(p - (r_t * m_hat / (v_hat.sqrt() + self.eps)) * self.lr)
                    else:
                        p.copy_(p - m_hat * self.lr)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class RMSprop:
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self._sq_avg = [None] * len(self.parameters)
        self._buf = [None] * len(self.parameters)

    def step(self):
        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                use_fused = p.dtype == "float32" and grad.dtype == "float32" and p.is_contiguous()
                if self._sq_avg[i] is None:
                    self._sq_avg[i] = grad * 0.0
                if self._buf[i] is None:
                    self._buf[i] = grad * 0.0

                if use_fused:
                    initialized = self._buf[i] is not None and self._sq_avg[i].sum().item() != 0.0 or True
                    # buf_initialized must track "has a momentum update ever been written",
                    # not "does the tensor exist" (it always exists once allocated above).
                    buf_was_used = getattr(self, "_buf_used", None)
                    if buf_was_used is None:
                        buf_was_used = [False] * len(self.parameters)
                        self._buf_used = buf_was_used
                    aakaar._C._rmsprop_step_fused(
                        p, grad, self._sq_avg[i], self._buf[i],
                        self.lr, self.alpha, self.eps, self.weight_decay, self.momentum,
                        int(self.momentum > 0), int(self._buf_used[i]),
                    )
                    if self.momentum > 0:
                        self._buf_used[i] = True
                else:
                    if self.weight_decay != 0:
                        grad = grad + p * self.weight_decay
                    self._sq_avg[i] = self._sq_avg[i] * self.alpha + (grad * grad) * (1 - self.alpha)
                    update = grad / (self._sq_avg[i].sqrt() + self.eps)
                    if self.momentum > 0:
                        if self._buf[i] is None:
                            self._buf[i] = update
                        else:
                            self._buf[i] = self._buf[i] * self.momentum + update
                        update = self._buf[i]
                    p.copy_(p - update * self.lr)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class Adagrad:
    def __init__(self, parameters, lr=0.01, eps=1e-10, weight_decay=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self._sum_sq = [None] * len(self.parameters)

    def step(self):
        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if self.weight_decay != 0:
                    grad = grad + p * self.weight_decay
                if self._sum_sq[i] is None:
                    self._sum_sq[i] = grad * 0.0
                self._sum_sq[i] = self._sum_sq[i] + grad * grad
                p.copy_(p - (grad / (self._sum_sq[i].sqrt() + self.eps)) * self.lr)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class Adadelta:
    def __init__(self, parameters, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self._sq_avg = [None] * len(self.parameters)
        self._acc_delta = [None] * len(self.parameters)

    def step(self):
        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                use_fused = p.dtype == "float32" and grad.dtype == "float32" and p.is_contiguous()
                if self._sq_avg[i] is None:
                    self._sq_avg[i] = grad * 0.0
                    self._acc_delta[i] = grad * 0.0

                if use_fused:
                    aakaar._C._adadelta_step_fused(
                        p, grad, self._sq_avg[i], self._acc_delta[i],
                        self.lr, self.rho, self.eps, self.weight_decay,
                    )
                else:
                    if self.weight_decay != 0:
                        grad = grad + p * self.weight_decay
                    self._sq_avg[i] = self._sq_avg[i] * self.rho + (grad * grad) * (1 - self.rho)
                    delta = (self._acc_delta[i] + self.eps).sqrt() / (self._sq_avg[i] + self.eps).sqrt() * grad
                    self._acc_delta[i] = self._acc_delta[i] * self.rho + (delta * delta) * (1 - self.rho)
                    p.copy_(p - delta * self.lr)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class ASGD:
    """Averaged SGD (Polyak-Ruppert averaging): maintains a running average
    of the parameter trajectory alongside the actively-updated parameter.
    `.averaged_parameters` gives the averaged values for evaluation, matching
    the common ASGD usage pattern (train with the raw params, evaluate with
    the average)."""
    def __init__(self, parameters, lr=0.01, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.lambd = lambd
        self.alpha = alpha
        self.t0 = t0
        self.weight_decay = weight_decay
        self._eta = lr
        self._mu = 1.0
        self._t = 0
        self._ax = [p * 1.0 for p in self.parameters]  # averaged copy

    @property
    def averaged_parameters(self):
        return self._ax

    def step(self):
        self._t += 1
        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if self.weight_decay != 0:
                    grad = grad + p * self.weight_decay
                p.copy_(p * (1 - self.lambd * self._eta) - grad * self._eta)
                if self._mu != 1:
                    self._ax[i].copy_(self._ax[i] + (p - self._ax[i]) * self._mu)
                else:
                    self._ax[i].copy_(p)
                self._eta = self.lr / ((1 + self.lambd * self.lr * self._t) ** self.alpha)
                self._mu = 1 / max(1, self._t - self.t0)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class Rprop:
    def __init__(self, parameters, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
        self.parameters = list(parameters)
        self.lr = lr
        self.eta_minus, self.eta_plus = etas
        self.step_min, self.step_max = step_sizes
        self._prev_grad = [None] * len(self.parameters)
        self._step_size = [None] * len(self.parameters)

    @staticmethod
    def _clip(t, lo, hi):
        raised = lo + (t - lo).relu()
        return hi - (hi - raised).relu()

    def step(self):
        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                use_fused = p.dtype == "float32" and grad.dtype == "float32" and p.is_contiguous()
                first_step = self._prev_grad[i] is None

                if use_fused:
                    if self._prev_grad[i] is None:
                        self._prev_grad[i] = grad * 0.0
                        self._step_size[i] = grad * 0.0
                    aakaar._C._rprop_step_fused(
                        p, grad, self._prev_grad[i], self._step_size[i],
                        self.lr, self.eta_minus, self.eta_plus, self.step_min, self.step_max,
                        int(first_step),
                    )
                else:
                    if first_step:
                        step_size = grad * 0.0 + self.lr
                    else:
                        sign_agree = (grad * self._prev_grad[i]).sign()
                        pos_mask = sign_agree.relu()
                        neg_mask = (-sign_agree).relu()
                        same_mask = 1 - pos_mask - neg_mask
                        step_size = self._step_size[i] * (pos_mask * self.eta_plus +
                                                           neg_mask * self.eta_minus +
                                                           same_mask * 1.0)
                        step_size = self._clip(step_size, self.step_min, self.step_max)
                        grad = grad * (1 - neg_mask)
                    self._step_size[i] = step_size
                    self._prev_grad[i] = grad
                    update = grad.sign() * step_size
                    p.copy_(p - update)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class Muon:
    """Orthogonalized momentum optimizer (Jordan et al., 2024) for 2D
    (matrix-shaped) parameters: momentum-SGD, then orthogonalize the update
    via a Newton-Schulz iteration (matrix square-root-free approximation of
    U@V^T from the update's SVD) before applying it. Uses only matmul/
    transpose, both already implemented. Only applies to parameters with
    exactly 2 dimensions (weight matrices) — 1D params (biases) fall back
    to plain momentum-SGD, matching the reference implementation's approach."""
    def __init__(self, parameters, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay
        self._buf = [None] * len(self.parameters)

    @staticmethod
    def _newton_schulz_orthogonalize(G, steps):
        norm = (G * G).sum().sqrt()
        X = G / (norm + 1e-7)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(steps):
            A = X.transpose(0, 1).contiguous() @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
        return X

    def step(self):
        with no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None:
                    continue
                grad = p.grad
                if self.weight_decay != 0:
                    grad = grad + p * self.weight_decay

                if self._buf[i] is None:
                    self._buf[i] = grad
                else:
                    self._buf[i] = self._buf[i] * self.momentum + grad * (1 - self.momentum)

                if len(p.shape) == 2:
                    update = self._newton_schulz_orthogonalize(self._buf[i], self.ns_steps)
                else:
                    update = self._buf[i]

                p.copy_(p - update * self.lr)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()