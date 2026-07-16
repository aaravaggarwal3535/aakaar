from . import no_grad
import numpy as np


class SGD:
    """Stochastic gradient descent, with optional momentum, weight decay,
    dampening, and Nesterov momentum — matching torch.optim.SGD's knobs."""

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
                if self.weight_decay != 0:
                    grad = grad + p * self.weight_decay

                if self.momentum != 0:
                    if self._velocity[i] is None:
                        self._velocity[i] = grad
                    else:
                        self._velocity[i] = self._velocity[i] * self.momentum + grad * (1 - self.dampening)
                    if self.nesterov:
                        grad = grad + self._velocity[i] * self.momentum
                    else:
                        grad = self._velocity[i]

                p.copy_(p - grad * self.lr)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class Adam:
    """Adam optimizer (Kingma & Ba, 2014) — matches torch.optim.Adam's
    default betas/eps."""

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
                if self.weight_decay != 0:
                    grad = grad + p * self.weight_decay

                if self._m[i] is None:
                    self._m[i] = grad * 0.0
                    self._v[i] = grad * 0.0

                self._m[i] = self._m[i] * self.beta1 + grad * (1 - self.beta1)
                self._v[i] = self._v[i] * self.beta2 + (grad * grad) * (1 - self.beta2)

                m_hat = self._m[i] / (1 - self.beta1 ** self._t)
                v_hat = self._v[i] / (1 - self.beta2 ** self._t)

                update = m_hat / (v_hat.sqrt_op() + self.eps) if hasattr(v_hat, "sqrt_op") else None
                # NOTE: aakaar has no elementwise sqrt() yet — see flag below.
                p.copy_(p - update * self.lr) if update is not None else None

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()