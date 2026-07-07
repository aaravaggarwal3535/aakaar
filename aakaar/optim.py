from . import no_grad

class SGD:
    """Basic stochastic gradient descent optimizer."""
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        with no_grad():
            for p in self.parameters:
                if p.grad is not None:
                    updated = p - p.grad * self.lr
                    p.copy_(updated)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()