# AdamW

Aakaar provides a native, highly optimized implementation of AdamW, which modifies the standard Adam algorithm by decoupling weight decay from the gradient update (introduced by Loshchilov & Hutter, 2017).

## What it does

In standard stochastic gradient descent (SGD), adding L2 regularization to the loss function is mathematically identical to applying weight decay. However, in adaptive optimizers like Adam, this equivalence breaks down. 

If you use standard Adam with L2 regularization, the weight decay penalty is added to the gradient, meaning it gets scaled by Adam's moving average of the squared gradients. This results in parameters with large gradients being penalized *less* by weight decay, which is counterproductive. 

AdamW solves this by completely decoupling the weight decay step. It calculates the momentum and variance using only the raw loss gradients, and applies the weight decay directly to the parameter values during the final update step.

## The Math (What it means)

Like standard Adam, AdamW tracks the first and second moments of the raw gradients (without any L2 penalty attached):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t)$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(\theta_t))^2$$

After applying the standard bias corrections to get $\hat{m}_t$ and $\hat{v}_t$, AdamW updates the parameters $\theta$ by subtracting both the adaptive gradient step and the decoupled weight decay step simultaneously:
$$\theta_{t+1} = \theta_t - \eta \lambda \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where $\eta$ is the learning rate and $\lambda$ is the weight decay coefficient.

## Usage

```python
import aakaar.optim as optim

# Initialize AdamW (Note: weight_decay defaults to 0.01)
optimizer = optim.AdamW(
    model.parameters(), 
    lr=0.001, 
    betas=(0.9, 0.999), 
    eps=1e-8, 
    weight_decay=0.01
)

# Standard training loop execution
optimizer.zero_grad()
loss = model(inputs).sum()
loss.backward()
optimizer.step()
```

## Parameters

- parameters: Iterable of Aakaar tensors to optimize.

- lr (float, default: 0.001): The learning rate (step size).

- betas (tuple, default: (0.9, 0.999)) : Coefficients used for computing the running averages of the gradient and its square.

- eps (float, default: 1e-8): A microscopically small term added to the denominator to prevent division by zero and improve numerical stability.

- weight_decay (float, default: 0.01): The decoupled weight decay rate. Note that this defaults to 0.01, unlike standard Adam which defaults to 0.0.

## When to use it
AdamW should be your default optimizer for almost all modern deep learning tasks.

It consistently yields better training loss and generalizes much better than standard Adam. It is the absolute industry standard for training Transformer architectures (like LLMs and Vision Transformers) and heavily regularized convolutional networks. If you are unsure which optimizer to choose for a new model, start with AdamW.