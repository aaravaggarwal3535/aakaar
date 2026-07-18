# Rprop

Aakaar provides a native implementation of Rprop (Resilient Backpropagation), an optimization algorithm that determines the update step size based solely on the *sign* of the gradient, completely ignoring its magnitude.

## What it does

In standard gradient descent algorithms, the size of the parameter update is proportional to the magnitude of the gradient. This can lead to problems in regions of the loss landscape with very small gradients (such as plateaus), where the updates become agonizingly slow, or regions with very steep gradients, where the updates become violently large.

Rprop solves this by entirely discarding the gradient's magnitude. It maintains a separate step size for every single parameter. It then looks at the *direction* (sign) of the gradient over two consecutive steps:
*   If the gradient points in the **same direction** as the last step, it means we are safely moving down a slope, so Rprop **increases** the step size to move faster.
*   If the gradient **changes direction**, it means we just jumped over a local minimum, so Rprop **decreases** the step size to backtrack and fine-tune.

## The Math (What it means)

For each parameter $\theta$, Rprop tracks a step size $\Delta_t$. Let $g_t$ be the gradient $\nabla L(\theta_t)$ at time step $t$.

First, Rprop compares the sign of the current gradient $g_t$ with the previous gradient $g_{t-1}$:

1.  **Same direction ($g_{t-1} \cdot g_t > 0$):**
    The step size is increased by a factor $\eta^+$, capped at a maximum $\Delta_{max}$:
    $$\Delta_t = \min(\Delta_{t-1} \cdot \eta^+, \Delta_{max})$$

2.  **Changed direction ($g_{t-1} \cdot g_t < 0$):**
    The step size is decreased by a factor $\eta^-$, bounded by a minimum $\Delta_{min}$:
    $$\Delta_t = \max(\Delta_{t-1} \cdot \eta^-, \Delta_{min})$$
    Additionally, the current gradient $g_t$ is forced to $0$ for the next step's comparison so that the optimizer doesn't penalize the step size twice.

3.  **Zero gradient ($g_{t-1} \cdot g_t = 0$):**
    The step size remains unchanged:
    $$\Delta_t = \Delta_{t-1}$$

Finally, the parameter is updated by taking a step in the direction of the negative gradient, using only the computed step size $\Delta_t$:
$$\theta_{t+1} = \theta_t - \text{sign}(g_t) \cdot \Delta_t$$

## Usage

```python
import aakaar.optim as optim

# Initialize Rprop
optimizer = optim.Rprop(
    model.parameters(), 
    lr=0.01, 
    etas=(0.5, 1.2), 
    step_sizes=(1e-06, 50)
)

# Standard training loop execution
optimizer.zero_grad()
loss = model(inputs).sum()
loss.backward()
optimizer.step()
```

## Parameters
- parameters: Iterable of Aakaar tensors to optimize.
- lr (float, default: 0.01): The initial step size ($\Delta_0$) for all parameters.
- etas (tuple, default: (0.5, 1.2)): A pair of scale factors $(\eta^-, \eta^+)$. $\eta^-$ is the multiplicative decrease factor (must be $< 1$), and $\eta^+$ is the multiplicative increase factor (must be $> 1$).
- step_sizes (tuple, default: (1e-6, 50)): A pair of bounds $(\Delta_{min}, \Delta_{max})$ that dictate the minimum and maximum allowed step sizes.

## When to use it
Rprop is strictly designed for full-batch optimization. Because it relies entirely on the exact sign of the gradient to determine when a minimum has been crossed, it fails catastrophically if the gradient signs are noisy. Therefore, it should never be used with mini-batch Stochastic Gradient Descent (SGD). Use Rprop only when your dataset is small enough that you can pass the entire dataset through the model in a single forward pass, or for deterministic mathematical optimization problems.