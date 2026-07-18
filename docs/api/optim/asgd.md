# ASGD

Aakaar provides a native implementation of ASGD (Averaged Stochastic Gradient Descent), which incorporates Polyak-Ruppert Averaging into the standard SGD algorithm to improve generalization and stability.

## What it does

Standard Stochastic Gradient Descent (SGD) can be highly noisy. As it approaches a local or global minimum, the noise from individual mini-batches often causes the parameters to heavily oscillate around the minimum rather than converging exactly on it.

ASGD addresses this by maintaining a running average of the network's weights over time. During training, the optimizer updates the parameters just like standard SGD. However, it simultaneously computes an ongoing average of these parameters. In many implementations, it is this averaged set of weights that ultimately provides better generalization on unseen data, effectively smoothing out the noisy trajectory of standard SGD.

## The Math (What it means)

ASGD performs a standard gradient descent step on the parameters $\theta$ at time $t$, utilizing a decaying learning rate $\eta_t$:
$$\eta_t = \frac{\eta_0}{(1 + \lambda \eta_0 t)^\alpha}$$
$$\theta_t = \theta_{t-1} - \eta_t \nabla L(\theta_{t-1})$$

Where $\eta_0$ is the initial learning rate, $\lambda$ is a decay term, and $\alpha$ determines the power of the decay schedule.

After the standard update, ASGD updates the averaged parameters $\bar{\theta}$. This averaging typically begins only after a specified number of steps, $t_0$. For $t > t_0$, the running average is mathematically defined as:
$$\bar{\theta}_t = \frac{1}{t - t_0 + 1} \sum_{i=t_0}^t \theta_i$$

To compute this efficiently without storing historical weights in memory, Aakaar calculates the average recursively:
$$\mu = \frac{1}{t - t_0 + 1}$$
$$\bar{\theta}_t = (1 - \mu) \bar{\theta}_{t-1} + \mu \theta_t$$

## Usage

```python
import aakaar.optim as optim

# Initialize ASGD
optimizer = optim.ASGD(
    model.parameters(), 
    lr=0.01, 
    lambd=0.0001, 
    alpha=0.75, 
    t0=1e6, 
    weight_decay=0.0
)

# Standard training loop execution
optimizer.zero_grad()
loss = model(inputs).sum()
loss.backward()
optimizer.step()
```

## Parameters
- parameters: Iterable of Aakaar tensors to optimize.

- lr (float, default: 0.01): The initial learning rate.

- lambd (float, default: 1e-4): The decay term used in the learning rate schedule.

- alpha (float, default: 0.75): The power for the learning rate decay schedule.

- t0 (float, default: 1e6): The point (number of steps) at which to start averaging. It is common to set this high so averaging only begins in the late stages of training.

- weight_decay (float, default: 0.0): Standard L2 weight decay penalty.

## When to use it
ASGD is highly beneficial when you are fine-tuning a model or in the very late stages of training where standard SGD or Adam begins to oscillate or bounce around the loss minimum. It is also particularly effective for convex optimization problems and large-scale linear models. By averaging the weights, ASGD can provide a much smoother and more robust final model than returning the weights of the absolute last training step.