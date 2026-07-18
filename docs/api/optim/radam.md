# RAdam

Aakaar provides a native implementation of RAdam (Rectified Adam), which introduces a dynamic variance rectification term to stabilize the adaptive learning rate during the early stages of training.

## What it does

In standard Adam, the adaptive learning rate has an extremely high variance during the first few steps of training because the second moment (variance) estimate is based on very few data points. This often causes the optimizer to take undesirably large steps, pushing the model into bad local optima and requiring manual "learning rate warmup" schedules to fix.

RAdam solves this by mathematically analyzing the variance of the adaptive learning rate and applying a rectification term. If the variance is deemed too high (typically in the early steps), RAdam turns off the adaptive scaling and falls back to standard momentum SGD. Once sufficient data has been seen to stabilize the variance, it seamlessly transitions into standard Adam behavior. This effectively acts as an automated, built-in learning rate warmup.

## The Math (What it means)

RAdam tracks the standard first and second moments:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t)$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(\theta_t))^2$$

It calculates the maximum possible length of the Simple Moving Average (SMA):
$$\rho_{\infty} = \frac{2}{1 - \beta_2} - 1$$

And the current length of the SMA at time step $t$:
$$\rho_t = \rho_{\infty} - \frac{2 t \beta_2^t}{1 - \beta_2^t}$$

The optimizer then branches based on whether the variance is considered tractable (typically when $\rho_t > 4$):

**If $\rho_t > 4$ (Variance is stable):**
It calculates a rectification term $r_t$:
$$r_t = \sqrt{\frac{(\rho_t - 4)(\rho_t - 2)\rho_{\infty}}{(\rho_{\infty} - 4)(\rho_{\infty} - 2)\rho_t}}$$
And updates the parameters using the rectified adaptive step:
$$\theta_{t+1} = \theta_t - \eta \left( r_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \right)$$

**If $\rho_t \leq 4$ (Variance is unstable):**
It bypasses the variance term entirely and updates using just the bias-corrected momentum:
$$\theta_{t+1} = \theta_t - \eta \hat{m}_t$$

## Usage

```python
import aakaar.optim as optim

# Initialize RAdam
optimizer = optim.RAdam(
    model.parameters(), 
    lr=0.001, 
    betas=(0.9, 0.999), 
    eps=1e-8, 
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

- lr (float, default: 0.001): The learning rate.

- betas (tuple, default: (0.9, 0.999)): Coefficients used for computing the running averages of the gradient and its square.

- eps (float, default: 1e-8): A small constant for numerical stability.

- weight_decay (float, default: 0.0): Standard L2 weight decay penalty.

## When to use it
RAdam is highly recommended when you are training models without a meticulously tuned learning rate warmup schedule. If you find your model diverging or loss exploding in the first few epochs with standard Adam or AdamW, switching to RAdam is often the easiest fix. It is particularly useful for automated training pipelines where manual hyperparameter tuning is not feasible.