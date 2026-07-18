# NAdam

Aakaar provides a native implementation of NAdam (Nesterov-accelerated Adam), which combines Adam's adaptive learning rates with Nesterov momentum for potentially faster convergence.

## What it does

Standard Adam uses standard momentum (a moving average of past gradients). While effective, standard momentum computes the gradient at the *current* parameter position and then applies the velocity. 

Nesterov momentum is "forward-looking." It theoretically calculates the gradient at the *next* projected position. In the context of Adam, Dozat (2016) showed that this Nesterov effect can be achieved without an extra forward pass by incorporating the momentum decay schedule directly into the bias-corrected first moment. NAdam often converges slightly faster than standard Adam, especially in the later stages of training.

## The Math (What it means)

NAdam tracks the first and second moments just like standard Adam:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t)$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(\theta_t))^2$$

However, instead of a static $\beta_1$ for bias correction, NAdam uses a dynamically decaying momentum schedule $\mu_t$:
$$\mu_t = \beta_1 \left(1 - 0.5 \cdot 0.96^{t \cdot \text{decay}}\right)$$

The bias-corrected first moment $\hat{m}_t$ is then calculated using a Nesterov look-ahead step, combining the updated momentum vector $m_t$ with the raw gradient $\nabla L(\theta_t)$:
$$\hat{m}_t = \frac{\mu_{t+1} m_t}{1 - \prod_{i=1}^{t+1} \mu_i} + \frac{(1 - \mu_t) \nabla L(\theta_t)}{1 - \prod_{i=1}^t \mu_i}$$

The final update applies this Nesterov-adjusted momentum against the standard bias-corrected variance:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

## Usage

```python
import aakaar.optim as optim

# Initialize NAdam
optimizer = optim.NAdam(
    model.parameters(), 
    lr=0.002, 
    betas=(0.9, 0.999), 
    eps=1e-8, 
    weight_decay=0.0,
    momentum_decay=0.004
)

# Standard training loop execution
optimizer.zero_grad()
loss = model(inputs).sum()
loss.backward()
optimizer.step()
```

## Parameters
- parameters: Iterable of Aakaar tensors to optimize.

- lr (float, default: 0.002): The learning rate. Like Adamax, NAdam often benefits from a slightly higher base learning rate than standard Adam.

- betas (tuple, default: (0.9, 0.999)): Coefficients used for computing the running averages of the gradient and its square.

- eps (float, default: 1e-8): A small constant for numerical stability.

- weight_decay (float, default: 0.0): Standard L2 weight decay penalty.

- momentum_decay (float, default: 0.004): The schedule multiplier for the Nesterov momentum decay factor.

## When to use it
NAdam is highly effective for tasks where standard Adam seems to plateau prematurely. The Nesterov look-ahead mechanism helps the optimizer navigate sharp curvatures in the loss landscape more gracefully. It is a strong drop-in replacement for Adam when training complex image classification models or deep autoencoders.