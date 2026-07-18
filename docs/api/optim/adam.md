# Adam

Aakaar provides a native, highly optimized implementation of the standard Adam (Adaptive Moment Estimation) optimization algorithm.

## What it does

Adam computes individual adaptive learning rates for different parameters from estimates of the first and second moments of the gradients. It combines the advantages of two other extensions of stochastic gradient descent: 
*   **AdaGrad:** Works well with sparse gradients.
*   **RMSProp:** Works well in on-line and non-stationary settings.

Unlike AdamW, standard Adam implements weight decay as standard L2 regularization, meaning the weight decay penalty is added directly to the gradient before calculating the momentum and variance estimates.

## The Math (What it means)

Adam maintains two moving averages for each parameter's gradient $g_t$ at time step $t$. 

First, it calculates the biased first moment (momentum) and second moment (uncentered variance):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Because these moment vectors are initialized with zeros, they are biased towards zero, especially during the initial time steps. Adam applies a bias-correction mechanism to counteract this:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Finally, the parameters $\theta$ are updated using the learning rate $\eta$:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

## Usage

```python
import aakaar.optim as optim

# Initialize the optimizer with the model's parameters
optimizer = optim.Adam(
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

- lr (float, default: 0.001): The learning rate (step size).

- betas (tuple, default: (0.9, 0.999)) : Coefficients used for computing the running averages of the gradient and its square.

- eps (float, default: 1e-8): A microscopically small term added to the denominator to prevent division by zero and improve numerical stability.

- weight_decay (float, default: 0.0): L2 penalty applied to the gradients.

## When to use it

Standard Adam is a robust, general-purpose optimizer that performs exceptionally well out-of-the-box for most deep learning applications without requiring extensive learning rate tuning. It is particularly well-suited for problems with large datasets, high-dimensional parameter spaces, or noisy/sparse gradients.

**Note**: If your training regime relies heavily on weight decay for regularization (which is common in modern Transformer architectures), you should use AdamW instead to ensure the decay is decoupled from the adaptive learning rate.