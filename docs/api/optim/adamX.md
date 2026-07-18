# Adamax

Aakaar provides a native implementation of Adamax, an extension of the Adam algorithm based on the infinity norm. While structurally similar to Adam, it replaces the L2 norm with the infinity norm for updating the second moment, which can offer increased numerical stability in certain architectures.

## What it does

In standard Adam, the second moment scales the gradients inversely proportional to the square root of the sum of past squared gradients (the L2 norm). As gradients accumulate, this value can become highly variable. 

Adamax simplifies and stabilizes this by replacing the L2 norm with the infinity norm ($\ell_\infty$). Instead of accumulating the squares, it tracks the maximum absolute value of the gradients seen so far, decayed by the $\beta_2$ parameter. This creates a strict ceiling for the gradient scaling, preventing erratic updates when gradients spike.

## The Math (What it means)

Like Adam, Adamax tracks the momentum (first moment) of the gradients:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t)$$

However, the variance tracker $v_t$ is replaced by an infinity norm tracker $u_t$. Aakaar efficiently computes this maximum using a ReLU-based identity ($\max(a, b) = b + \text{ReLU}(a - b)$) under the hood:
$$u_t = \max(\beta_2 u_{t-1}, \vert{}\nabla L(\theta_t)\vert{})$$

Unlike Adam, the $u_t$ term does not require bias correction. Only the first moment $m_t$ is corrected:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

The final parameter update is then applied:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\hat{u}_t + \epsilon} \hat{m}_t$$

## Usage

```python
import aakaar.optim as optim

# Initialize Adamax (Note: default learning rate is 0.002)
optimizer = optim.Adamax(
    model.parameters(), 
    lr=0.002, 
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

- lr (float, default: 0.002): The learning rate. Note: Adamax typically requires a slightly higher learning rate than standard Adam, hence the default of 0.002.

- betas (tuple, default: (0.9, 0.999)) : Coefficients used for computing the running averages of the gradient and its infinity norm.

- eps (float, default: 1e-8): A small constant for numerical stability.

- weight_decay (float, default: 0.0): Standard L2 weight decay penalty.

## When to use it
Adamax is particularly well-suited for models that have sparse parameter updates, such as those relying heavily on word embeddings in Natural Language Processing (NLP). Because the infinity norm places a strict upper bound on the gradient scale, it can also be a helpful fallback if you are training a model with standard Adam or AdamW and experiencing unexplained divergence or exploding gradients.
