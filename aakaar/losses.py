"""
aakaar.losses — torch.nn-style loss modules (nn.L1Loss, nn.MSELoss, etc.),
each a plain callable class (no separate forward() method needed since
these have no learnable parameters — matches how a stateless nn.Module
would behave, just without inheriting from a Module base you may not have).

Convention: reduction='mean' (default), 'sum', or 'none' (per-element loss).
Class-index-style targets (CrossEntropyLoss, NLLLoss, MultiMarginLoss) take
ONE-HOT tensors, not integer class indices — aakaar has no gather/fancy-
indexing op yet to do logits[i, target[i]] directly. Convert integer labels
to one-hot before calling (see one_hot() helper pattern used in earlier
training scripts this session).
"""

import numpy as np
from . import softmax, log_softmax
import aakaar


def _reduce(loss_per_element, reduction):
    if reduction == 'mean':
        return loss_per_element.sum() / loss_per_element.size
    elif reduction == 'sum':
        return loss_per_element.sum()
    elif reduction == 'none':
        return loss_per_element
    else:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")


class L1Loss:
    """Mean/sum absolute error. Needs abs() — now available."""
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, pred, target):
        return _reduce((pred - target).abs(), self.reduction)


class MSELoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, pred, target):
        diff = pred - target
        return _reduce(diff * diff, self.reduction)


class NLLLoss:
    """Negative log likelihood given log-probabilities (e.g. from
    log_softmax) and ONE-HOT targets."""
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, log_probs, target_onehot):
        per_sample = -(target_onehot * log_probs).sum(dim=-1)
        return _reduce(per_sample, self.reduction)


class CrossEntropyLoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, logits, target_onehot):
        if (self.reduction == 'mean' and logits.dtype == 'float32' and
                target_onehot.dtype == 'float32' and len(logits.shape) == 2):
            return aakaar._C._cross_entropy_fused(logits, target_onehot)
        # Fallback: original unfused path for other reductions/shapes/dtypes
        return NLLLoss(reduction=self.reduction)(log_softmax(logits, dim=-1), target_onehot)


class PoissonNLLLoss:
    def __init__(self, log_input=True, full=False, eps=1e-8, reduction='mean'):
        self.log_input = log_input
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def __call__(self, pred, target):
        if self.log_input:
            per_element = pred.exp() - target * pred
        else:
            per_element = pred - target * (pred + self.eps).log()

        if self.full:
            two_pi_target = target * (2 * np.pi)
            stirling = target * (target + self.eps).log() - target + (two_pi_target + self.eps).log() * 0.5
            per_element = per_element + stirling

        return _reduce(per_element, self.reduction)


class GaussianNLLLoss:
    """Negative log likelihood of a Gaussian: 0.5*(log(var) + (pred-target)^2/var),
    matching torch's eps-clamped variance floor."""
    def __init__(self, full=False, eps=1e-6, reduction='mean'):
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def __call__(self, pred, target, var):
        var_clamped = var + self.eps  # simple floor; torch uses a max(var,eps)-
                                       # style clamp — additive eps is a close,
                                       # stated simplification since no elementwise
                                       # max(tensor, scalar) exists yet.
        diff = pred - target
        per_element = 0.5 * (var_clamped.log() + (diff * diff) / var_clamped)
        if self.full:
            per_element = per_element + 0.5 * np.log(2 * np.pi)
        return _reduce(per_element, self.reduction)


class KLDivLoss:
    """KL divergence: target * (log(target) - input), where `input` is
    expected to already be log-probabilities (matches torch's default
    log_target=False convention: input=log-probs, target=probs)."""
    def __init__(self, reduction='mean', log_target=False):
        self.reduction = reduction
        self.log_target = log_target

    def __call__(self, input_log_probs, target):
        if self.log_target:
            per_element = target.exp() * (target - input_log_probs)
        else:
            eps = 1e-8
            per_element = target * ((target + eps).log() - input_log_probs)
        if self.reduction == 'batchmean':
            return per_element.sum() / input_log_probs.shape[0]
        return _reduce(per_element, self.reduction)


class BCELoss:
    """Binary cross-entropy from probabilities (NOT logits — see
    BCEWithLogitsLoss for the numerically stable logits version)."""
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, pred_probs, target):
        eps = 1e-7
        per_element = -(target * (pred_probs + eps).log() +
                        (1 - target) * (1 - pred_probs + eps).log())
        return _reduce(per_element, self.reduction)


class BCEWithLogitsLoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, logits, target):
        max_term = logits.relu()
        per_element = max_term - logits * target + (1 + (-logits.abs()).exp()).log()
        return _reduce(per_element, self.reduction)
    

class HuberLoss:
    def __init__(self, delta=1.0, reduction='mean'):
        self.delta = delta
        self.reduction = reduction

    def __call__(self, pred, target):
        if self.reduction == 'mean' and pred.dtype == 'float32' and target.dtype == 'float32':
            return aakaar._C._huber_loss_fused(pred, target, self.delta)
        # Fallback: original multi-op path for sum/none reduction or non-float32
        diff = (pred - target).abs()
        capped = diff - (diff - self.delta).relu()
        quadratic = 0.5 * capped * capped
        per_element = quadratic + (diff - capped)
        return _reduce(per_element, self.reduction)


class SmoothL1Loss:
    def __init__(self, beta=1.0, reduction='mean'):
        self.beta = beta
        self.reduction = reduction

    def __call__(self, pred, target):
        diff = (pred - target).abs()
        capped = diff - (diff - self.beta).relu()  # min(diff, beta)
        quadratic = 0.5 * capped * capped / self.beta
        per_element = quadratic + (diff - capped)
        return _reduce(per_element, self.reduction)
    

class MarginRankingLoss:
    """loss = max(0, -y*(x1-x2) + margin), where y in {1,-1} indicates
    which input should rank higher."""
    def __init__(self, margin=0.0, reduction='mean'):
        self.margin = margin
        self.reduction = reduction

    def __call__(self, x1, x2, y):
        per_element = (-y * (x1 - x2) + self.margin).relu()
        return _reduce(per_element, self.reduction)


class HingeEmbeddingLoss:
    """loss = x if y==1, else max(0, margin-x) if y==-1."""
    def __init__(self, margin=1.0, reduction='mean'):
        self.margin = margin
        self.reduction = reduction

    def __call__(self, x, y):
        # y is a tensor of +1/-1. Select branch via a mask built from y
        # (y==1 -> mask 1, y==-1 -> mask 0), since no elementwise
        # equality-comparison-to-tensor-of-bools op exists — built via
        # (y+1)/2, exact for the +1/-1 convention, not an approximation.
        pos_mask = (y + 1) * 0.5
        neg_mask = 1 - pos_mask
        per_element = pos_mask * x + neg_mask * (self.margin - x).relu()
        return _reduce(per_element, self.reduction)


class SoftMarginLoss:
    """loss = log(1 + exp(-y*x)), y in {1,-1}. Numerically stabilized using
    the same max(x,0)+log1p(exp(-|x|)) trick as BCEWithLogitsLoss."""
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, x, y):
        z = -y * x
        per_element = z.relu() + (1 + (-z.abs()).exp()).log()
        return _reduce(per_element, self.reduction)


class MultiLabelSoftMarginLoss:
    """Multi-label version of BCEWithLogitsLoss: averages the per-class
    BCE-with-logits loss across all classes for each sample."""
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self._bce = BCEWithLogitsLoss(reduction='none')

    def __call__(self, logits, target):
        per_class = self._bce(logits, target)
        per_sample = per_class.sum(dim=-1) / logits.shape[-1]
        return _reduce(per_sample, self.reduction)


class MultiMarginLoss:
    """Multi-class hinge loss: mean_j!=y max(0, margin - x[y] + x[j])^p.
    Takes a ONE-HOT target (see module docstring). p=1 (default) or p=2."""
    def __init__(self, p=1, margin=1.0, reduction='mean'):
        if p not in (1, 2):
            raise ValueError("MultiMarginLoss only supports p=1 or p=2")
        self.p = p
        self.margin = margin
        self.reduction = reduction

    def __call__(self, logits, target_onehot):
        correct_score = (logits * target_onehot).sum(dim=-1, keepdim=True)
        margins = (self.margin - correct_score + logits).relu()
        # zero out the j == y term (torch excludes the correct class itself)
        margins = margins * (1 - target_onehot)
        if self.p == 2:
            margins = margins * margins
        per_sample = margins.sum(dim=-1) / logits.shape[-1]
        return _reduce(per_sample, self.reduction)


class CosineEmbeddingLoss:
    """loss = 1 - cos_sim(x1,x2) if y==1, else max(0, cos_sim(x1,x2)-margin) if y==-1.
    Needs sqrt() (for the norm) — now available."""
    def __init__(self, margin=0.0, reduction='mean'):
        self.margin = margin
        self.reduction = reduction

    def __call__(self, x1, x2, y):
        eps = 1e-8
        dot = (x1 * x2).sum(dim=-1)
        norm1 = (x1 * x1).sum(dim=-1).sqrt()
        norm2 = (x2 * x2).sum(dim=-1).sqrt()
        cos_sim = dot / (norm1 * norm2 + eps)

        pos_mask = (y + 1) * 0.5
        neg_mask = 1 - pos_mask
        per_element = pos_mask * (1 - cos_sim) + neg_mask * (cos_sim - self.margin).relu()
        return _reduce(per_element, self.reduction)


class TripletMarginLoss:
    """loss = max(0, d(anchor,pos) - d(anchor,neg) + margin), with d being
    Euclidean distance by default (p=2). Needs sqrt() — now available."""
    def __init__(self, margin=1.0, p=2, reduction='mean'):
        if p != 2:
            raise NotImplementedError(
                "TripletMarginLoss currently only supports p=2 (Euclidean distance). "
                "General p-norms would need an elementwise pow(x, 1/p), which "
                "doesn't exist yet — only sqrt (p=2's special case) is implemented."
            )
        self.margin = margin
        self.reduction = reduction

    def __call__(self, anchor, positive, negative):
        eps = 1e-8
        d_pos = (((anchor - positive) * (anchor - positive)).sum(dim=-1) + eps).sqrt()
        d_neg = (((anchor - negative) * (anchor - negative)).sum(dim=-1) + eps).sqrt()
        per_element = (d_pos - d_neg + self.margin).relu()
        return _reduce(per_element, self.reduction)