"""Tier 3: MixStyle (Zhou et al., 2021) for EfficientNetB0.

A feature map's channel-wise mean and standard deviation are, empirically, its
"style" — lighting, colour cast, background texture — while the normalized map
carries the content. MixStyle normalizes each sample by its own stats, then
re-applies stats linearly interpolated with a *shuffled* sample's stats:

    lambda ~ Beta(alpha, alpha)
    mu_mix  = lambda * mu(x)  + (1 - lambda) * mu(x[perm])
    sig_mix = lambda * sig(x) + (1 - lambda) * sig(x[perm])
    out     = sig_mix * (x - mu(x)) / sig(x) + mu_mix

The label follows the *content* sample, so a given class is seen under a
continuum of synthesized styles and style stops being a usable shortcut for it.
This is the random-shuffle variant: the paper's domain-label variant needs
multiple source domains, and we have one (PlantVillage).

Two properties make this integrate cleanly:

  * Parameter-free — no weights, no buffers, so it contributes NOTHING to the
    state_dict. The plain builder in `common.evaluate` can load a MixStyle-
    trained checkpoint, which is why this row is measured by the shared,
    unmodified evaluator rather than a duplicated one (contrast Tier 2, whose
    resolution change forced its own eval path). `assert_eval_compatible`
    verifies this at startup instead of trusting it.
  * Identity at eval — `self.training` is False, so the network measured is the
    plain baseline architecture.

The mixers are registered as CHILD MODULES of the backbone (not just closures
held by a hook) specifically so `model.train()` / `model.eval()` propagates to
them. A mixer outside the module tree would keep `training=True` forever and
would silently corrupt evaluation.

Gradients: mu/sig are detached before mixing, per the reference implementation —
MixStyle perturbs the forward statistics, it is not a path we optimize through.
"""
import random

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """Mix channel-wise feature statistics across a shuffled batch.

    Args:
        p: probability of applying MixStyle to a given batch (paper default 0.5).
           With prob 1-p the batch passes through untouched, so the model still
           sees real styles.
        alpha: Beta(alpha, alpha) shape (paper default 0.1 — U-shaped, so lambda
           lands near 0 or 1 and the mixed style usually resembles one of the two
           real styles rather than a mushy average).
        eps: variance floor for numerical stability.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"MixStyle p must be in [0, 1], got {p}.")
        if alpha <= 0.0:
            raise ValueError(f"MixStyle alpha must be > 0, got {alpha}.")
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def extra_repr(self) -> str:
        return f"p={self.p}, alpha={self.alpha}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identity at eval — this is what lets the shared evaluator measure the
        # plain architecture. Also identity for B < 2 (nothing to shuffle with).
        if not self.training or random.random() > self.p or x.size(0) < 2:
            return x

        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        mu_mix = mu * lmda + mu[perm] * (1 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1 - lmda)
        return x_normed * sig_mix + mu_mix


def attach_mixstyle(model: nn.Module, layers, p: float, alpha: float):
    """Insert MixStyle after the given 1-based MBConv stages of a timm EfficientNet.

    Uses forward hooks rather than wrapping each stage in an nn.Sequential (the
    way Sequential_CBAM does). Wrapping would renumber the state_dict keys
    (blocks.1.x -> blocks.1.0.x) and force a duplicated eval path; hooks leave
    the key layout byte-identical to the baseline's, so this row is scored by the
    same evaluator as every other row.

    The mixers are ALSO registered as child modules so train()/eval() reaches
    them; being parameter-free they add no state_dict entries.

    Returns the hook handles (kept alive by the caller).
    """
    if not layers:
        raise ValueError("mixstyle.layers is empty — nothing to attach.")
    n_stages = len(model.blocks)
    handles = []
    for stage in layers:
        idx = int(stage) - 1  # configs name stages 1-based, per plan2 §2
        if not 0 <= idx < n_stages:
            raise ValueError(
                f"mixstyle.layers contains stage {stage}, but this backbone has "
                f"{n_stages} MBConv stages (valid: 1..{n_stages}).")
        mixer = MixStyle(p=p, alpha=alpha)
        model.add_module(f"mixstyle_after_stage{stage}", mixer)

        def hook(module, inputs, output, _mixer=mixer):
            return _mixer(output)

        handles.append(model.blocks[idx].register_forward_hook(hook))
    print(f"MixStyle attached after MBConv stage(s) {list(layers)} "
          f"(p={p}, alpha={alpha}); train-mode only, parameter-free.", flush=True)
    return handles
