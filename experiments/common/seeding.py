"""Global reproducibility control for every ablation run.

A single seed governs Python, NumPy, and PyTorch (CPU + CUDA) so that
augmentation sampling, weight initialization, and DataLoader shuffling are
identical across runs. This is what makes an OFF/ON pair a valid controlled
comparison: the only difference between them is the solution stack, never the
random draw. See CP2_implementation_plan.md sections 3.3 and 7.
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs and (optionally) force deterministic algorithms.

    deterministic=True trades a little speed for run-to-run reproducibility.
    torch.use_deterministic_algorithms is best-effort: if a specific op has no
    deterministic implementation on this hardware, we fall back to seeded-but-
    nondeterministic rather than crashing the whole training job.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cuBLAS determinism for matmul-heavy backbones.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch without warn_only — don't let this abort training.
            pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn so each worker's augmentation RNG is seeded
    deterministically off the global torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
