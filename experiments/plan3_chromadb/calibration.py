"""CNN calibration check + temperature scaling (plan3 sec 4.5).

The bgrand_real confusion matrices show CONFIDENT errors (healthy -> Late_blight,
Bacterial_spot -> Septoria). Fusing on miscalibrated softmax makes the fusion
over-trust the CNN exactly when it is confidently wrong. So: measure Expected
Calibration Error on the PlantVillage VALIDATION split, fit a single temperature T on
validation, and (if it helps) use calibrated probabilities in fusion.

Temperature scaling is a single scalar: probs = softmax(logits / T). It cannot change
the argmax, so controlled accuracy is unchanged; it only rescales confidence.

ASCII-only output.
"""
import numpy as np
import torch


def expected_calibration_error(probs, y_true, n_bins=10):
    """ECE with equal-width confidence bins. probs: (N, C). Returns (ece, bins) where
    bins is a list of (lo, hi, mean_conf, accuracy, count) for reporting/plotting."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            bins.append((float(lo), float(hi), None, None, 0))
            continue
        mean_conf = float(conf[m].mean())
        acc = float(correct[m].mean())
        ece += (cnt / n) * abs(mean_conf - acc)
        bins.append((float(lo), float(hi), mean_conf, acc, cnt))
    return float(ece), bins


def fit_temperature(logits, y_true, max_iter=100):
    """Fit T>0 minimising NLL on (val) logits via LBFGS. Returns float T.

    T>1 softens overconfident predictions; T<1 sharpens. Optimised on validation only.
    """
    z = torch.from_numpy(logits.astype("float32"))
    y = torch.from_numpy(np.asarray(y_true)).long()
    log_t = torch.zeros(1, requires_grad=True)   # optimise log T so T stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def report_calibration(logits, probs, y_true, n_bins=10):
    """Print an ECE reliability summary (raw), fit T, print calibrated ECE. Returns
    (T, ece_raw, ece_cal). The figure is a legitimate contribution on its own - it
    explains WHY fusion does or does not help."""
    ece_raw, _ = expected_calibration_error(probs, y_true, n_bins)
    T = fit_temperature(logits, y_true)
    z = torch.from_numpy(logits.astype("float32")) / T
    probs_cal = torch.softmax(z, dim=1).numpy()
    ece_cal, _ = expected_calibration_error(probs_cal, y_true, n_bins)
    print(f"  CNN calibration on validation: ECE raw {ece_raw:.4f} -> "
          f"T={T:.3f} -> ECE calibrated {ece_cal:.4f}", flush=True)
    return T, ece_raw, ece_cal
