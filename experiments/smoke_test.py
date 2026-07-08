"""Fast structural check for all six backbones, both OFF and ON.

Run this on the HPC BEFORE the full 12-run job to catch any per-architecture
wiring bug (CBAM insertion, head swap, Stage-B unfreeze, forward shape) in
seconds instead of discovering it hours into training:

    python -m experiments.smoke_test

It builds each backbone, runs a dummy forward, configures both training
stages, and takes one optimizer step — no dataset required.
"""
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.common.backbones import build_backbone

BACKBONES = ["resnet34", "resnet50", "vgg16", "alexnet", "mobilenetv2", "efficientnetb0"]


def check(name, on: bool, device):
    built = build_backbone(name, num_classes=10, strong_head=on, cbam=on)
    built.module.to(device)
    built.warm_up(device)

    # Stage A: head only.
    head_params = built.freeze_to_head()
    optim.Adam(head_params, lr=1e-3)

    # Stage B: deep unfreeze + one optimizer step.
    param_dicts = built.configure_stage_b(two_group=on)
    optimizer = optim.Adam(param_dicts)
    built.module.train()
    x = torch.zeros(2, 3, 224, 224, device=device)
    y = torch.zeros(2, dtype=torch.long, device=device)
    out = built.module(x)
    assert out.shape == (2, 10), f"{name}: bad output shape {tuple(out.shape)}"
    loss = nn.CrossEntropyLoss()(out, y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    trainable = sum(p.numel() for p in built.module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in built.module.parameters())
    print(f"OK  {name:15} stack={'ON ' if on else 'OFF'}  out={tuple(out.shape)}  "
          f"groups={len(param_dicts)}  trainable={trainable:,}/{total:,}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    failures = []
    for name in BACKBONES:
        for on in (False, True):
            try:
                check(name, on, device)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                failures.append(f"{name}/{'on' if on else 'off'}: {exc}")
    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("All 12 backbone configurations built, forwarded, and stepped cleanly.")


if __name__ == "__main__":
    main()
