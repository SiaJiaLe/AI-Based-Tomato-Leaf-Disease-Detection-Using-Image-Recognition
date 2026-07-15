"""Tier 1 backbone: EfficientNetB0 + stochastic depth (drop-path).

This is a byte-for-byte copy of `common.backbones._build_efficientnetb0` with
ONE difference: `drop_path_rate` is passed to `timm.create_model`. Everything
else — the head, the CBAM insertion points, and the Stage-B group layout — is
reused from `common` so the only thing that differs from the `efficientnetb0_on`
baseline is stochastic depth.

timm distributes the rate linearly across MBConv blocks (block i gets
`drop_path_rate * i / num_blocks`), so `drop_path_rate` is the *maximum* rate,
applied at the deepest block. Drop-path is parameter-free and is identity in
eval() mode, which is why `common.evaluate.evaluate_run` can load this
checkpoint with the plain builder and still measure the exact same network.
"""
from experiments.common.backbones import BuiltModel
from experiments.common.cbam import Sequential_CBAM
from experiments.common.heads import build_head


def build_efficientnetb0_droppath(num_classes: int, strong_head: bool, cbam: bool,
                                  drop_path_rate: float) -> BuiltModel:
    import timm
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,  # <-- the one variable this row tests
    )
    in_features = model.num_features  # 1280
    model.classifier = build_head(in_features, num_classes, strong_head)
    if cbam:
        model.blocks[-1] = Sequential_CBAM(model.blocks[-1])
        model.blocks[-2] = Sequential_CBAM(model.blocks[-2])
    groups = [[model.blocks[i]] for i in range(len(model.blocks) - 2)]
    groups.append([model.blocks[-2]])
    groups.append([model.blocks[-1], model.conv_head, model.bn2])
    return BuiltModel(module=model, head=model.classifier, groups=groups)


def assert_eval_compatible(num_classes: int, strong_head: bool, cbam: bool,
                           drop_path_rate: float) -> None:
    """Fail fast if drop-path changed the state_dict.

    `common.evaluate.evaluate_run` rebuilds the model with the PLAIN builder
    (no drop_path_rate) and loads our checkpoint into it. That is only valid
    because drop-path is parameter-free. Verify it at startup — otherwise the
    mismatch would only surface at --eval-only, after an hour of training.
    """
    from experiments.common.backbones import build_backbone

    plain = build_backbone("efficientnetb0", num_classes, strong_head, cbam)
    dropped = build_efficientnetb0_droppath(num_classes, strong_head, cbam, drop_path_rate)
    kp = {k: tuple(v.shape) for k, v in plain.module.state_dict().items()}
    kd = {k: tuple(v.shape) for k, v in dropped.module.state_dict().items()}
    if kp != kd:
        only_d = sorted(set(kd) - set(kp))
        only_p = sorted(set(kp) - set(kd))
        raise RuntimeError(
            "drop_path_rate changed the state_dict, so common.evaluate cannot load this "
            f"checkpoint.\n  extra in drop-path model: {only_d[:5]}\n  missing: {only_p[:5]}")
    print(f"Eval-compatibility OK: drop_path_rate={drop_path_rate} leaves all "
          f"{len(kp)} state_dict entries unchanged (drop-path is parameter-free).", flush=True)


def build_arch_backbone(cfg: dict, num_classes: int) -> BuiltModel:
    """Dispatch on cfg['architecture_mod'].

    Tier 1 (`drop_path_rate`) needs the drop-path builder. Tier 2
    (`input_resolution`) does NOT touch the architecture at all — EfficientNetB0
    is fully convolutional and ends in a global pool, so it accepts any input
    size and the head's feature dim (1280) is unchanged. The plan's "adapt the
    first layers' expected input" caveat does not apply to this backbone, so
    Tier 2 builds the plain baseline model via common.build_backbone.
    """
    from experiments.common.backbones import build_backbone

    stack = cfg["stack"]
    mod = cfg.get("architecture_mod", {})
    if cfg["backbone"] != "efficientnetb0":
        raise ValueError("Plan 2 rows are EfficientNetB0-only.")

    if "drop_path_rate" in mod:
        return build_efficientnetb0_droppath(
            num_classes=num_classes,
            strong_head=stack["strong_head"],
            cbam=stack["cbam"],
            drop_path_rate=float(mod["drop_path_rate"]),
        )
    if "input_resolution" in mod:
        return build_backbone("efficientnetb0", num_classes,
                              stack["strong_head"], stack["cbam"])
    raise ValueError(
        "architecture_mod must set drop_path_rate (Tier 1) or input_resolution (Tier 2).")
