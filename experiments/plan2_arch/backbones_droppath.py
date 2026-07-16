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


def assert_eval_compatible(cfg: dict, num_classes: int) -> None:
    """Fail fast if this row's modification changed the state_dict.

    `common.evaluate.evaluate_run` rebuilds the model with the PLAIN builder and
    loads our checkpoint into it. That is only valid when the modification is
    parameter-free — true for drop-path (Tier 1) and MixStyle (Tier 3), which is
    exactly why both rows can be scored by the shared, unmodified evaluator
    instead of a duplicated one. Verify it at startup, otherwise the mismatch
    would only surface at --eval-only, after an hour of training.
    """
    from experiments.common.backbones import build_backbone

    stack = cfg["stack"]
    mod = cfg.get("architecture_mod", {})
    (mod_name, mod_value), = mod.items()

    plain = build_backbone("efficientnetb0", num_classes,
                           stack["strong_head"], stack["cbam"])
    modified = build_arch_backbone(cfg, num_classes)
    kp = {k: tuple(v.shape) for k, v in plain.module.state_dict().items()}
    km = {k: tuple(v.shape) for k, v in modified.module.state_dict().items()}
    if kp != km:
        only_m = sorted(set(km) - set(kp))
        only_p = sorted(set(kp) - set(km))
        raise RuntimeError(
            f"{mod_name}={mod_value} changed the state_dict, so common.evaluate cannot "
            f"load this checkpoint.\n  extra in modified model: {only_m[:5]}\n"
            f"  missing: {only_p[:5]}")
    print(f"Eval-compatibility OK: {mod_name}={mod_value} leaves all {len(kp)} "
          f"state_dict entries unchanged (the modification is parameter-free).", flush=True)


def build_arch_backbone(cfg: dict, num_classes: int) -> BuiltModel:
    """Dispatch on cfg['architecture_mod'].

    Tier 1 (`drop_path_rate`) needs the drop-path builder. Tier 2
    (`input_resolution`) does NOT touch the architecture at all — EfficientNetB0
    is fully convolutional and ends in a global pool, so it accepts any input
    size and the head's feature dim (1280) is unchanged. The plan's "adapt the
    first layers' expected input" caveat does not apply to this backbone, so
    Tier 2 builds the plain baseline model via common.build_backbone. Tier 3
    (`mixstyle`) builds the plain baseline too, then attaches parameter-free
    MixStyle hooks after the named early MBConv stages.
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
    if "mixstyle" in mod:
        from .mixstyle import attach_mixstyle

        ms = mod["mixstyle"]
        built = build_backbone("efficientnetb0", num_classes,
                               stack["strong_head"], stack["cbam"])
        # Handles are kept alive by the model they are attached to; PyTorch holds
        # the hook in the module's _forward_hooks dict, so dropping the returned
        # handles here does not detach them.
        attach_mixstyle(built.module, layers=ms["layers"],
                        p=float(ms.get("p", 0.5)), alpha=float(ms.get("alpha", 0.1)))
        return built
    raise ValueError(
        "architecture_mod must set drop_path_rate (Tier 1), input_resolution (Tier 2), "
        "or mixstyle (Tier 3).")
