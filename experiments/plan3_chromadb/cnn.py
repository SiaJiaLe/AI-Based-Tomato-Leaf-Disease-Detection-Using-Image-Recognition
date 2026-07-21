"""CNN side of the fusion: load the frozen bgrand_real checkpoint and produce, for a
folder, per-image SOFTMAX vectors in the SAME order as the file paths (so a DINO
embedding and a CNN probability for the same image align by index).

Reuses experiments.common.evaluate._load_model and the shared eval transform, so the
CNN half of plan3 sees each image EXACTLY as the published bgrand_real eval did - no
preprocessing drift (the hazard CLAUDE.md warns about). Nothing is trained here.

Softmax supports an optional temperature T (calibration.py fits it on validation):
probs = softmax(logits / T). T=1.0 is the raw model.

ASCII-only output.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.common.data import AlbumentationsImageFolder, build_eval_transform
from experiments.common.evaluate import _load_model


def load_cnn(ckpt_dir, device):
    """Return (model, cfg, class_to_idx) for the bgrand_real checkpoint dir."""
    return _load_model(ckpt_dir, device)


@torch.no_grad()
def cnn_forward_folder(model, folder, cnn_class_to_idx, device,
                       batch_size=32, temperature=1.0):
    """Run the CNN over `folder` and return aligned per-image outputs.

    Returns dict:
      paths       list[str]         file path per image, in loader order
      logits      (N, C) float32    raw logits
      probs       (N, C) float32    softmax(logits / temperature)
      yt          (N,) int          ground truth in the CNN's TRAINING label space,
                                    remapped from the folder's own ordering BY NAME
                                    (identical to common.evaluate:117)
      class_names list[str]         training classes in index order
    """
    idx_to_class = {v: k for k, v in cnn_class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    eval_tf = build_eval_transform(224)
    ds = AlbumentationsImageFolder(folder, transform=eval_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    folder_idx_to_class = {v: k for k, v in ds.class_to_idx.items()}

    paths = [p for p, _ in ds.samples]
    all_logits, yt_local = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        all_logits.append(model(inputs).cpu().numpy().astype("float32"))
        yt_local.extend(labels.tolist())
    logits = np.concatenate(all_logits, 0) if all_logits else np.zeros((0, len(class_names)), "float32")

    # Remap folder label indices into the training label space by NAME so a different
    # on-disk class ordering cannot silently scramble the labels.
    yt = np.array([cnn_class_to_idx[folder_idx_to_class[i]] for i in yt_local])

    t = float(temperature) if temperature else 1.0
    z = torch.from_numpy(logits) / t
    probs = torch.softmax(z, dim=1).numpy().astype("float32")

    return {"paths": paths, "logits": logits, "probs": probs,
            "yt": yt, "class_names": class_names}
