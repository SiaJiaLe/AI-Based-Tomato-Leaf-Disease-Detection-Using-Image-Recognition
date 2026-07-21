"""DINOv2 embedding extraction (plan3 section 3.1 / 4.1).

Why DINOv2 and not the fine-tuned CNN: retrieval on the CNN's own representation
returns correlated errors (when the CNN is wrong, the embedding is already in the
wrong region). A self-supervised encoder trained on a large diverse corpus gives an
INDEPENDENT view whose failures decorrelate from the CNN's - the entire premise of
fusion. See the plan's "Why a different encoder" note.

Deterministic: model in .eval(), no dropout, no augmentation. The ONLY genuine
run-to-run variation in plan3 is Chroma's HNSW index build (index.py), not this.

ASCII-only output (compute-node C locale).
"""
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# DINOv2 uses patch size 14, so input dims must be multiples of 14. 224 = 16 patches,
# and matches the CNN pipeline's crop size. ImageNet normalization (same constants the
# rest of the project uses). No augmentation at embed time.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

_MODEL_CACHE = {}


def load_dino(model_name="dinov2_vits14", device=None):
    """Load a DINOv2 backbone once per (name, process). ViT-S/14 -> 384-dim.

    torch.hub.load fetches weights (~85 MB) on first call and needs internet. On
    Colab, redirect TORCH_HOME to Drive so it is not re-downloaded every session:
        os.environ["TORCH_HOME"] = "/content/drive/MyDrive/torch_cache"
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (model_name, str(device))
    if key not in _MODEL_CACHE:
        model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
        model = model.to(device).eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


@torch.no_grad()
def embed_batch(paths, model=None, device=None, batch_size=32, model_name="dinov2_vits14"):
    """Return L2-normalised (N, D) float32 embeddings for the given image paths,
    in the SAME order as `paths` (fusion aligns CNN prob and DINO embedding by index).

    L2-normalisation is required: with unit-norm vectors Chroma's cosine distance is
    well-behaved and comparable across queries, which the abstention logic depends on.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model or load_dino(model_name, device)
    out = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        imgs = []
        for p in chunk:
            if isinstance(p, str):
                imgs.append(_TF(Image.open(p).convert("RGB")))
            else:
                imgs.append(_TF(p.convert("RGB")))
        x = torch.stack(imgs).to(device)
        f = model(x)                                      # (B, D) CLS feature
        f = torch.nn.functional.normalize(f, dim=1)       # unit norm -> cosine space
        out.append(f.cpu().numpy().astype("float32"))
    return np.concatenate(out, 0) if out else np.zeros((0, 384), dtype="float32")


def embed_paths_cached(paths, cache_file, model=None, device=None,
                       batch_size=32, model_name="dinov2_vits14"):
    """embed_batch with an on-disk cache keyed by the exact path list.

    Extracting ~4,750 train embeddings is the slow step; caching to Drive makes
    later Colab sessions instant. The cache is trusted ONLY if it stores the exact
    same ordered path list - any change (different split, added images) recomputes.
    """
    if cache_file and os.path.isfile(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        cached_paths = list(data["paths"])
        if cached_paths == list(paths):
            print(f"  loaded {len(cached_paths)} cached embeddings <- "
                  f"{os.path.basename(cache_file)}", flush=True)
            return data["emb"].astype("float32")
        print("  cache path-list mismatch; recomputing embeddings.", flush=True)

    emb = embed_batch(paths, model=model, device=device,
                      batch_size=batch_size, model_name=model_name)
    if cache_file:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        np.savez(cache_file, emb=emb, paths=np.array(list(paths), dtype=object))
        print(f"  cached {len(paths)} embeddings -> {os.path.basename(cache_file)}",
              flush=True)
    return emb
