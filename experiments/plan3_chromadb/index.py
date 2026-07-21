"""Build / load the ChromaDB index over PlantVillage TRAIN embeddings (plan3 sec 4.2).

Non-negotiable hygiene (plan3 sec 2):
  * TRAIN split only. Never val, never test, never real-world. A leak assertion on
    every ingested path enforces this - storing test embeddings would put the answer
    literally in the database; storing field embeddings would invalidate every
    generalization claim in the project.
  * Class-index alignment between the Chroma label space and the CNN softmax is
    asserted (sec 3.4): a one-position mismatch adds one class's CNN probability to a
    different class's retrieval score with NO error raised - plausible-looking garbage.

ASCII-only output.
"""
import glob
import os

import numpy as np

from .embed_dino import embed_paths_cached

FORBIDDEN = ("/val/", "/test/", "real_environment", "real_world", "real-environment")
COLLECTION_NAME = "plantvillage_train_dinov2"


def list_train(train_dir):
    """Return (paths, labels, classes) for the train split in a deterministic order.

    classes = sorted(listdir) - the SAME alphabetical order torchvision ImageFolder
    uses, so the Chroma label space matches the CNN's class_to_idx (asserted later).
    Every path is checked against FORBIDDEN so a mis-set TRAIN_DIR cannot silently
    ingest a held-out split.
    """
    classes = sorted(d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d)))
    paths, labels = [], []
    for cls in classes:
        for p in sorted(glob.glob(os.path.join(train_dir, cls, "*"))):
            norm = p.replace("\\", "/")
            assert not any(f in norm for f in FORBIDDEN), f"LEAK: forbidden path ingested: {p}"
            paths.append(p)
            labels.append(cls)
    return paths, labels, classes


def build_index(train_dir, chroma_path, cache_file=None, device=None,
                model_name="dinov2_vits14", rebuild=False):
    """Build (or open) the persistent Chroma collection of train embeddings.

    Returns (collection, classes). If the collection already holds the right number of
    vectors and rebuild is False, it is reused as-is (HNSW is stochastic - see sec 2.4 -
    so avoid needless rebuilds). chroma_path on Colab should point at Drive to survive
    the runtime; otherwise the index is rebuilt each session (a few minutes).
    """
    import chromadb

    paths, labels, classes = list_train(train_dir)
    print(f"Train split: {len(paths)} images across {len(classes)} classes.", flush=True)

    client = chromadb.PersistentClient(path=chroma_path)
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    col = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    if col.count() == len(paths) and not rebuild:
        print(f"  reusing existing index ({col.count()} vectors). Pass rebuild=True to "
              f"force a fresh build.", flush=True)
        return col, classes
    if col.count() not in (0, len(paths)):
        # Partial/foreign index - start clean rather than mix vector sets.
        client.delete_collection(COLLECTION_NAME)
        col = client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    emb = embed_paths_cached(paths, cache_file, device=device, model_name=model_name)
    B = 512
    for i in range(0, len(paths), B):
        j = min(i + B, len(paths))
        col.add(ids=[f"tr_{n}" for n in range(i, j)],
                embeddings=emb[i:j].tolist(),
                metadatas=[{"label": l} for l in labels[i:j]])
    print(f"  indexed {col.count()} train embeddings into '{COLLECTION_NAME}'.", flush=True)
    return col, classes


def sanity_check(col, train_dir, cache_file=None, device=None,
                 model_name="dinov2_vits14", n=10):
    """Embed n images that ARE in the index, query them back, confirm self-retrieval
    at cosine distance ~0 (plan3 sec 3.5). Fails loudly BEFORE any fusion is built on
    top of a broken index / normalisation / metric.
    """
    from .embed_dino import embed_batch
    paths, _, _ = list_train(train_dir)
    probe = paths[:n]
    e = embed_batch(probe, device=device, model_name=model_name)
    worst = 0.0
    print("Retrieval sanity check (expect ~0.00000 self-distance):", flush=True)
    for p, v in zip(probe, e):
        r = col.query(query_embeddings=[v.tolist()], n_results=1)
        d = float(r["distances"][0][0])
        worst = max(worst, abs(d))
        print(f"  {d:.5f}  {os.path.basename(p)}", flush=True)
    if worst > 1e-3:
        raise RuntimeError(
            f"Self-retrieval distance {worst:.5f} is not ~0. Something is wrong with "
            f"L2-normalisation, the cosine metric, or the index. Fix this before "
            f"building fusion on top of it (plan3 sec 3.5).")
    print("  sanity check PASSED.", flush=True)


def assert_class_alignment(classes, cnn_class_to_idx):
    """Confirm the Chroma label order == the CNN's ImageFolder class order (sec 3.4).

    Both sort alphabetically so they SHOULD match; a one-position drift would fuse
    mismatched classes with no error, so it is asserted explicitly.
    """
    cnn_order = [c for c, _ in sorted(cnn_class_to_idx.items(), key=lambda kv: kv[1])]
    assert list(classes) == cnn_order, (
        "CLASS MISALIGNMENT between Chroma label space and CNN softmax:\n"
        f"  chroma classes: {list(classes)}\n"
        f"  cnn classes   : {cnn_order}\n"
        "Fusion would add one class's CNN prob to a different class's retrieval score.")
    print(f"  class alignment OK ({len(classes)} classes match CNN order).", flush=True)
