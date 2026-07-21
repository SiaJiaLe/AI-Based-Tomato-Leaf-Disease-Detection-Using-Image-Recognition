"""Retrieval scoring, fusion, and abstention (plan3 sec 4.3 / 4.4).

Chroma returns DISTANCES; to fuse with softmax we need a probability-like vector over
classes. Similarity-weighted voting over the k nearest neighbours does that.

Verified design point (do not skip, sec 4.3): the normalised score vector SATURATES
(max stays >=0.89 across temp in {0.02,0.1,0.3,1.0}) whether the nearest neighbour is
confidently close or the votes are split among far neighbours. So the normalised score
CANNOT serve as a novelty/abstention signal. Abstention uses the RAW nearest distance,
which chroma_scores returns separately.

Neighbours are queried ONCE per image at k_max and reused for every (k, temp) in the
validation grid - slicing the sorted neighbour list for smaller k - so the grid search
does not re-hit the index thousands of times.

ASCII-only output.
"""
import numpy as np


def query_neighbors(col, embs, k_max=20, chunk=256):
    """Query Chroma for the k_max nearest neighbours of each embedding.

    Returns a list (len N) of (dists float32[<=k_max], labels list[str]), sorted by
    ascending distance as Chroma returns them. Batched to keep the number of index
    calls small.
    """
    embs = np.asarray(embs, dtype="float32")
    out = []
    for i in range(0, len(embs), chunk):
        block = embs[i:i + chunk]
        r = col.query(query_embeddings=block.tolist(), n_results=k_max)
        for dists, metas in zip(r["distances"], r["metadatas"]):
            out.append((np.asarray(dists, dtype="float32"),
                        [m["label"] for m in metas]))
    return out


def chroma_scores(dists, labs, classes_idx, n_classes, k=10, temp=0.3):
    """Similarity-weighted class scores from one image's neighbour list.

    dists, labs come from query_neighbors (already ascending). Uses the first k.
    Returns (scores[n_classes] summing to 1, nearest_distance).
    """
    d = dists[:k]
    l = labs[:k]
    sims = 1.0 - d                                   # cosine similarity
    w = np.exp(sims / temp)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    s = np.zeros(n_classes, dtype="float32")
    for wi, lab in zip(w, l):
        s[classes_idx[lab]] += wi
    return s, float(d.min())


def retrieval_matrix(neighbors, classes_idx, n_classes, k=10, temp=0.3):
    """Vectorise chroma_scores over a whole split. Returns (S (N,n_classes) each row
    sums to 1, near_dists (N,))."""
    S = np.zeros((len(neighbors), n_classes), dtype="float32")
    near = np.zeros(len(neighbors), dtype="float32")
    for i, (dists, labs) in enumerate(neighbors):
        S[i], near[i] = chroma_scores(dists, labs, classes_idx, n_classes, k, temp)
    return S, near


def fuse(p_cnn, s_chroma, w=0.6):
    """Convex combination of the CNN softmax and the retrieval score vector. Both
    inputs sum to 1 per row, so the output does too. w=1 -> pure CNN, w=0 -> pure
    retrieval (sec 5 includes both endpoints as sanity anchors)."""
    return w * p_cnn + (1.0 - w) * s_chroma


def abstain_mask(fused, near_dists, tau, d_max):
    """True where the prediction should be WITHHELD: low fused confidence (max < tau)
    OR nearest neighbour too far (near_dist > d_max). Two independent triggers (sec 4.4),
    both selected on validation."""
    low_conf = fused.max(axis=1) < tau
    far = near_dists > d_max
    return low_conf | far
