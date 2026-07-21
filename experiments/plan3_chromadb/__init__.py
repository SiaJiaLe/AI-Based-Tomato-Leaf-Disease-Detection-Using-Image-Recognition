"""Plan 3 - DINOv2 + ChromaDB retrieval and CNN fusion.

Additive, self-contained package. Imports from experiments.common.evaluate and
experiments.compile_results for the SHARED metric/loader code so Plan 3 rows are
directly comparable to every prior row; it modifies none of them.

See plan3_chromadb_retrieval_fusion.md for the full design and the pre-registered
decision rule (fusion is a contribution only if real-world macro-F1 beats
efficientnetb0_on_bgrand_real by more than +0.03).
"""
