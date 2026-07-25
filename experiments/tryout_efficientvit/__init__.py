"""EfficientViT-B0 TRY-OUT (NOT part of the CP2 research).

Standalone, additive exploration of a Vision Transformer backbone on the same
8-class tomato split. This package ONLY IMPORTS the shared study modules
(experiments/common/*) read-only — it never modifies them, adds no config to
experiments/configs/, and does not touch any existing run. Everything it writes
is namespaced under results/efficientvit_b0_{off,on} and labelled "tryout".
"""
