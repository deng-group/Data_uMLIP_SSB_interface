"""MACE-MP-0a medium (pre-trained baseline for the fine-tuned MACE models)

Download 2023-12-03-mace-128-L1_epoch-199.model and put it in CHECKPOINT_DIR (current dir if not set).
"""
import os

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", ".")


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_calculator():
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=os.path.join(CHECKPOINT_DIR, "2023-12-03-mace-128-L1_epoch-199.model"),
                          default_dtype="float32", device=_device())
