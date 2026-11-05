"""MACE-MH-0 (OMat head)

Download mace-mh-0.model and put it in CHECKPOINT_DIR (current dir if not set).
"""
import os

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", ".")


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_calculator():
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=os.path.join(CHECKPOINT_DIR, "mace-mh-0.model"),
                          default_dtype="float32", device=_device(), head="omat_pbe")
