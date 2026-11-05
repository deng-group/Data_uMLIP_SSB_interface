"""MACE fine-tuned on LiPON(100)/Li(210) interfaces and slabs

The checkpoint is in this repo at Finetuning/Mace_interface_slab_finetuned.model
"""
from pathlib import Path

CHECKPOINT = Path(__file__).resolve().parents[3] / "Finetuning" / "Mace_interface_slab_finetuned.model"


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_calculator():
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=str(CHECKPOINT), default_dtype="float32", device=_device())
