"""PET-OAM-XL

Download pet-oam-xl-v1.0.0.ckpt and put it in CHECKPOINT_DIR (current dir if not set).
"""
import os

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", ".")


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_calculator():
    from upet.calculator import UPETCalculator
    return UPETCalculator(checkpoint_path=os.path.join(CHECKPOINT_DIR, "pet-oam-xl-v1.0.0.ckpt"),
                          device=_device())
