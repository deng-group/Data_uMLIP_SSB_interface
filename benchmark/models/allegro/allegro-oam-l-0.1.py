r"""Allegro-OAM-L

Compile the checkpoint yourself (needs `pip install nequip nequip-allegro`) and put
mir-group__Allegro-OAM-L__0.1.nequip.pt2 in CHECKPOINT_DIR (current dir if not set):

    nequip-compile nequip.net:mir-group/Allegro-OAM-L:0.1 mir-group__Allegro-OAM-L__0.1.nequip.pt2 \
        --mode aotinductor --device cuda --target ase

Recompile it for a different GPU or torch version.
"""
import os

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", ".")


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_calculator():
    from nequip.ase import NequIPCalculator
    return NequIPCalculator.from_compiled_model(
        compile_path=os.path.join(CHECKPOINT_DIR, "mir-group__Allegro-OAM-L__0.1.nequip.pt2"),
        device=_device())
