"""Orb-v3 conservative (inf, OMat)

The checkpoint is downloaded automatically.
"""


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_calculator():
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator
    orbff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(device=_device(),
                                                                   precision="float32-high")
    return ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=_device())
