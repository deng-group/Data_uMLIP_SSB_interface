"""UMA-s-1.1 (OMat task)

Download uma-s-1p1.pt and put it in CHECKPOINT_DIR (current dir if not set).
"""
import os

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", ".")


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_calculator():
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    predictor = load_predict_unit(os.path.join(CHECKPOINT_DIR, "uma-s-1p1.pt"), device=_device(),
                                  inference_settings="default", workers=1)
    return FAIRChemCalculator(predictor, task_name="omat")
