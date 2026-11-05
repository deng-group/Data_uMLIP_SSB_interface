"""Calculator for the model described in the .yml file of the same name."""


def get_calculator():
    """Return the model's ASE calculator: energies (eV), forces (eV/A) and stress."""
    # A MACE model, for example:
    #     from mace.calculators import MACECalculator
    #     return MACECalculator(model_paths="my_model.model", device="cuda", default_dtype="float64")
    raise NotImplementedError("Return your model's ASE calculator here.")
