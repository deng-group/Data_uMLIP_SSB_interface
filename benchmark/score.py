"""
epsilon_interface, the normalised error of a model over the interface properties

for each property and system: (model - DFT) / sigma, with sigma the std of the DFT values of
that property in the system. rmse of these per system, then the mean over the systems (each
system counts the same). epsilon_interface = geometric mean of the property terms.

properties: energy and force of the rattled structures in reference/rattled_dft.csv (energy
error in meV/atom, force error |F_model - F_DFT| per atom in meV/A), plus one per column of
reference/interfaces_dft.csv except system, interface, label (wadh, eform, sigmaf in J/m^2)

to add a property add its DFT column to reference/interfaces_dft.csv and a function to
INTERFACE_PROPERTIES in run_benchmark.py with the column name as key.
each system needs at least 2 DFT values per property, otherwise sigma is undefined.
interfaces a model can't build are left out (run_benchmark.py lists them).

rescore a model from its saved results:

    python benchmark/score.py <model_key>
"""

"""
epsilon_interface: the normalised error of a model across the interface properties.

For each property and each system, the model's errors against DFT are divided by
sigma, the standard deviation of the DFT values of that property within the
system. The RMSE of these normalised errors is taken per system, then averaged
over the systems with equal weight. epsilon_interface is the geometric mean of
the property terms.

Properties
    energy, force   built in: computed on the rattled structures listed in
                    reference/rattled_dft.csv (meV/atom, and |F_model - F_DFT|
                    per atom in meV/A)
    everything else one property per column of reference/interfaces_dft.csv,
                    apart from system, interface and label. With the reference
                    as distributed: wadh, eform and sigmaf (J/m^2).

To add a property, add a column of DFT values to reference/interfaces_dft.csv
and a function of the same name to INTERFACE_PROPERTIES in run_benchmark.py.

Every sigma comes from DFT alone, so a model's score never depends on which
other models have been run. Each system needs at least two DFT values per
property for its sigma to be defined.

Interfaces a model could not build are left out of the interface properties,
and listed by run_benchmark.py.

Re-score a model from its saved results without recomputing anything:

    python benchmark/score.py <model_key>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REF = HERE / "reference"
RESULTS = HERE / "results"

ID_COLUMNS = {"system", "interface", "label"}


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def load_reference():
    """DFT reference, the property list, and sigma for every system and property."""
    systems = pd.read_csv(REF / "systems.csv")
    interface = pd.read_csv(REF / "interfaces_dft.csv")
    rattled = pd.read_csv(REF / "rattled_dft.csv")
    forces = pd.read_csv(REF / "rattled_forces_dft.csv")
    properties = [c for c in interface.columns if c not in ID_COLUMNS]

    sigma = pd.DataFrame({
        "energy": (rattled.dft_E0_eV / rattled.n_atoms * 1000).groupby(rattled.system).std(),
        "force": forces.groupby("system").f_total.std() * 1000,
        **{p: interface.groupby("system")[p].std() for p in properties},
    }).reindex(systems.system)
    undefined = sigma[sigma.isna().any(axis=1)]
    if not undefined.empty:
        raise SystemExit("sigma is undefined (fewer than two DFT values) for:\n" + undefined.to_string())

    return {"systems": systems, "interface": interface, "rattled": rattled, "forces": forces,
            "properties": properties, "terms": ["energy", "force"] + properties, "sigma": sigma}


def score_model(energy_err, force_err, interface_vals, ref):
    """Every term and epsilon_interface for one model.

    energy_err      DataFrame: system, interface, error_meV_per_atom
    force_err       DataFrame: system, error_meV_per_A   (one row per atom)
    interface_vals  DataFrame: system, interface, and one column per property
    """
    sig = ref["sigma"]
    t = {}

    n = energy_err.error_meV_per_atom / energy_err.system.map(sig.energy)
    t["energy"] = n.groupby(energy_err.system).apply(rmse).mean()

    n = force_err.error_meV_per_A / force_err.system.map(sig.force)
    t["force"] = n.groupby(force_err.system).apply(rmse).mean()

    m = interface_vals.merge(ref["interface"], on=["system", "interface"], suffixes=("", "_dft"))
    for p in ref["properties"]:
        n = (m[p] - m[f"{p}_dft"]) / m.system.map(sig[p])
        t[p] = n.groupby(m.system).apply(rmse).mean()

    t["eps_interface"] = float(np.exp(np.mean(np.log(np.clip([t[p] for p in ref["terms"]], 1e-6, None)))))
    t["n_interfaces"] = len(m)
    return t


def saved_results(model_key, ref):
    """Score a model from the files run_benchmark.py saved in results/<model_key>/."""
    d = RESULTS / model_key
    return score_model(pd.read_csv(d / "rattled_energies.csv"),
                       pd.read_csv(d / "rattled_force_errors.csv"),
                       pd.read_csv(d / "interface_properties.csv"), ref)


def report(model_key, t, ref):
    """Print one model's terms and epsilon_interface."""
    print(f"\n{model_key}")
    for p in ref["terms"]:
        print(f"  {p:<14s}{t[p]:8.3f}")
    print(f"  interfaces used: {t['n_interfaces']} of {len(ref['interface'])}")
    print(f"  epsilon_interface = {t['eps_interface']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python benchmark/score.py <model_key>")
    ref = load_reference()
    report(sys.argv[1], saved_results(sys.argv[1], ref), ref)


