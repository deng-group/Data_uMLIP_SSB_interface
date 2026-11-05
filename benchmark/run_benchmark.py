"""
epsilon_interface for one model with an ASE calculator.

    python benchmark/run_benchmark.py benchmark/models/<family>/<model_key>.yml

the yml has the model info (see benchmark/models/template/), the .py with the same name has
get_calculator(), which returns the model's ASE calculator

steps
  1. relax the bulks (atoms + cell)
  2. build all interfaces from the relaxed bulks up to MAX_INDEX (build_interfaces.py), list them
     in results/<model_key>/interfaces/<system>/candidates.csv and keep the DFT ones with their slabs
  3. relax the kept interfaces and slabs (cell fixed)
  4. relax the oriented bulks along z only (for the interfacial energy)
  5. interface properties (INTERFACE_PROPERTIES)
  6. energy and forces of the rattled structures
  7. score (score.py), save the predictions to <date>-interface.json.gz and the scores to the yml
     under metrics.interface (matbench discovery format)

results go to benchmark/results/<model_key>/. energies are saved after each relaxation, so a
stopped run can be restarted and skips what is already done.
interfaces that can't be built from the model's bulk, or give a non-stoichiometric slab, are left
out of the interface properties and printed at the end (energy and force still use all rattled
structures)
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read, write
from ase.optimize import BFGS
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ruamel.yaml import YAML

try:
    from ase.filters import ExpCellFilter
except ImportError:                                   # older ASE
    from ase.constraints import ExpCellFilter
# ExpCellFilter is the cell filter of the paper's relaxations; recent ASE recommends
# FrechetCellFilter and warns on every use.
warnings.filterwarnings("ignore", message="Use FrechetCellFilter", category=DeprecationWarning)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_interfaces                               # noqa: E402
import score                                          # noqa: E402

# ================================ SETTINGS ===================================
FMAX = 0.05            # eV/A, every relaxation, run until reached

# Interfaces, as in the paper: every Miller index up to MAX_INDEX whose slabs are
# symmetric and non-polar (tested on slabs of MIN_SLAB_SIZE A with MIN_VACUUM A of
# vacuum), every termination; then only the interfaces in the DFT reference are kept.
MAX_INDEX = 2
MIN_SLAB_SIZE = 20
MIN_VACUUM = 10
# Bonds (maximum length in A) kept intact when the electrolyte is cleaved for the Tasker
# test, per system, so that no Miller index is accepted only through slabs that cut the
# polyanions. These are the paper's.
FILM_BONDS = {
    "beta_Li3PO4": {("P", "O"): 2},
    "gamma_Li3PO4": {("P", "O"): 2},
    "LiPON": {("P", "O"): 2, ("P", "N"): 2},
}

# Lattice matching. Widen these if an interface cannot be built with a model's
# relaxed bulk.
ZSL = dict(max_area=400, max_area_ratio_tol=0.09, max_length_tol=0.03, max_angle_tol=0.01)
# =============================================================================

REPO = HERE.parent


def build_settings(system, s=None):
    """The settings that decide which interfaces are built for one system (s: its row of
    systems.csv, which adds the slab layers)."""
    bonds = {f"{a}-{b}": v for (a, b), v in (FILM_BONDS.get(system) or {}).items()}
    layers = {} if s is None else dict(film_layers=int(s.film_layers), substrate_layers=int(s.substrate_layers))
    return dict(max_index=MAX_INDEX, min_slab_size=MIN_SLAB_SIZE, min_vacuum=MIN_VACUUM,
                film_bonds=bonds, **layers, **ZSL)
STRUCT = REPO / "Structures"
EV_PER_A2_TO_J_PER_M2 = 16.0218


# ------------------------------ interface properties --------------------------
# Each function receives the quantities of one interface (see energetics()) and
# returns the property in the units of its DFT column in reference/interfaces_dft.csv.
# To add a property: add its DFT column there and a function here with the same name.
#
#   E_int, E_film, E_sub      relaxed energies of the interface and its two slabs (eV)
#   N_film, N_sub             formula units in each slab
#   bulk_film, bulk_sub       bulk energy per formula unit (eV), relaxed atoms and cell
#   oriented_film, oriented_sub   oriented bulk energy per formula unit (eV), relaxed along z
#   S                         interface area (A^2)
#   atoms                     dict of relaxed ASE structures: interface, film_slab, substrate_slab
def work_of_adhesion(q):
    return (q["E_film"] + q["E_sub"] - q["E_int"]) / q["S"] * EV_PER_A2_TO_J_PER_M2


def strained_surface_energy(q, side):
    """gamma^strained of the film or substrate slab (eV/A^2), Eq. 3 of the paper: the strained
    slab against the oriented bulk, with the same formula units and area S as in the interface.
    It contains no strain energy, so E_f includes the full strain energy and sigma_f none."""
    return (q[f"E_{side}"] - q[f"N_{side}"] * q[f"oriented_{side}"]) / (2 * q["S"])


def formation_energy(q):
    return ((q["E_int"] - q["N_film"] * q["bulk_film"] - q["N_sub"] * q["bulk_sub"]) / q["S"]
            - (strained_surface_energy(q, "film") + strained_surface_energy(q, "sub"))) * EV_PER_A2_TO_J_PER_M2


def interfacial_energy(q):
    return ((q["E_int"] - q["N_film"] * q["oriented_film"] - q["N_sub"] * q["oriented_sub"]) / q["S"]
            - (strained_surface_energy(q, "film") + strained_surface_energy(q, "sub"))) * EV_PER_A2_TO_J_PER_M2


INTERFACE_PROPERTIES = {"wadh": work_of_adhesion, "eform": formation_energy, "sigmaf": interfacial_energy}


# ------------------------------ helpers --------------------------------------
class Cache:
    """Energies of finished calculations, saved after each one."""

    def __init__(self, path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value
        self.path.write_text(json.dumps(self.data, indent=1))

    def drop(self, prefix):
        self.data = {k: v for k, v in self.data.items() if not k.startswith(prefix)}
        self.path.write_text(json.dumps(self.data, indent=1))


def relax(atoms, calc, mode, logfile):
    """Relax and return (energy, converged).

    mode: "cell" atoms and cell; "z" atoms and the cell along z only; "atoms" cell fixed.
    """
    atoms.calc = calc
    if mode == "z":
        # a and b in the xy-plane, so that the zz strain is the only one allowed to change.
        atoms.set_cell(atoms.cell.standard_form()[0], scale_atoms=True)
        target = ExpCellFilter(atoms, mask=[0, 0, 1, 0, 0, 0])
    elif mode == "cell":
        target = ExpCellFilter(atoms)
    else:
        target = atoms
    converged = BFGS(target, logfile=str(logfile)).run(fmax=FMAX)
    return float(atoms.get_potential_energy()), bool(converged)


def read_cif(path):
    """Read a CIF with pymatgen, which handles the ICSD files that ASE's CIF reader rejects."""
    return AseAtomsAdaptor.get_atoms(Structure.from_file(path))


def area(atoms):
    return float(np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1])))


def formula_units(atoms):
    """Reduced formula and number of formula units."""
    formula, factor = AseAtomsAdaptor.get_structure(atoms).composition.get_integer_formula_and_factor()
    return formula, int(round(factor))


def miller_text(m):
    return "".join(str(x) for x in m)


def targets(ref):
    """The reference interfaces, with their Miller indices read from the label."""
    out = []
    for _, r in ref["interface"].iterrows():
        film, sub = [tuple(int(x) for x in m.split(",")) for m in re.findall(r"\(([\d, ]+)\)", r.label)]
        out.append(dict(system=r.system, interface=r.interface, label=r.label,
                        film_miller=film, substrate_miller=sub))
    return out


def load_model(yml_path):
    """Model metadata from the YAML file, and get_calculator() from the .py beside it."""
    meta = YAML().load(yml_path)
    py = yml_path.with_suffix(".py")
    spec = importlib.util.spec_from_file_location(yml_path.stem, py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return meta, module.get_calculator


# ------------------------------ 1. bulk --------------------------------------
def relax_bulks(calc, systems, out, cache):
    bulk = {}
    (out / "bulk").mkdir(parents=True, exist_ok=True)
    for name in sorted(set(systems.film_bulk) | set(systems.substrate_bulk)):
        path = out / "bulk" / f"{name}.cif"
        key = f"bulk/{name}"
        if not (cache.get(key) and path.exists()):
            atoms = read_cif(STRUCT / "unrelaxed_bulk" / f"{name}.cif")
            e, conv = relax(atoms, calc, "cell", out / "bulk" / f"{name}.log")
            write(path, atoms, format="cif")
            cache.set(key, {"energy": e, "converged": conv})
            print(f"  bulk {name}: {e:.4f} eV, converged {conv}")
        # As in the paper's workflow, the relaxed bulk is written to a CIF with ASE and
        # read back with pymatgen, and the slabs and interfaces are built from that. Always
        # continuing from the saved file also makes a resumed run build exactly the same
        # interfaces as an uninterrupted one.
        atoms = read_cif(path)
        formula, n = formula_units(atoms)
        bulk[name] = {"atoms": atoms, "per_fu": cache.get(key)["energy"] / n, "formula": formula}
    return bulk


# ------------------------------ 2-3. interfaces --------------------------------
def build_system(system, s, bulk, wanted, d, cache):
    """Build every interface of one system from the model's bulks, list them in
    candidates.csv, and write the ones in the DFT reference (labels in `wanted`) with
    their slabs. Returns {label: reason} for the reference interfaces not built.

    Anything built or relaxed before for this system is removed first, so results made
    with different settings are never mixed; build.json records the settings used."""
    if d.exists():
        shutil.rmtree(d)
    cache.drop(f"interfaces/{system}/")
    film = AseAtomsAdaptor.get_structure(bulk[s.film_bulk]["atoms"])
    substrate = AseAtomsAdaptor.get_structure(bulk[s.substrate_bulk]["atoms"])
    built, film_millers, sub_millers = build_interfaces.build_all(
        film, substrate, MAX_INDEX, int(s.film_layers), int(s.substrate_layers), ZSL,
        MIN_SLAB_SIZE, MIN_VACUUM, FILM_BONDS.get(system))
    (d / "all").mkdir(parents=True, exist_ok=True)
    rows, missing = [], {}
    for k, (name, iface) in enumerate(built.items()):
        write(d / "all" / f"{k:03d}.extxyz", AseAtomsAdaptor.get_atoms(iface))
        rows.append(dict(file=f"all/{k:03d}.extxyz", label=name, n_atoms=len(iface),
                         area_A2=iface.lattice.a * iface.lattice.b * np.sin(np.radians(iface.lattice.gamma)),
                         in_dft_reference=name in wanted))
    pd.DataFrame(rows).to_csv(d / "candidates.csv", index=False)
    for name, t in wanted.items():
        if name in built:
            (d / t["interface"]).mkdir(parents=True, exist_ok=True)
            for k, atoms in build_interfaces.split(built[name]).items():
                write(d / t["interface"] / f"{k}.extxyz", atoms)
        elif t["film_miller"] not in film_millers:
            missing[name] = (f"film Miller index {t['film_miller']} gives no symmetric, non-polar slab "
                             "of the model's bulk (Tasker test: MIN_SLAB_SIZE, MIN_VACUUM, FILM_BONDS)")
        elif t["substrate_miller"] not in sub_millers:
            missing[name] = (f"substrate Miller index {t['substrate_miller']} gives no symmetric, non-polar "
                             "slab of the model's bulk (Tasker test: MIN_SLAB_SIZE, MIN_VACUUM)")
        else:
            missing[name] = "not built: no lattice match or termination with the model's relaxed bulk (ZSL)"
    print(f"  {system}: {len(built)} interfaces up to index {MAX_INDEX}, "
          f"{len(wanted) - len(missing)} of {len(wanted)} DFT reference interfaces among them")
    (d / "build.json").write_text(json.dumps({"settings": build_settings(system, s), "missing": missing}, indent=1))
    return missing


def interfaces(calc, ref, bulk, out, cache):
    """Build every interface, keep the DFT ones, relax them and their slabs; return energies."""
    systems = ref["systems"].set_index("system")
    done, excluded = {}, []
    by_system = {}
    for t in targets(ref):
        by_system.setdefault(t["system"], {})[t["label"]] = t
    for system, wanted in by_system.items():
        d = out / "interfaces" / system
        parts = ("interface", "film_slab", "substrate_slab")
        saved = json.loads((d / "build.json").read_text()) if (d / "build.json").exists() else None
        if saved and saved["settings"] == build_settings(system, systems.loc[system]):
            missing = saved["missing"]                 # built before with these settings
        else:
            missing = build_system(system, systems.loc[system], bulk, wanted, d, cache)
        for name, t in wanted.items():
            built = {k: d / t["interface"] / f"{k}.extxyz" for k in parts}
            if not all(p.exists() for p in built.values()):
                why = missing.get(name, "not built with the model's relaxed bulk")
                excluded.append((t, why))
                print(f"  {system} {t['interface']}: {why}")
                continue
            done[(system, t["interface"])] = (t, relax_interface(calc, t, built, d / t["interface"], cache))
            print(f"  {system} {t['interface']}: done")
    return done, excluded


def relax_interface(calc, t, built, d, cache):
    """Relax one interface and its two slabs; return their relaxed structures and energies."""
    res = {}
    for k, path in built.items():
        key = f"interfaces/{t['system']}/{t['interface']}/{k}"
        if not cache.get(key):
            atoms = read(path)
            e, conv = relax(atoms, calc, "atoms", d / f"{k}.log")
            write(d / f"{k}_relaxed.extxyz", atoms)
            cache.set(key, {"energy": e, "converged": conv})
        relaxed = d / f"{k}_relaxed.extxyz"
        res[k] = {"atoms": read(relaxed if relaxed.exists() else path), "energy": cache.get(key)["energy"]}
    return res


# ------------------------------ 4. oriented bulk along z -----------------------
def oriented_bulk(calc, material, miller, out, cache):
    """Energy per formula unit of the oriented bulk cell, relaxed along z only."""
    key = f"oriented_bulk/{material}/{miller}"
    src = STRUCT / "bulk_strained_unrelaxed" / material / miller / "POSCAR"
    if not cache.get(key):
        atoms = read(src)
        (out / "oriented_bulk").mkdir(parents=True, exist_ok=True)
        e, conv = relax(atoms, calc, "z", out / "oriented_bulk" / f"{material}_{miller}.log")
        write(out / "oriented_bulk" / f"{material}_{miller}.extxyz", atoms)
        cache.set(key, {"energy": e, "converged": conv})
    return cache.get(key)["energy"] / formula_units(read(src))[1]


# ------------------------------ 5. interface properties -----------------------
def energetics(calc, ref, bulk, done, out, cache):
    systems = ref["systems"].set_index("system")
    rows, excluded = [], []
    for (system, iface), (t, res) in done.items():
        s = systems.loc[system]
        bf, bs = bulk[s.film_bulk], bulk[s.substrate_bulk]
        I, F, B = res["interface"], res["film_slab"], res["substrate_slab"]
        (f_formula, n_f), (s_formula, n_s) = formula_units(F["atoms"]), formula_units(B["atoms"])
        if f_formula != bf["formula"] or s_formula != bs["formula"]:
            excluded.append((t, "non-stoichiometric slab"))
            print(f"  {system} {iface}: non-stoichiometric slab; excluded")
            continue
        S = area(I["atoms"])
        q = {"E_int": I["energy"], "E_film": F["energy"], "E_sub": B["energy"],
             "N_film": n_f, "N_sub": n_s, "bulk_film": bf["per_fu"], "bulk_sub": bs["per_fu"],
             "oriented_film": oriented_bulk(calc, s.film_oriented_bulk, miller_text(t["film_miller"]), out, cache),
             "oriented_sub": oriented_bulk(calc, s.substrate_oriented_bulk, miller_text(t["substrate_miller"]), out, cache),
             "S": S, "atoms": {k: v["atoms"] for k, v in res.items()}}
        row = dict(system=system, interface=iface, area_A2=S,
                   gamma_film_J_m2=strained_surface_energy(q, "film") * EV_PER_A2_TO_J_PER_M2,
                   gamma_substrate_J_m2=strained_surface_energy(q, "sub") * EV_PER_A2_TO_J_PER_M2)
        row.update({p: INTERFACE_PROPERTIES[p](q) for p in ref["properties"]})
        rows.append(row)
    cols = ["system", "interface", "area_A2", "gamma_film_J_m2", "gamma_substrate_J_m2"] + ref["properties"]
    return pd.DataFrame(rows, columns=cols), excluded


# ------------------------------ 6. rattled structures --------------------------
def rattled(calc, ref):
    e_rows, f_rows = [], []
    forces_ref = ref["forces"]
    for _, r in ref["rattled"].iterrows():
        atoms = read(REPO / r.structure)
        atoms.calc = calc
        e = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        e_rows.append(dict(system=r.system, interface=r.interface, energy_eV=e,
                           error_meV_per_atom=(e - r.dft_E0_eV) / r.n_atoms * 1000))

        fd = forces_ref[(forces_ref.system == r.system) & (forces_ref.interface == r.interface)].sort_values("atom_index")
        if list(fd.element) != atoms.get_chemical_symbols():
            raise ValueError(f"{r.structure}: atoms do not match the DFT forces")
        err = np.linalg.norm(forces - fd[["fx", "fy", "fz"]].to_numpy(), axis=1) * 1000
        f_rows += [dict(system=r.system, interface=r.interface, atom_index=i, error_meV_per_A=x)
                   for i, x in zip(fd.atom_index, err)]
        print(f"  rattled {r.system} {r.interface}: done")
    return pd.DataFrame(e_rows), pd.DataFrame(f_rows)


# ------------------------------ 7. YAML -----------------------------------------
def write_predictions(out, model_key, props, e_err, f_err):
    """All predictions of the run in one file, <date>-interface.json.gz, the counterpart of the
    prediction files of Matbench Discovery's tasks. Returns its pred_file entry."""
    path = out / f"{date.today()}-interface.json.gz"
    data = {"model_key": model_key, "interface_properties": props.to_dict(orient="records"),
            "rattled_energies": e_err.to_dict(orient="records"),
            "rattled_force_errors": f_err.to_dict(orient="records")}
    with gzip.open(path, "wt") as fh:
        json.dump(data, fh)
    raw = path.read_bytes()
    return {"name": str(path.relative_to(REPO)), "size": len(raw), "md5": hashlib.md5(raw).hexdigest()}


def write_metrics(yml_path, t, ref, excluded, pred_file):
    """Record the scores under metrics.interface in the model's YAML file, as Matbench Discovery
    records its tasks (metrics.<task> with a pred_file entry; add the file's download url to
    pred_file after uploading it)."""
    yaml = YAML()
    yaml.preserve_quotes = True
    meta = yaml.load(yml_path)
    if meta.get("metrics") is None:
        meta["metrics"] = {}
    meta["metrics"]["interface"] = {
        "pred_file": pred_file,
        "eps_interface": round(t["eps_interface"], 6),
        **{p: round(float(t[p]), 6) for p in ref["terms"]},
        "n_interfaces": int(t["n_interfaces"]),
        "excluded": [f"{x['system']} {x['interface']}: {why}" for x, why in excluded],
        "date": str(date.today()),
        "settings": {"fmax": FMAX, "max_index": MAX_INDEX,
                     "min_slab_size": MIN_SLAB_SIZE, "min_vacuum": MIN_VACUUM,
                     "film_bonds": {k: build_settings(k)["film_bonds"] for k in FILM_BONDS}, **ZSL},
    }
    yaml.dump(meta, yml_path)


# ------------------------------ main -----------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_yml", type=Path, help="benchmark/models/<family>/<model_key>.yml")
    yml = ap.parse_args().model_yml.resolve()

    meta, get_calculator = load_model(yml)
    model_key = meta["model_key"]
    out = HERE / "results" / model_key
    out.mkdir(parents=True, exist_ok=True)
    cache = Cache(out / "energies.json")
    ref = score.load_reference()

    missing = [p for p in ref["properties"] if p not in INTERFACE_PROPERTIES]
    if missing:
        raise SystemExit(f"No function in INTERFACE_PROPERTIES for reference column(s) {missing}.")

    calc = get_calculator()
    print("1. Relaxing bulk structures")
    bulk = relax_bulks(calc, ref["systems"], out, cache)
    print("2-3. Building and relaxing interfaces and slabs")
    done, excluded = interfaces(calc, ref, bulk, out, cache)
    print("4-5. Oriented bulk along z; interface properties")
    props, excluded_s = energetics(calc, ref, bulk, done, out, cache)
    excluded += excluded_s
    props.to_csv(out / "interface_properties.csv", index=False)
    print("6. Rattled structures: energies and forces")
    e_err, f_err = rattled(calc, ref)
    e_err.to_csv(out / "rattled_energies.csv", index=False)
    f_err.to_csv(out / "rattled_force_errors.csv", index=False)

    print("7. Scoring")
    t = score.score_model(e_err, f_err, props, ref)
    score.report(meta.get("model_name", model_key), t, ref)
    write_metrics(yml, t, ref, excluded, write_predictions(out, model_key, props, e_err, f_err))

    if excluded:
        print(f"\nExcluded from the interface properties ({len(excluded)}):")
        for x, why in excluded:
            print(f"  {x['system']} {x['interface']}: {why}")
        print("To include them, change the setting named with each reason at the top of this file "
              "(ZSL tolerances for lattice matching; MIN_SLAB_SIZE, MIN_VACUUM or FILM_BONDS for the "
              "Tasker test), or lower FMAX so the bulk converges further, and run again. A system "
              "whose build settings changed is rebuilt and relaxed again; other finished work is kept.")
    not_converged = [k for k, v in cache.data.items() if not v["converged"]]
    if not_converged:
        print(f"\n{len(not_converged)} relaxations did not reach FMAX: "
              f"{not_converged[:5]}{' ...' if len(not_converged) > 5 else ''}")
    print(f"\nScores written to {yml.relative_to(REPO) if yml.is_relative_to(REPO) else yml} under metrics.interface")


if __name__ == "__main__":
    main()
