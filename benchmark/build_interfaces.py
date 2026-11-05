"""
builds all interfaces between an electrolyte (film) and an electrode (substrate) bulk with pymatgen.
no MLIP here, run_benchmark.py calls this with the model's relaxed bulks and keeps the DFT ones.

miller indices up to max_index are kept if they give at least one symmetric, non-polar slab (tasker).
for each film/substrate miller pair and each termination the first interface from
CoherentInterfaceBuilder (ZSL matching) is kept. split() cuts out the film and substrate slabs.

name = film miller + first 2 characters of the film termination + substrate miller + first 2
characters of the substrate termination, e.g. "(1, 1, 0)Li(2, 1, 0)Li" (the label column in
reference/interfaces_dft.csv). if two terminations give the same name the last one is kept.
"""

from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
from pymatgen.io.ase import AseAtomsAdaptor


def tasker_millers(bulk, max_index, min_slab_size, min_vacuum_size, bonds=None):
    """Miller indices up to max_index with at least one symmetric, non-polar slab."""
    millers = []
    for miller in get_symmetrically_distinct_miller_indices(bulk, max_index):
        try:
            slabs = SlabGenerator(bulk, miller, min_slab_size, min_vacuum_size).get_slabs(bonds=bonds or {})
        except Exception:                             # some indices fail inside pymatgen
            continue
        for slab in slabs:
            if slab.is_symmetric() and not slab.is_polar() and slab.miller_index not in millers:
                millers.append(slab.miller_index)
    return millers


def label(film_miller, substrate_miller, termination):
    return str(film_miller) + termination[0][0:2] + str(substrate_miller) + termination[1][0:2]


def build_all(film, substrate, max_index, film_layers, substrate_layers, zsl, min_slab_size,
              min_vacuum_size, film_bonds=None):
    """Every interface of film on substrate (pymatgen Structures).

    Returns (interfaces, film_millers, substrate_millers); interfaces maps each name to its
    pymatgen Interface.
    """
    film_millers = tasker_millers(film, max_index, min_slab_size, min_vacuum_size, film_bonds)
    substrate_millers = tasker_millers(substrate, max_index, min_slab_size, min_vacuum_size)
    built = {}
    for fm in film_millers:
        for sm in substrate_millers:
            try:
                cib = CoherentInterfaceBuilder(film_structure=film, substrate_structure=substrate,
                                               film_miller=fm, substrate_miller=sm,
                                               zslgen=ZSLGenerator(**zsl))
            except Exception:                         # slicing or centring fails for some pairs
                continue
            for term in cib.terminations:
                try:
                    interfaces = list(cib.get_interfaces(termination=term, film_thickness=film_layers,
                                                         substrate_thickness=substrate_layers))
                except Exception:                     # singular matrices and other pymatgen failures
                    continue
                if interfaces:
                    built[label(fm, sm, term)] = interfaces[0]
    return built, film_millers, substrate_millers


def split(interface):
    """ASE structures of the interface and its film and substrate slabs."""
    film = interface.copy()
    film.remove_sites(interface.substrate_indices)
    substrate = interface.copy()
    substrate.remove_sites(interface.film_indices)
    return {k: AseAtomsAdaptor.get_atoms(s) for k, s in
            [("interface", interface), ("film_slab", film), ("substrate_slab", substrate)]}

