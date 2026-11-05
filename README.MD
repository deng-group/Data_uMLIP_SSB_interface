# Applicability Domain of Universal Machine Learning Interatomic Potentials at Solid-State Battery Interfaces

Code and DFT data for benchmarking universal machine learning interatomic potentials (uMLIPs) on
24 Li-metal/solid-electrolyte interfaces in six interface chemistries: Li₂O/Li, Li₂S/Li, LiCl/Li,
β-Li₃PO₄/Li, γ-Li₃PO₄/Li and LiPON/Li (four Miller-index combinations each).

Any MLIP with an ASE calculator can be benchmarked with one command. The result is the interface
error, ε_interface (lower is better), and its five normalised terms: interface energy
(E_interface), interatomic forces (F), interface formation energy (E_f), interfacial energy (σ_f)
and work of adhesion (W_adh). The repository also contains the three fine-tuned MACE models of
the paper.

Paper: *link to be added*

## Contents

```
├── benchmark/
│   ├── run_benchmark.py      Benchmarks one model: relaxation, interface construction, scoring
│   ├── build_interfaces.py   Builds the interfaces (Zur–McGill lattice matching)
│   ├── score.py              Recomputes ε_interface from saved results
│   ├── leaderboard.py        Ranks all benchmarked models
│   ├── models/               One folder per model (start from template/)
│   └── reference/            DFT reference data
├── Structures/               DFT structures and benchmark inputs
├── Finetuning/               Fine-tuned MACE models
├── environment.yml
└── LICENSE
```

## Installation

```bash
conda env create -f environment.yml -n <env-name>
conda activate <env-name>
pip install <your model's package>
```

The six uMLIPs of the paper need different package versions, so use one environment per model:

| Models | Extra package |
|---|---|
| MACE-MH-0, MACE-MP-0a, fine-tuned MACE | `pip install mace-torch==0.3.15` |
| NequIP-OAM-L, Allegro-OAM-L | `pip install nequip nequip-allegro` |
| UMA-s-1.1 | `pip install fairchem-core` |
| Orb-v3 | `pip install orb-models` |
| PET-OAM-XL | `pip install upet` |

## Benchmark a model

1. Copy `benchmark/models/template/` to `benchmark/models/<family>/` and rename both files to
   your model key (e.g. `mymodel-v1.yml`, `mymodel-v1.py`).
2. In the `.py`, make `get_calculator()` return the model's ASE calculator. In the `.yml`, set
   `model_key` to the file name.
3. Run from the repository root:

   ```bash
   python benchmark/run_benchmark.py benchmark/models/<family>/<model_key>.yml
   ```

The run prints the five terms and ε_interface and writes them to the model's `.yml`. All
predictions are saved in `benchmark/results/<model_key>/`; an interrupted run continues where it
stopped.

```bash
python benchmark/score.py <model_key>     # re-score saved results
python benchmark/leaderboard.py           # rank all benchmarked models
```

The benchmark follows the workflow and definitions in the Methods of the paper. An interface that
cannot be built from the model's relaxed bulk is left out of E_f, σ_f and W_adh and reported at
the end of the run. All settings are at the top of `run_benchmark.py`.

## DFT data

| File or folder | Contents |
|---|---|
| `benchmark/reference/interfaces_dft.csv` | W_adh, E_f and σ_f of the 24 interfaces (J/m²) |
| `benchmark/reference/rattled_dft.csv` | DFT energies of the 24 rattled interfaces (eV) |
| `benchmark/reference/rattled_forces_dft.csv` | DFT forces on every atom of these structures (eV/Å) |
| `benchmark/reference/systems.csv` | Bulk files and slab settings of the six chemistries |
| `Structures/unrelaxed_bulk/`, `Structures/dft_relaxed_bulk/` | Bulk structures before and after DFT relaxation |
| `Structures/dft_relaxed_interfaces/` | DFT-relaxed interfaces and their two slabs |
| `Structures/bulk_strained_unrelaxed/`, `Structures/strained_bulk_dft_relaxed/` | Oriented bulk cells for E_f and σ_f, before and after relaxation along *z* |
| `Structures/rattled_structures/` | The 24 rattled interfaces |

Interface folders are named by Miller indices and terminations: `100Li210Li` is the (100)
electrolyte surface on the (210) Li surface.

## Models of the paper

The pre-trained checkpoints are not included; download them from their authors into the
working directory, or into the folder set by `CHECKPOINT_DIR`.

| Model | Files in `benchmark/models/` |
|---|---|
| Allegro-OAM-L | `allegro/allegro-oam-l-0.1` |
| MACE-MH-0 (OMat head) | `mace/mace-mh-0` |
| NequIP-OAM-L | `nequip/nequip-oam-l-0.1` |
| PET-OAM-XL | `pet/pet-oam-xl-1.0.0` |
| Orb-v3 conservative (inf, OMat) | `orb/orb-v3-conservative-inf-omat` |
| UMA-s-1.1 (OMat task) | `uma/uma-s-1p1` |

## Fine-tuned MACE models

The three fine-tuned MACE-MP-0a models of the paper are in `Finetuning/` and can be benchmarked
directly:

| Model | Fine-tuned on | Files in `benchmark/models/mace-finetuned/` |
|---|---|---|
| `Mace_slab_finetuned.model` | Slabs | `mace-finetuned-slab` |
| `Mace_interface_finetuned.model` | Interfaces | `mace-finetuned-interface` |
| `Mace_interface_slab_finetuned.model` | Interfaces and slabs | `mace-finetuned-interface-slab` |

The pre-trained baseline is `mace/mace-mp-0`.

## Citation

*BibTeX to be added once the preprint is posted.*

## License

MIT License. See `LICENSE`.
