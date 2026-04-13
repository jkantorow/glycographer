# Glycographer
## 3D receptor-glycoligand binding affinity landscape generation through fragment-based grid docking and volumetric characterization
![glycographer_example](doc/img/glycog_example_1k9i.png)

The large size and structural complexity of most glycans and an emphasis on residue-dependent binding preference observed frequently in glycan-binding targets lends itself well to fragment-based molecular docking simulation; however, standard docking workflows are often unsuitable for accurately representing glycosidic torsions. Rosetta's GlycanDock protocol provides a significantly more robust approach to modeling and sampling receptor-glycoligand interactions, but is only intended for pose refinement and not de novo pose prediction. **Here, we provide a pipeline for mapping per-glycan fragment and per-glycan residue interactions in order to quantify the full binding-energy landscape between a queried receptor-glycan interaction.** Mapping per-fragment/residue glycan interaction energies over a putative target binding site shows detailed insight into possible glycan-binding orientation, the spatial dependence and relative magnitude of glycan residue binding preference between a range of queried glycans and a queried target, and design strategies tailored towards target binding optimization. Additionally, the volumetric data generated through GlycoMap is well suited for machine learning model training for spatial binding prediciton across a range of similar target structures.

## Goal

`glycographer` is the public, polished interface for the private `glycomaps` molecular modeling pipeline. This repository now uses a standard Python layout and includes onboarding docs so new users can install, initialize, and validate a working project structure quickly.

## Repository layout

```text
glycographer/
├── docs/                    # User/developer documentation
├── examples/                # Example datasets and tutorials
├── scripts/                 # Utility scripts
├── src/glycographer/        # Python package source
├── tests/                   # Automated tests
├── pyproject.toml           # Packaging and CLI entry points
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## Quick usage

```bash
glycographer init ./my-run
glycographer validate ./my-run
```

For a full walkthrough, see [`docs/quickstart.md`](docs/quickstart.md).

## Next migration step from `glycomaps`

As private modules are brought over, place them under `src/glycographer/` by pipeline stage and expose stable commands through the CLI. This preserves a clean user-facing interface while allowing internal refactors.