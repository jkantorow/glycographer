# Glycographer

## 3D receptor-glycoligand binding affinity landscape generation through fragment-based grid docking and volumetric characterization

![glycographer_example](glycog_example_1k9i.png)

The large size and structural complexity of most glycans and an emphasis on residue-dependent binding preference observed frequently in glycan-binding targets lends itself well to fragment-based molecular docking simulation; however, standard docking workflows are often unsuitable for accurately representing glycosidic torsions. Rosetta's GlycanDock protocol provides a significantly more robust approach to modeling and sampling receptor-glycoligand interactions, but is only intended for pose refinement and not de novo pose prediction. **Here, we provide a pipeline for mapping per-glycan fragment and per-glycan residue interactions in order to quantify the full binding-energy landscape between a queried receptor-glycan interaction.** Mapping per-fragment/residue glycan interaction energies over a putative target binding site shows detailed insight into possible glycan-binding orientation, the spatial dependence and relative magnitude of glycan residue binding preference between a range of queried glycans and a queried target, and design strategies tailored towards target binding optimization. Additionally, the volumetric data generated through GlycoMap is well suited for machine learning model training for spatial binding prediciton across a range of similar target structures.

## Dependencies

Glycographer requires the following software:

- Python 3.13+
- PyMOL 3+
- PyRosetta 4+

**Note:** PyRosetta is freely available only for academic and other non-commercial use. Consult the [PyRosetta License](https://github.com/RosettaCommons/rosetta/blob/main/LICENSE.PyRosetta.md) for more information.

Only tested on Linux architecture so far.

## Installation and Environment Setup

```bash
# Clone this repository:
git clone https://github.com/jkantorow/glycographer.git
cd glycographer

# With standard pip:
python -m venv glycographer.venv
source glycographer.venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Or alternatively with conda:
conda env create -f environment.yaml
conda activate glycographer
```

## Quick Usage

Currently working on CLI integration and more streamlined usage.

```bash

# Generate glycan probes (one sequence, or a file of them) and docking grid:
python scripts/model_glycan_probes.py '<glycan-iupac-string>' -o probes/
python scripts/model_glycan_probes.py <probes.iupac> -o probes/
python scripts/build_sampling_grid.py <receptor.pdb> -o grids/
python scripts/build_sampling_grid.py <receptor.pdb> -o grids/ \
    -g <grid-id> --site '<pymol-selection>' --site-radius 12

# Concatenate receptor and ligand (or clean up a pre-assembled complex with -c):
python scripts/utils/prepare_complexes.py -rec <receptor.pdb> -lig <ligand.pdb>
python scripts/utils/prepare_complexes.py -c <complex.pdb> --lig-sel "chain X"

# Run probe sampling and generate energy voxel maps:
python scripts/run_glycandock.py <complex.pdb> -n <n-samples> -grid <grid>
python scripts/energy_map_combined.py -p <output_poses_regex_pattern>

# Output maps visualization as a PyMOL session file:
python scripts/vis/vis_glycomaps.py <map_files_regex_pattern> -rec <receptor.pdb>

```

Cluster-side steps have a SLURM passthrough of the same name under `scripts/slurm/` (e.g. `sbatch scripts/slurm/model_glycan_probes.sh probes.iupac -o probes/`).

| Step | Docs |
| --- | --- |
| Probe modeling and naming | [docs/probe_modeling.md](docs/probe_modeling.md) |
| Sampling grid generation | [docs/sampling_grids.md](docs/sampling_grids.md) |
| Mapping and visualization | [docs/visualization_workflow.md](docs/visualization_workflow.md) |
| Whole pipeline, step by step | [docs/workflow_outline.md](docs/workflow_outline.md) |