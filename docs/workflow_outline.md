# Standard Workflow for Glycographer Probe Docking and Mapping
- Generate a uniform grid of sampling coordinates within a shell superimposed onto the surface of a putative binding site (`scripts/build_sampling_grid.py`, see [sampling_grids.md](sampling_grids.md)):
    - Inputs:
        - Prepared receptor structure in pdb format <receptor.pdb>
        - Optionally, a PyMOL selection of the residues to sample around (`--site`), or an explicit box (`--box`); with neither, the whole surface is sampled
    - Outputs:
        - Sampling coordinate points as pseudoatoms in pdb format <receptor_grid_<grid_id>.pdb>, plus a .json sidecar of the build parameters (and .xyz with `--write-xyz`)
- Execute an extended-sampling grid box docking simulation between a query receptor and glycan candidates:
    - Inputs:
        - Receptor structure in pdb format <receptor.pdb>
        - Sampling coordinate points in pdb format <sampling_grid.pdb>
        - Any queried glycan sequence in IUPAC format <glycoligand_queries.iupac>
    - Outputs:
        - Ensembles of scored GlycanDock simulations between the receptor and each glycan / glycan fragment </docking/frag1/rec_frag1_complex_*.pdb, etc.>
        - Rosetta score files summarizing the output statistics <scores.sc>
- Construct volumetric maps for binding hotspot identification:
    - Inputs:
        - Scored docked complex poses from Rosetta <rec_frag_complex_*.pdb>
    - Outputs:
        - Volumetric maps of interaction energy, residue contacts, atomic number density, etc. in dx format <rec_frag_energymap.dx>
- Visualize maps with PyMOL or VMD:
    - Inputs:
        - Receptor structure in pdb format <receptor.pdb>
        - Any known crystal ligand pose in pdb format <crystal_ligand.pdb>
        - Volumetric maps in dx format <rec_frag_energymap.dx>
    - Outputs:
        - Visualization session files <rec_frag_vis.pse> <rec_frag_vis.vmd>
        - Isosurface coordinates <rec_frag_energymap_isolevel_n.xyz>
        - Rendered images <rec_frag_isomap_vis.png>

# Standard Workflow for GlycanDock Receptor-Glycoligand Docking
(To do: fill this out properly)