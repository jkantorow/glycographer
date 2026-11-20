# Performing a standard Glycographer probe docking protocol between a queried receptor and specified glycan fragment probes

Here, we walk through the example of setting up and running a glycographer global probe docking simulation for the receptor of PDB 1PMH.

I have glycographer installed in my home directory, but script paths may vary depending on where you've installed glycographer.

I would like to later split this into a broader workflow (stored here) and a notebook example showing specific usage. Also, this information is subject to change as I try to design a simpler way of executing all of these steps instead of having to do them one by one.

## Part 1 - Sampling Grid Generation

A glycographer sampling grid is just a simple point cloud of xyz coordinates arranged in a 1x1x1 Angstrom (by default) lattice. It is natively output as a pdb file containing pseudoatoms with xyz coordinates corresponding to those of the grid. A global sampling grid can be generated and output as a pdb file via the included python script:

Note: We may want to check the following arguments to make sure the output would be in this form if this command were specified.
```bash
python ~/glycographer/scripts/build_sampling_grid.py 1pmh_rec.pdb --out ./1pmh_grid_global.pdb
# -> ./1pmh_grid_global.pdb
# -> ./1pmh_grid_global.json
```

## Part 2 - Glycan Fragment Probe Model Generation from Sequence

Rosetta provides the functionality to generate and minimize glycan sequences from an input simplified IUPAC string alone (see [Working with Glycans - Building Pure Glycans](https://docs.rosettacommons.org/docs/latest/application_documentation/carbohydrates/WorkingWithGlycans#building-pure-glycans)). Glycographer leverages this functionality to accept an input simplified glycan IUPAC string or text file containing multiple IUPAC strings and output a minimized glycan pose for each. We first create a simple text file named '1pmh_probes.iupac' containing the iupac string of each desired probe. The text I used can be seen below:

```text
# Probes built from the native 1PMH glycoligand: beta-1,4-mannohexaose
# (PDB 1PMH chain A, MAN1 <- BMA2 <- BMA3 <- BMA4 <- BMA5 <- BMA6, with the
# alpha anomer at the reducing end, exactly as the crystal models it).
#
# A fragment series Man1..Man6. Every internal linkage is the same
# b-D-Manp-(1->4), so the series is nested: each probe is the previous one
# extended by one residue at the non-reducing end.
#
# Here I'm only including up to a 3-mer of residues of the original ligand
# since anything longer might as well just be a global docking run on the
# whole ligand.

a-D-Manp
b-D-Manp
b-D-Manp-(1->4)-a-D-Manp
b-D-Manp-(1->4)-b-D-Manp
b-D-Manp-(1->4)-b-D-Manp-(1->4)-a-D-Manp
```

We then pass 1pmh_probes.iupac as the input to the following python script:

```bash
python ~/glycographer/scripts/model_glycan_probes.py 1pmh_probes.iupac
# -> ./man.pdb
# -> ./bma.pdb
# -> ./bma14man.pdb
# -> ./bma14bma.pdb
# -> ./bma14bma14man.pdb
```

The slurm wrapper can also be used to submit this job on a cluster or as a dependency:

```bash
sbatch ~/glycographer/scripts/model_glycan_probes.sh 1pmh_probes.iupac
# -> ./man.pdb
# -> ./bma.pdb
# -> ./bma14man.pdb
# -> ./bma14bma.pdb
# -> ./bma14bma14man.pdb
```

Note: the default probe names use a shorthand consisting of PDB three-letter sugar codes to indicate anomeric and L/D isomeric form of each glycan residue sandwiched between two numbers to indicate linkage. Curly braces {} indicate branching to avoid most file naming conflicts (although this can cause issues in PyMOL).

## Part 3 - Complex Preparation