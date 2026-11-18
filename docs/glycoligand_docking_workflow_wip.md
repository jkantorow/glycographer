# Performing standard a standard GlycanDock protocol on a putative receptor-glycoligand complex

Here, we walk through the example of docking the glycoligand of PDB 1SL4 aligned to the known binding site of chain C of a tetrameric dc-sign carbohydrate recognition domain model.

I have glycographer installed in my home directory, but script paths may vary depending on where you've installed glycographer.

I would like to later split this into a broader workflow (stored here) and a notebook example showing specific usage. Also, this information is subject to change as I try to design a simpler way of executing all of these steps instead of having to do them one by one.

## Part 1 - Complex Preparation

In this case, the receptor and ligand are already present with the glycoligand from PDB 1SL4 marked as chain E. Note: this step does not require pyrosetta, but subsequent steps will.

```bash
python ~/glycographer/scripts/utils/prepare_complexes.py -c tetra-native1sl4C.pdb tetra-mapaln1sl4C.pdb --lig-sel "chain E" -o cleaned
# -> ./cleaned_tetra-native1sl4C.pdb
```

## Part 2a - Complex Prepacking

You can do this in your current terminal by calling the python script:

```bash
python ~/glycographer/scripts/prepack_complex.py cleaned_tetra-native1sl4C.pdb -o prepacked_tetra-native1sl4C.pdb
# -> ./prepacked_tetra-native1sl4C.pdb
```

Or, if you are working on a SLURM-enabled computing cluster, you can use the following bash script:

```bash
sbatch ~/glycographer/scripts/slurm/prepack_complex.sh cleaned_tetra-native1sl4C.pdb -o prepacked_tetra-native1sl4C.pdb
# -> ./prepacked_tetra-native1sl4C.pdb
# -> ./gdock_prepack_<slurm-job-id>.err
# -> ./gdock_prepack_<slurm-job-id>.out
```

## Part 2b (optional) - GlycanDock Test Run

This can be performed in your interactive bash terminal through the following python script:

```bash
python ~/glycographer/scripts/run_glycandock.py prepacked_tetra-native1sl4C.pdb -n 5 -o test_tetra-native1sl4C.pdb --native cleaned_tetra-native1sl4C.pdb
# -> ./test_tetra-native1sl4C.tar.xz
```

Or, alternatively via slurm batch job submission including parallelization (run_glycandock.py automatically determines block sizes and lengths from the slurm --array flag if one is supplied):

```bash
sbatch --job-name=test_tetra-native1sl4C --array=0-4 --time=00:30:00 ~/glycographer/scripts/slurm/run_glycandock.sh prepacked_tetra-native1sl4C.pdb -n 5 -o test_tetra-native1sl4C.pdb --native cleaned_tetra-native1sl4C.pdb
# -> ./test_tetra-native1sl4C_block000.tar.xz
# -> ./test_tetra-native1sl4C_block001.tar.xz
# -> ./test_tetra-native1sl4C_block002.tar.xz
# -> ./test_tetra-native1sl4C_block003.tar.xz
# -> ./test_tetra-native1sl4C_block004.tar.xz
# -> ./gdock_<slurm-job-id>_0.out
# -> ./gdock_<slurm-job-id>_1.out
# -> ./gdock_<slurm-job-id>_2.out
# -> ./gdock_<slurm-job-id>_3.out
# -> ./gdock_<slurm-job-id>_4.out
```

This testing step can be useful for determining compute time per each output decoy which can then be used to assign efficient block sizing for a full run.

## Part 3 - Full Production GlycanDock Run

The original GlycanDock publication cites an output of 1000-2000 decoys per input putative receptor-glycoligand complex. Here, we output 2000 decoys with an array of 50 blocks each consisting of 40 output decoys. With this large, tetrameric receptor and branched glycoligand, my tests resulted in each output decoy taking around 14 minutes of compute time; ~14 min/decoy * 40 decoys/block * 1/60 hr/min = ~9.33 hr/block, and with each block running in a parallel array, all 50 blocks should be done in around 12-15 hr accounting for node performance variation and decoys occasionally failing the score filter.

Since a full 2000 pose run on a large receptor-glycoligand complex like this one could take a whole month to complete without parallelization, it is recommended that SLURM batch submission be used to accelerate glycan docking:

```bash
sbatch --job-name=tetra-native1sl4C --array=0-49 --time=15:00:00 ~/glycographer/scripts/slurm/run_glycandock.sh prepacked_tetra-native1sl4C.pdb -n 2000 -o tetra-native1sl4C --native cleaned_tetra-native1sl4C.pdb
# -> ./tetra-native1sl4C_block0[00-49].tar.xz
# -> ./gdock_<slurm-job-id>_[0-49].out
```

## Part 4 (optional) - Merge Output Blocks

50 output decoy blocks and 50 output log files could become an unmanagable number of inodes when performing multiple runs. Glycographer provides a script to merge each output block and log file of a run into a single compressed archive for convenience:

```bash
python ~/glycographer/scripts/utils/merge_glycandock_blocks.py . --run-id tetra-native1sl4C --logs gdock_<slurm-job-id>_*.out
# -> ./tetra-native1sl4C.tar.xz
```

Or, alternatively with the corresponding slurm script:

```bash
slurm ~/glycographer/scripts/slurm/merge_glycandock_blocks.sh . --run-id tetra-native1sl4C --logs gdock_<slurm-job-id>_*.out
# -> ./tetra-native1sl4C.tar.xz
```

Note: this removes the original unmerged files after merging by default, but they can be kept with the --keep-input flag.