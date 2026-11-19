# Probe modeling

`scripts/model_glycan_probes.py` builds energetically minimized free-glycan
probe models from Rosetta simplified-IUPAC sequences and writes one
`<probe_name>.pdb` per probe. These are the glycoligands that
`scripts/utils/prepare_complexes.py` combines with a receptor before docking.

The logic lives in the package:

| Module | Role |
| --- | --- |
| `glycographer.model` | Reads sequence files, builds the pose (`pose_from_saccharide_sequence`), minimizes it (`GlycanSampler`), writes the PDB and manifest |
| `glycographer.naming` | Parses simplified IUPAC (pure Python) and builds the probe name |
| `glycographer.utils.init_rosetta` | Starts the Rosetta session from a flags file in `config/` |
| `config/glycan_modeling.init` | Rosetta flags for modeling (the same flags the old `glycomodeler.py` used) |

## Usage

Model a single sequence. Quote it, because `( ) [ ] >` are special to the shell:

```bash
python scripts/model_glycan_probes.py 'a-Neup5Ac-(2->3)-b-D-Galp-(1->3)-b-D-GalpNAc'
# -> ./sia23gal13nga.pdb

python scripts/model_glycan_probes.py 'a-Neup5Ac-(2->3)-b-D-Galp' -n 3sl -o probes/
# -> probes/3sl.pdb
```

Model a batch from a file:

```bash
python scripts/model_glycan_probes.py probes.iupac -o probes/ --manifest probes/probes.csv
```

Check names and output paths before modeling. This works without PyRosetta,
so you can run it on a workstation:

```bash
python scripts/model_glycan_probes.py probes.iupac -o probes/ --dry-run
```

On the cluster, submit the same arguments through the slurm wrapper. It runs
as a single job; most probes take seconds to minutes each (any probe containing sialic acid will be 10s of minutes to 1-2 hours due to its rotamers):

```bash
sbatch scripts/slurm/model_glycan_probes.sh probes.iupac -o probes/ --manifest probes/probes.csv
```

### Options

| Option | Effect |
| --- | --- |
| `-o, --outdir DIR` | Where the `<name>.pdb` files go (default `.`) |
| `-n, --name NAME` | Explicit name for a single-sequence input |
| `--seed N` | Fix Rosetta's RNG. `GlycanSampler` starts from a randomized conformation, so set this when models must be reproducible |
| `--no-minimize` | Write the idealized built pose without running `GlycanSampler` |
| `--overwrite` | Remodel probes whose PDB already exists. By default they are skipped, so a failed batch can be rerun as is |
| `--manifest FILE` | CSV with one row per probe: name, sequence, pdb, status, residue count, total score, residue-code check, error |
| `--options FILE` | A different Rosetta flags file (default `config/glycan_modeling.init`) |
| `--dry-run` | Print names and paths, then exit without initializing Rosetta |

If any probe fails, the script reports the failure, finishes the rest of the
batch, and exits 1. A `--dependency=afterok` job will therefore not start on
an incomplete probe set.

## Sequence files

A sequence file (`.iupac`, `.txt` or any other name) has one sequence per
line. A line can optionally add whitespace and an explicit probe name after
the sequence. Blank lines and `#` comments are ignored.

```text
# O-glycan core 1, sialylated: auto-named sia23gal13nga
a-Neup5Ac-(2->3)-b-D-Galp-(1->3)-b-D-GalpNAc
# Lewis X: auto-named gal14{fuc13}nag
b-D-Galp-(1->4)-[a-L-Fucp-(1->3)]-b-D-GlcpNAc
# explicit name
a-Neup5Ac-(2->3)-b-D-Galp-(1->4)-b-D-Glcp    3sl
```

Sequences follow Rosetta's simplified IUPAC:
- Write residues from the non-reducing end to the reducing end.
- Put each branch in `[...]` directly before the residue it attaches to, and end the branch with its linkage, e.g. `[a-L-Fucp-(1->3)]`.
- If you omit the anomer, Rosetta assumes alpha, so `NeupAc` is α-Neu5Ac. State the anomer and D/L explicitly to be safe.

Every sequence in the batch is named before Rosetta starts, and all naming
problems are reported together. Naming fails on duplicate sequences,
including the same glycan written with its branches in a different order,
and on residues the naming table doesn't cover.

## Probe names

The default name abbreviates each residue to its lowercase PDB
3-letter code and each linkage to its two locants, with no punctuation:

| Sequence | Name |
| --- | --- |
| `b-D-Galp` | `gal` |
| `a-Neup5Ac-(2->8)-a-Neup5Ac` | `sia28sia` |
| `a-Neup5Ac-(2->3)-b-D-Galp-(1->3)-b-D-GalpNAc` | `sia23gal13nga` |
| `b-D-Galp-(1->4)-[a-L-Fucp-(1->3)]-b-D-GlcpNAc` | `gal14{fuc13}nag` |
| `a-Neup5Ac-(2->3)-b-D-Galp-(1->4)-[a-L-Fucp-(1->3)]-b-D-GlcpNAc` | `sia23gal14{fuc13}nag` |
| `a-D-Manp-(1->3)-[a-D-Manp-(1->6)]-b-D-Manp-(1->4)-b-D-GlcpNAc-(1->4)-b-D-GlcpNAc` | `man16{man13}bma14nag14nag` |

- **Anomers and modifications come from the code.** PDB codes already
  distinguish α from β, D from L, and modifications such as NAc and uronic acid
  (`gal`/`gla`, `nag`/`ndg`, `sia`/`slb`). Codes are taken from Rosetta's
  `pdb_sugar.codes`, the table Rosetta uses to write the PDB.
- **Branches use `{}`.** They sit where Rosetta puts `[...]`. Square brackets
  are glob characters and would silently break `glob.glob()` on any run
  directory or pose path built from the name. Braces are literal to Python's
  glob and, with no comma inside, to bash. PyMOL is the one exception: it
  does not allow `{}` in object names and replaces them with `_`, so the map
  `tetra-man16{man13}man_..._b0p5.dx` loads as the object
  `tetra-man16_man13_man_..._b0p5`. The `vis.py` commands account for this
  (see [visualization_workflow.md](visualization_workflow.md)).
- **Names are canonical.** At each branch point the largest child subtree
  continues the main chain. On a tie, the child linked at the higher-numbered
  position continues it. So the same glycan always gets the same name,
  however its sequence was written.
- **Names contain no `_`.** The name becomes the `<ligand>` field of a run id
  (`<receptor>-<ligand>_<grid_id>`, e.g. `1o7v-gal14{fuc13}nag_global`), where
  `_` separates fields. Explicit names are held to the same rule: letters,
  digits, `-` and `{}`.

After writing each model, the residue codes in the PDB are checked against the
codes the name was built from. A mismatch is printed as a warning and recorded
in the manifest's `code_check` column. It means the naming table and Rosetta
disagree, and that probe's name should not be trusted.

### Residues without an auto-name

The naming table (`glycographer/naming.py`, `_CODE_TABLE`) covers common
residues: Glc, Gal, Man, GlcNAc, GalNAc, GlcN, GlcA, GalA, Fuc, Rha, Xyl, Ara,
Rib, Fru, Neu5Ac, Neu5Gc, Kdo and a few others. Some sequences use a residue
the table doesn't cover, for example sulfated or methylated sugars like
`b-D-Galp6S`. Name those probes explicitly (second column or `-n`), or add a
row to `_CODE_TABLE` from Rosetta's
`database/input_output/3-letter_codes/pdb_sugar.codes`.

### Why not GlyTouCan, GLYCAM or WURCS names?

- **GlyTouCan accessions** (8-character IDs like `G12345AB`) are opaque, need a registry
  lookup, and many small fragments aren't registered. They are worth recording
  as metadata, not as filenames.
- **GLYCAM condensed** (`DNeup5Aca2-3DGalpb1-3DGalpNAcb1-OH`), **IUPAC
  condensed** (`Neu5Ac(a2-3)Gal(b1-3)GalNAc`), **WURCS** and **GlycoCT** are
  unambiguous. But they are long and full of characters that need quoting in
  a shell or break globbing.

The PDB-code shorthand is short, readable and filename-safe, and it maps
exactly onto the residue names in the model. Keep the full sequence beside it
with `--manifest`.

## Using the API

```python
from glycographer.model import init_modeling, probe_specs, read_sequence_file, write_probe_models
from glycographer.naming import glycan_name

glycan_name('b-D-Galp-(1->4)-[a-L-Fucp-(1->3)]-b-D-GlcpNAc')  # 'gal14{fuc13}nag', no PyRosetta needed

init_modeling(seed=1)              # once per process
specs = probe_specs(read_sequence_file('probes.iupac'))
records = write_probe_models(specs, outdir='probes', manifest='probes/probes.csv')
```

Rosetta reads its options once per process, and modeling uses different flags
from docking, so modeling cannot share a process with a GlycanDock run. A
pipeline should run it as its own step, as it already does for Stage 0
prepacking.

## Notes

- **Score function.** `config/glycan_modeling.init` passes `-beta`, so probes
  are minimized under beta_nov16, while GlycanDock samples under the default
  score function. This matches every existing probe model. If you change it,
  remodel all probes together.
- **Old script.** `scripts/glycomodeler.py` is replaced by this script. Its
  `-o/--outprefix` wrote `<prefix>_<i>.pdb`; outputs are now named per probe.
