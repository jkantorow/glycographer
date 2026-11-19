# Sampling grids

`scripts/build_sampling_grid.py` builds the point cloud GlycanDock starts
probes from: a set of coordinates in a thin shell above the receptor surface,
written as `PSD` pseudo-atoms on chain P. `scripts/run_glycandock.py --grid`
hands that file to Rosetta's `StartFrom` mover, which moves the glycoligand
onto one of the points before each decoy is docked.

**A grid point is where the ligand's centre of mass starts, not where it makes
contact.** That is why the default shell (1–4 Å above the surface) sits closer
in than a contact distance would suggest: `StartFrom` places the centre of
mass, then `RigidBodyRandomizeMover` spins the probe about that point, and
GlycanDock docks from there.

The logic lives in the package:

| Module | Role |
| --- | --- |
| `glycographer.grid` | Reads the receptor, resolves selections, computes the surface, builds and writes the grid |
| `glycographer.naming` | `validate_grid_id` — the grid id is the last field of a run_id |
| `glycographer.sample` | Passes the grid PDB to `StartFrom` (`GlycanDockRunner`) |
| `glycographer.vis` | `vis_grid` loads a grid into PyMOL beside the receptor |

Nothing here needs PyRosetta, open3d, or a surface mesh — only numpy, scipy
and PyMOL — so grids can be built on a workstation.

## Usage

Sample the whole surface:

```bash
python scripts/build_sampling_grid.py receptors/1pmh_rec.pdb -o grids/
# -> grids/1pmh_rec_grid_global.pdb  (+ .json)
```

Sample one site, named with a PyMOL selection:

```bash
python scripts/build_sampling_grid.py receptors/tetra.pdb -o grids/ \
    -g crd --site 'chain A and resn CA' --site-radius 14
# -> grids/tetra_grid_crd.pdb
```

Sample two sites at once, with nothing sampled in between:

```bash
python scripts/build_sampling_grid.py receptors/tetra.pdb -o grids/ \
    -g crd-ac --site 'chain A and resn CA' --site 'chain C and resn CA'
```

Check what a selection matches, and how big the job is, without building:

```bash
python scripts/build_sampling_grid.py receptors/tetra.pdb \
    --site 'chain A and resn CA' -g crd --dry-run
```

```text
  site 'chain A and resn CA': 3 atoms, 3 residues [A/CA405 A/CA406 A/CA407]
Region: within 14 A of [chain A and resn CA]
  bounds  10.3..63.7  4.3..61.3  17.1..69.5
  surface lattice 54 x 58 x 53 = 165,996 voxels at 1 A
  sampling nodes  21,141 before the shell and fit filters
  would write     grids/tetra_grid_crd.pdb
```

On the cluster, the same arguments go through the SLURM wrapper:

```bash
sbatch scripts/slurm/build_sampling_grid.sh receptors/1pmh_rec.pdb -o grids/
```

Then look at it in PyMOL (see [visualization_workflow.md](visualization_workflow.md)):

```text
vis_receptor  receptors/1pmh_rec.pdb
vis_grid      grids/1pmh_rec_grid_global.pdb
```

### Build the grid from the *prepared* receptor

`scripts/utils/prepare_complexes.py` sets the receptor to chain A, strips
hydrogens and may renumber residues (it writes a `*_resmap.csv` when it does).
Coordinates never move, so a grid built from the raw PDB is still
geometrically valid — but any `--site` selection written against the raw
numbering will then match the wrong atoms, or nothing. Prepare first, select
second.

## Choosing where to sample

| Option | Region |
| --- | --- |
| *(nothing)* | The whole receptor surface. The grid id defaults to `global`. |
| `--site SEL [--site-radius R]` | Everything within `R` Å (default 10) of any atom matching the PyMOL selection. Repeatable. |
| `--box XMIN YMIN ZMIN XMAX YMAX ZMAX` | An explicit axis-aligned box. `--bb-min`/`--bb-max` take the same corners as the old `build_gridbox.py`. |

`--site` and `--box` can be combined, in which case a point must satisfy both.

`--site` is a union of spheres around the selected atoms, not a bounding box
around them, so:

- Selections far apart give **disjoint patches** of grid, with nothing sampled
  in the space between. Two faces of a protein can be scanned in one run
  without paying for the volume between them — separated sites are each given
  their own lattice internally, and the points come out identical to what one
  big lattice would have produced.
- The region follows the shape of the selection. `--site 'chain A and resi
  110-130'` samples along a loop, not inside the cube that encloses it.

Useful selections:

```bash
--site 'chain A and resn CA'                     # a metal site and its surroundings
--site 'byres (chain A within 6 of resn CA)'     # the residues coordinating it
--site 'chain A and resi 365+366+383'            # named binding residues
--site 'chain A and resi 110-130'                # a loop
--site 'chain A' --site 'chain D'                # two subunits of a multimer
```

`drawBoundingBox.py` (in `scripts/utils/`) is still there if you want to
eyeball a box in PyMOL, but `--site` removes most of the reason to: there is
no longer a guess-and-check loop between PyMOL and the command line.

### Naming

The grid file is `<receptor>_grid_<grid_id>.pdb`, and `<grid_id>` becomes the
trailing field of every run_id built from it:

```text
1o7v_rec_grid_global.pdb   ->   1o7v-gal_global   ->   1o7v-gal_global_1000p
```

So a grid id is letters, digits and `-` only — no `_`, which separates run_id
fields (`validate_grid_id` in `glycographer/naming.py`, the same rule probe
names follow). `global` is conventional for a whole-surface grid; name local
grids after the site (`crd`, `cleft`, `ca2`). **`--grid-id` is required for
any grid that covers only part of the receptor**, so that two grids over the
same receptor can never collide in a run_id.

Every build also writes a `.json` sidecar with each parameter and the
point counts at each filtering stage; the same key/value pairs are written as
`REMARK` lines in the PDB, and `grid.read_grid_params()` reads them back.

## How the surface is computed

Distances are measured from the **solvent-excluded (molecular) surface**,
computed directly from atom coordinates and van der Waals radii. Two radii
define the geometry:

| Option | Default | Role |
| --- | --- | --- |
| `--probe-radius` | 1.4 Å (water) | Rolls over the atoms to define the surface that `--shell-min`/`--shell-max` are measured from. Fills in crevices narrower than a water molecule, as a Connolly surface does. |
| `--fit-radius` | 2.5 Å (a pyranose ring) | Drops points that only a smaller sphere could reach from bulk solvent — the bottom of a narrow groove, a sealed interior cavity. This is the cryptic-pocket filter. |

Both are one computation at two radii, on a single lattice (default pitch
1 Å, `--lattice-spacing`):

1. `d(x) = min_i(|x − c_i| − r_i)`, exact, from one KD-tree per distinct radius.
2. Probe centres that fit are `d ≥ r`. Label them; components that reach the
   lattice edge are bulk solvent, the rest are sealed cavities.
3. Solvent is everything within `r` of a bulk probe centre (one Euclidean
   distance transform). Everything else is receptor interior.
4. Height above the surface is the distance transform of that interior,
   clipped by `d` — which is exact, and is the answer wherever the surface is
   convex, i.e. almost everywhere.

The lattice therefore only costs accuracy inside concavities, and only in the
direction of keeping points rather than dropping them. A whole-surface grid on
a 4,000-atom tetramer takes about a second and ~20 MB.

### What replaced the STL pipeline

The old `build_gridbox.py` exported a PyMOL surface to STL and raycast it with
open3d. That is gone, and with it two hard dependencies:

- **Open-source PyMOL cannot write STL** (`save *.stl` is incentive-only), so
  the script could not run in glycographer's own conda environment. Its
  fallback shelled out to Chimera, which was never a declared dependency.
- **open3d does not solve on Windows** in this environment, so grids could
  only be built on the cluster.

Neither is needed now: the surface comes from the atoms. A mesh was always an
approximation of what the raycasting then measured against, so dropping it
removes a discretization step rather than adding one.

Grids built the old way are still valid input — nothing about the file format
changed. The new global grid is a superset of the old one in practice (on
`1o7v_rec`, 99.9% of the legacy points have a new point within 2 Å), mostly
because the surface is now derived from every atom rather than from a
mesh, and because metal ions are treated properly (below).

### Metal ions

`glycographer.grid.ELEMENT_RADII` uses Shannon **ionic** radii for metals
(Ca²⁺ 1.14 Å), not neutral-atom van der Waals radii (Ca 2.31 Å). This is
load-bearing for C-type lectins: a sugar's O3/O4 coordinates Ca²⁺ at ~2.4 Å,
which is *inside* the neutral-atom radius. With the vdW value the coordination
site reads as receptor interior and the grid thins out exactly where the
glycan is known to bind; with the ionic value a ligand centre lands 2.1–5.1 Å
from the ion.

Measured on the DC-SIGN tetramer (`tetra_remodel.pdb`, 12 Ca²⁺), the nearest
grid point to the principal Ca²⁺ of each CRD moves from 3.4–3.9 Å (vdW radii)
to 2.3–2.9 Å (ionic radii).

Override any element with `--radius ELEM=R`, e.g. `--radius CA=2.31` to get
the old behaviour back. Also check that ions are actually *in*
`--surface-sel`: the default `not (solvent or hydro)` keeps them, but a
selection like `polymer` would silently drop them and open a hole in the
surface where the ion sits.

### Is the cryptic-pocket filter worth it?

Yes, but for correctness rather than speed. Measured on two receptors at the
default `--fit-radius 2.5`:

| Receptor | Points in shell | Dropped | Receptor atoms within 8 Å, kept | …dropped |
| --- | --- | --- | --- | --- |
| `1o7v_rec` (1,041 atoms) | 2,942 | 32 (1.1%) | 23.1 | 67.9 |
| `tetra_remodel` (4,180 atoms) | 8,785 | 231 (2.6%) | 26.9 | 78.3 |

The dropped points are about three times as buried as the kept ones, so the
filter is selecting what it claims to. But it only removes 1–3% of points, so
it saves very little compute — the old per-point, 12-ray raycasting
implementation cost more than the docking it avoided. As one morphological
pass over a lattice it is effectively free (no measurable change in runtime),
which is the only reason to keep it on by default.

Raise `--fit-radius` to be stricter (3.5 Å drops 4–7%), or pass
`--no-fit-filter` to keep every point in the shell.

## Options

| Option | Effect |
| --- | --- |
| `-g, --grid-id ID` | Grid name; last field of the run_id. Defaults to `global`, required for a partial grid |
| `-o, --outdir DIR` | Where the grid files go (default `.`) |
| `--out PATH` | Write the PDB at an explicit path instead |
| `--site SEL` | PyMOL selection to sample around; repeatable |
| `--site-radius R` | How far from the `--site` atoms to sample (default 10 Å) |
| `--box`, `--bb-min`, `--bb-max` | Explicit bounding box |
| `-s, --spacing A` | Distance between sampling points (default 2.0 Å) |
| `--shell-min`, `--shell-max` | Sampling shell above the surface (default 1.0–4.0 Å) |
| `--surface-sel SEL` | Atoms that make up the surface (default `not (solvent or hydro)`) |
| `--probe-radius R` | Surface-defining probe (default 1.4 Å) |
| `--fit-radius R` | Cryptic-pocket filter probe (default 2.5 Å) |
| `--no-fit-filter` | Keep every point in the shell |
| `--radius ELEM=R` | Override one element's radius |
| `--lattice-spacing A` | Geometry lattice pitch; must divide `--spacing` (default: largest divisor ≤ 1 Å) |
| `--max-voxels N` | Refuse a lattice larger than this (default 40,000,000, ~50 bytes each) |
| `--write-xyz` | Also write the points as `.xyz` |
| `--no-manifest` | Skip the `.json` sidecar |
| `--overwrite` | Replace an existing grid file |
| `--dry-run` | Report selections and lattice size, then exit |

### Options that are gone

| Old (`build_gridbox.py`) | Now |
| --- | --- |
| `-res, --gridres` | `-s, --spacing` |
| `--padding` | Removed. The lattice margin is derived from `--shell-max` and the two probe radii, so there is nothing to tune |
| `--chunk-size` | Removed. Memory is bounded by `--max-voxels`; KD-tree queries chunk themselves |
| `--cryptic-radius` | `--fit-radius`, with a different (and principled) meaning |
| `--outprefix` | `-g/--grid-id` plus `-o/--outdir`; geometry is recorded in the `.json`/`REMARK`s instead of the filename |
| STL input | Removed. Pass the receptor PDB |

## Using the API

```python
from glycographer.grid import (load_receptor, SamplingRegion,
                               build_sampling_grid, write_grid)

rec = load_receptor('receptors/tetra.pdb')          # default surface selection
region = SamplingRegion.around(rec.site_coords('chain A and resn CA'),
                               radius=14.0)
grid = build_sampling_grid(rec, region=region, spacing=2.0, shell=(1.0, 4.0))
write_grid(grid, 'grids/tetra_grid_crd.pdb')

grid.n_points, grid.counts, grid.bounds()
```

Lower-level pieces, for scoring or filtering points of your own:

```python
from glycographer.grid import vdw_distance, surface_fields, read_grid_points

d = vdw_distance(points, rec.coords, rec.radii)     # exact, signed
axes, height, reachable = surface_fields(rec, lo, hi, 1.0)
pts = read_grid_points('grids/tetra_grid_crd.pdb')  # read a grid back
```

`build_sampling_grid` takes a coordinate-only `Receptor` too (build one
directly with `Receptor(path=..., selection=..., coords=..., radii=...)`); only
`site_coords` needs the live PyMOL object.

## Notes

- **Point count and the PDB format.** Serial numbers run out at 99,999 atoms;
  the writer warns and wraps. A global grid at 2 Å on a tetramer is ~9,000
  points, so this only matters at very fine spacing.
- **Grids are reproducible.** Every lattice is snapped to a global one of
  pitch `--spacing`, so two grids over the same receptor — a global one and a
  site one, or two sites — put their points on exactly the same nodes, and
  rebuilding a grid gives a byte-identical file.
- **Sampling density vs. decoy count.** The grid sets *where* probes can
  start, not how many decoys are run; `run_glycandock.py -n` does that. Halving
  `--spacing` multiplies the points by ~4 without changing the number of
  decoys, so each start point gets a quarter as many samples.
