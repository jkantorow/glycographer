# Glycographer visualization workflow

Two ways to turn scored GlycanDock ensembles into binding-map visualizations:

- **Part 1 — interactive PyMOL**: load a receptor and explore individual probe
  maps and consensus maps by hand. Best for inspection and figure tweaking.
- **Part 2 — the CLI pipeline**: go from docking-output pose directories to a
  finished `.pse` (plus hotspot tables) in two scripted steps. Best when a run
  produces tens of maps.

Throughout, maps are on the REU scale where **favorable binding is negative** and
empty voxels are `0.0`.

> Run the project Python from the `glycographer` conda env:
> `/c/Users/computer7/miniconda3/envs/glycographer/python.exe` (plain `python`
> is not on PATH). On the cluster, use the SLURM wrappers in `scripts/slurm/`.

---

## Part 1 — Interactive PyMOL session

Start PyMOL and load the interactive commands:

```
run ~/glycographer/glycographer/vis.py
```

This registers the helper commands below (all usable directly at the PyMOL
prompt). It also makes the analysis functions importable.

### 1a. Receptor and reference geometry

```
format_background white
vis_receptor  1o7v_receptor.pdb          # loads as a white surface, returns "1o7v_receptor"
vis_grid      1o7v_gridbox.pdb           # optional: show the sampling grid
vis_crystal_ligand  1o7v_ligand.pdb      # optional: SNFG-colored reference pose
```

### 1b. Inspect a map before contouring

```
get_map_stats  maps/1o7v-Man_boltzmann_b0p5.dx
# -> Range (occupied): [-18.42, -0.31]  Mean: ...  Occupied voxels: N / M
```

Always pass the **`.dx` path** (not a loaded object name) — the stats come from
the numpy array, which is exact. A loaded-object name falls back to PyMOL's
histogram, whose bounds are only approximate.

### 1c. Per-probe isocontours (compare several probes)

`draw_map_contours` takes the **`.dx` path**, chooses levels for you, and shades
each contour by depth. Give each probe a distinct base color so overlays stay
readable:

```
draw_map_contours  maps/1o7v-Man_boltzmann_b0p5.dx, base_color=green,  mode=components
draw_map_contours  maps/1o7v-Glc_boltzmann_b0p5.dx, base_color=marine, mode=components
draw_map_contours  maps/1o7v-Fuc_boltzmann_b0p5.dx, base_color=red,    n=4, mode=absolute, step=1.0
```

Level modes:
- `absolute` (default) — start at the map minimum, step up by `step` REU
  (matches the by-hand method; comparable across probes on the shared scale).
- `quantile` — the deepest few percent of favorable voxels (self-normalizing).
- `components` — data-driven anti-bleed: stops below where distinct sites merge.

### 1d. Single-map surface (one probe, or a consensus field)

Color the receptor surface by a *single* continuous map — a per-probe map, or a
`consensus_min` / `consensus_mean` field. Load the map, then ramp:

```
load_volmap_from_dx  maps/consensus_boltzmann_b0p5_min.dx      # -> object "consensus_boltzmann_b0p5_min"
draw_mapped_surface  1o7v_receptor, consensus_boltzmann_b0p5_min, map_dx=maps/consensus_boltzmann_b0p5_min.dx
```

`map_dx=` lets it auto-derive the ramp bounds `[map_min, 0]`; alternatively set
`ramp_min=` explicitly (e.g. from `get_map_stats`). The recommended combination
is **consensus surface + per-probe contour cages** (1c) in the same scene.

### 1e. Identify and rank hotspots

```
show_hotspots  maps/1o7v-Man_boltzmann_b0p5.dx
# drops ranked, labeled spheres at hotspot centroids and prints the table:
#   rank  peak_value  mean_value  n_voxels  volume_A3   x     y     z
```

Attribute a hotspot to the receptor residues lining it:

```
show_hotspot_residues  maps/1o7v-Man_boltzmann_b0p5.dx, 1o7v_receptor.pdb, rank=1, radius=4.0
# shows + labels the lining residues as sticks and prints:
#   hotspot_rank  peak_value  chain  resid  resname  min_dist  n_atoms
```

Use `rank=all` to attribute every hotspot at once.

### 1f. Save

```
save  1o7v_session.pse
```

> The categorical **best_probe** surface ("which probe wins where") takes a probe
> list, so it is easiest to produce via the CLI (Part 2) rather than typed at the
> prompt.

---

## Part 2 — CLI pipeline: pose directories → `.pse`

Two steps: `map_ensembles.py` builds the maps + a manifest; `vis_maps.py` reads
the manifest and builds the PyMOL session.

### Step 1 — generate maps (`map_ensembles.py`)

All probes docked against the **same** receptor are voxelized on one shared grid
so the maps are index-aligned and consensus maps can be built in the same pass.

```
PY=/c/Users/computer7/miniconda3/envs/glycographer/python.exe

$PY scripts/utils/map_ensembles.py \
    runs/1o7v-Man runs/1o7v-Glc runs/1o7v-Gal runs/1o7v-Fuc runs/1o7v-Sia runs/1o7v-Xyl \
    --grid-pdb 1o7v_receptor.pdb \
    --map-type boltzmann --beta 0.5 \
    --outdir maps \
    --consensus mean best_probe support
```

Writes `maps/<probe>_boltzmann_b0p5.dx` for each probe, `maps/consensus_*.dx`,
and **`maps/map_manifest.json`** (grid geometry + every map's probe/tag/path +
consensus entries). For heterogeneous per-probe parameters use `--config
jobs.json`; see `map_ensembles.py --help`.

### Step 2 — build the visualization (`vis_maps.py`)

Point it at the manifest and the receptor. The default scene is **per-probe
contour cages + a consensus surface**:

```
$PY scripts/vis/vis_maps.py  maps/map_manifest.json \
    --receptor 1o7v_receptor.pdb \
    --tag boltzmann_b0p5 \
    --contour-mode components \
    --surface best_probe \
    --hotspots --radius 4.0 \
    --outdir vis  -o 1o7v_vis
```

Produces:
- `vis/1o7v_vis.pse` — receptor + per-probe contours (colorblind-safe palette) +
  the best_probe surface.
- with `--hotspots`: `vis/<probe>_boltzmann_b0p5_hotspots.csv` and
  `..._lining.csv` per probe — the ranked hotspots and their lining residues.

Notes:
- Only one map **tag** is visualized at a time (betas are never mixed); omit
  `--tag` to use the first and list the rest.
- `--surface` accepts `best_probe`, `min`, `mean`, `support`, `selectivity`, or
  `none`; it is skipped gracefully if that consensus map is absent.
- `--render` also ray-traces a PNG (needs OSMesa/libGL when headless).
- The manifest may be JSON or YAML (chosen by extension).

### On the cluster (SLURM)

The wrappers are thin pass-throughs — every argument after the script name goes
straight to the Python CLI:

```
sbatch scripts/slurm/map_ensembles.sh \
    runs/1o7v-Man runs/1o7v-Glc runs/1o7v-Gal runs/1o7v-Fuc runs/1o7v-Sia runs/1o7v-Xyl \
    --grid-pdb 1o7v_receptor.pdb --beta 0.5 --outdir maps --consensus mean best_probe support

sbatch scripts/slurm/vis_maps.sh  maps/map_manifest.json \
    --receptor 1o7v_receptor.pdb --tag boltzmann_b0p5 --contour-mode components --hotspots -o 1o7v_vis
```

Open the resulting `.pse` locally to render final figures interactively; the
`.pse` is always saved even if a headless `--render` can't ray-trace.

---

## Where things live

| Module | Role |
|---|---|
| `glycographer/map.py` | `GridSpec`, `Mapper` subclasses, `VolMap`, `ConsensusMap` |
| `glycographer/analysis.py` | pure numpy/pandas: `map_stats`, `choose_contour_levels`, `component_sweep`, `find_hotspots`, `attribute_hotspots_to_residues`, interface aggregation |
| `glycographer/colors.py` | SNFG colors, `color_by_magnitude`, colorblind-safe `probe_palette` |
| `glycographer/vis.py` | all PyMOL commands/recipes (imports analysis + colors) |
| `scripts/utils/map_ensembles.py` | CLI: poses → maps + manifest |
| `scripts/vis/vis_maps.py` | CLI: manifest → `.pse` + hotspot tables |
| `scripts/slurm/*.sh` | thin SLURM pass-throughs |
