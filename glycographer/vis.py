'''
Functions for creating and rendering volume map visualizations in PyMOL.

Each function can also be imported and used within a PyMOL session via the
PyMOL run command:
> run /path/to/glycographer/vis.py
'''

import os
import sys

import pandas as pd
from pymol import cmd
from pymol import stored

# vis.py doubles as a package module and a script run inside an interactive
# PyMOL session via `run /path/to/glycographer/vis.py`. In the latter case the
# package is not necessarily importable, so put the repo root on sys.path before
# importing sibling modules. Harmless when imported normally.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from glycographer.colors import get_snfg_color, color_by_magnitude  # noqa: E402

@cmd.extend
def format_background(bg_rgb='white'):
    '''
    Standardize the viewing environment for the output in pymol or vmd.
    '''
    cmd.set('bg_rgb', bg_rgb)
    cmd.set('depth_cue', 0)
    cmd.set('ray_shadow', 0)

@cmd.extend
def vis_receptor(receptor_pdb: str):
    '''
    Load and display receptor as a white surface:
    '''
    rec_name = os.path.basename(receptor_pdb).replace('.pdb', '')
    cmd.load(receptor_pdb, rec_name)
    cmd.hide('cartoon', rec_name)
    cmd.show('surface', rec_name)
    cmd.color('white', rec_name)

    return rec_name

@cmd.extend
def vis_grid(grid_pdb: str):
    '''
    Load and display sampling grid.
    '''
    grid_name = os.path.basename(grid_pdb).replace('.pdb', '')
    cmd.load(grid_pdb, grid_name)

    return grid_name

@cmd.extend
def vis_crystal_ligand(lig_pdb: str):
    '''
    Load and display known crystal glycoligand pose as licorice.
    '''
    lig_name = os.path.basename(lig_pdb).replace('.pdb', '')
    cmd.load(lig_pdb, lig_name)
    cmd.show('licorice', lig_name)

    stored.lig_residues = []
    cmd.iterate(f'{lig_name} and name c1', 'stored.lig_residues.append(resn)')
    unique_res = set(stored.lig_residues)

    for res in unique_res:
        cmd.color(get_snfg_color(res), f'{lig_name} and resn {res}')

    return lig_name

@cmd.extend
def load_volmap_from_dx(map_dx: str):
    '''
    Load a volume density map from a dx file and create a pymol
    object from it.
    '''
    map_name = os.path.basename(map_dx).replace('.dx', '')
    cmd.load(map_dx, map_name)

    return map_name

@cmd.extend
def get_map_stats(map_source: str, occupied_only=True):
    '''
    Print and return distribution statistics for a map.

    Pass a path to a .dx file for accurate numpy-computed statistics (preferred:
    PyMOL's get_volume_histogram reports histogram bin edges, not true min/max,
    and so disagrees with the values seen at map load). If a loaded PyMOL map
    object name is passed instead, falls back to get_volume_histogram with a
    note. Empty voxels (sentinel 0.0) are excluded by default.
    '''
    if os.path.isfile(map_source):
        from glycographer.map import VolMap
        from glycographer.analysis import map_stats
        volmap = VolMap.from_dx(map_source)
        stats = map_stats(volmap, occupied_only=occupied_only)
        scope = 'occupied' if occupied_only else 'all'
        print(f'Range ({scope}): [{stats["min"]}, {stats["max"]}]')
        print(f'Mean: {stats["mean"]} (stdev: {stats["std"]})')
        print(f'Occupied voxels: {stats["n_occupied"]} / {stats["n_voxels"]}')
        return stats

    print(f'"{map_source}" is not a file; reading the loaded PyMOL object via '
          'get_volume_histogram (bin-edge bounds, may differ from true min/max '
          '-- pass the .dx path for exact values).')
    stats = cmd.get_volume_histogram(map_source, 0)
    print(f'Range: [{stats[0]}, {stats[1]}]')
    print(f'Mean: {stats[2]} (stdev: {stats[3]})')
    return stats

@cmd.extend
def draw_contour(map_name: str, level: float):
    '''
    Draw an isomesh contour from a volume density map at a specified level.
    '''
    contour_name = '_'.join(['contour', map_name, str(level)])
    cmd.isomesh(contour_name, map_name, float(level))

    return contour_name

@cmd.extend
def draw_map_contours(map_dx: str, base_color='red', n=4, mode='absolute',
                      step=1.0, smooth_sigma=None):
    '''
    Draw a set of isocontours for one map at programmatically chosen levels,
    each shaded by its level (deepest = full base_color, shallower = lighter).

    Replaces the by-hand "start near the minimum, step up until they bleed"
    workflow. Levels come from analysis.choose_contour_levels; see it for the
    modes:
      'absolute'   -- start at the map minimum, step up by `step` REU (default).
      'quantile'   -- tail quantiles of the favorable distribution.
      'components' -- data-driven anti-bleed: stops below where sites merge.

    Requires the .dx path (not a loaded object name) so the levels can be
    computed from the voxel array. For a per-probe overlay, pass each probe a
    distinct base_color from colors.probe_palette.

    Returns
    -------
    (contour_names, levels)
    '''
    from glycographer.map import VolMap
    from glycographer.analysis import choose_contour_levels

    n = int(n)
    step = float(step)
    smooth_sigma = float(smooth_sigma) if smooth_sigma else None

    volmap = VolMap.from_dx(map_dx)
    levels = choose_contour_levels(volmap, n=n, mode=mode, step=step,
                                   smooth_sigma=smooth_sigma)
    if not levels:
        print(f'No favorable (negative) voxels in {map_dx}; nothing to contour.')
        return [], []

    map_name = load_volmap_from_dx(map_dx)

    # Shade deepest level at full base_color, shallower toward (but not to)
    # white: map the shallowest level to t~0.5 by doubling the range span so
    # every contour stays visible on a light background.
    lo, hi = min(levels), max(levels)
    span = hi - lo
    lvl_range = (lo, hi + span) if span > 0 else (lo, lo + 1.0)

    contour_names = []
    for i, lvl in enumerate(levels):
        contour_name = draw_contour(map_name, lvl)
        rgb = color_by_magnitude(base_color, lvl, lvl_range,
                                 negative_is_better=True)
        color_name = f'mag_{map_name}_{i}'
        cmd.set_color(color_name, list(rgb))
        cmd.color(color_name, contour_name)
        contour_names.append(contour_name)

    print(f'{map_name}: drew {len(levels)} contours at '
          f'{[round(l, 2) for l in levels]} (mode={mode})')
    return contour_names, levels

@cmd.extend
def draw_mapped_surface(rec_name: str, map_name: str,
                        ramp_min=None, ramp_max=0.0,
                        color_min='red', color_max='white',
                        map_dx=None):
    '''
    Project a mapped molecular surface of probe score density onto the receptor.

    For REU maps (favorable = negative), the ramp runs from ramp_min (the most
    favorable value, colored color_min) to ramp_max=0 (empty, colored color_max),
    so strong binding shows as saturated color fading to white. If ramp_min is
    None it is auto-derived from the map's favorable minimum; pass map_dx (the
    .dx path) so the value can be read from the array, or set ramp_min explicitly.

    Best suited to a single continuous field -- a per-probe map, or a
    consensus_min / consensus_mean map to show generalist binding strength.
    '''
    if ramp_min is None:
        src = map_dx if (map_dx and os.path.isfile(map_dx)) else (
            map_name if os.path.isfile(map_name) else None)
        if src:
            from glycographer.map import VolMap
            from glycographer.analysis import map_stats
            ramp_min = map_stats(VolMap.from_dx(src))['min']
        if ramp_min is None:
            raise ValueError('ramp_min could not be auto-derived; pass map_dx '
                             '(the .dx path) or set ramp_min explicitly.')
    ramp_min, ramp_max = float(ramp_min), float(ramp_max)

    ramp_name = f'ramp_{map_name}'
    cmd.ramp_new(ramp_name, map_name, [ramp_min, ramp_max],
                 [color_min, color_max])

    cmd.set('surface_color', ramp_name, rec_name)
    cmd.set('surface_ramp_above_mode', 1)
    cmd.set('surface_quality', 1)
    cmd.show('surface', rec_name)

    return ramp_name

@cmd.extend
def show_hotspots(map_dx: str, level=None, min_voxels=3, smooth_sigma=None):
    '''
    Segment a map into discrete hotspots and mark each ranked centroid as a
    labeled pseudoatom sphere (rank 1 = most favorable). Prints the ranked
    hotspot table (peak/mean energy, size, world-space centroid).

    If level is None it is chosen automatically at the level where the most
    distinct sites resolve (analysis.find_hotspots via the component sweep).
    '''
    from glycographer.map import VolMap
    from glycographer.analysis import find_hotspots

    level = float(level) if level not in (None, 'None') else None
    min_voxels = int(min_voxels)
    smooth_sigma = float(smooth_sigma) if smooth_sigma else None

    volmap = VolMap.from_dx(map_dx)
    df = find_hotspots(volmap, level=level, min_voxels=min_voxels,
                       smooth_sigma=smooth_sigma)
    map_name = os.path.basename(map_dx).replace('.dx', '')
    if df.empty:
        print(f'No hotspots found in {map_name}.')
        return df

    group = f'hotspots_{map_name}'
    for _, row in df.iterrows():
        name = f'{group}_{int(row["rank"])}'
        cmd.pseudoatom(name, pos=[float(row.x), float(row.y), float(row.z)],
                       b=float(row.peak_value))
        cmd.show('spheres', name)
        cmd.set('sphere_scale', 0.5, name)
        cmd.label(name, f'"{int(row["rank"])}"')
    cmd.group(group, f'{group}_*')

    print(f'{map_name}: {len(df)} hotspots (level={df.attrs.get("level"):.2f})')
    print(df.to_string(index=False))
    return df

@cmd.extend
def show_hotspot_residues(map_dx: str, receptor_pdb: str, rank=1, radius=4.0,
                          level=None, min_voxels=3, highlight='orange'):
    '''
    Attribute a hotspot to its lining receptor residues (geometry only) and
    display them: shows those residues as sticks, colors and labels them, and
    prints the ranked attribution table.

    Pass rank='all' to display the lining residues of every hotspot. The
    receptor is loaded (as a surface) if not already present. Answers "which
    residues line hotspot N of this probe" -- the design-guidance view.
    '''
    from glycographer.map import VolMap
    from glycographer.analysis import attribute_hotspots_to_residues

    radius = float(radius)
    min_voxels = int(min_voxels)
    level = float(level) if level not in (None, 'None') else None

    rec_name = os.path.basename(receptor_pdb).replace('.pdb', '')
    if rec_name not in cmd.get_names('objects'):
        vis_receptor(receptor_pdb)

    volmap = VolMap.from_dx(map_dx)
    att = attribute_hotspots_to_residues(volmap, receptor_pdb, level=level,
                                         min_voxels=min_voxels, radius=radius)
    if att.empty:
        print('No lining residues found (no hotspots or none within radius).')
        return att

    if str(rank).lower() != 'all':
        att = att[att['hotspot_rank'] == int(rank)]
        if att.empty:
            print(f'No hotspot with rank {rank}.')
            return att

    # Build a residue selection and highlight it.
    parts = []
    for _, r in att.iterrows():
        chain = str(r['chain']).strip()
        resi = f'resi {int(r["resid"])}'
        parts.append(f'(chain {chain} and {resi})' if chain else f'({resi})')
    sel = f'{rec_name} and ({" or ".join(parts)})'

    sel_name = f'lining_{os.path.basename(map_dx).replace(".dx", "")}'
    cmd.select(sel_name, sel)
    cmd.show('sticks', sel_name)
    cmd.color(highlight, f'{sel_name} and elem C')
    cmd.set('label_size', 14)
    cmd.label(f'{sel_name} and name CA', 'f"{resn}{resi}"')

    print(f'Lining residues (radius {radius} A):')
    print(att.to_string(index=False))
    return att

def draw_best_probe_surface(rec_name: str, best_probe_map: str, probe_labels):
    '''
    Color a single receptor surface by which probe wins at each voxel, using a
    best_probe consensus map (1-based probe index per voxel, 0 where none).

    This is the "compare all probes at a glance" view: one surface, each patch
    colored by the identity of its strongest probe from a colorblind-safe
    categorical palette. probe_labels is the ordered probe list matching the
    map's 1..N indices (as written by map_ensembles.py's ConsensusMap).

    Not @cmd.extend (takes a label list -> use from a script/notebook). Depends
    on the consensus best_probe map, so it is ready for when that methodology is
    finalized. Note PyMOL ramps interpolate, so boundaries between two probe
    regions blend slightly; treat this as a qualitative identity view.
    '''
    from glycographer.colors import probe_palette

    pal = probe_palette(list(probe_labels))
    color_names, values = [], []
    for i, label in enumerate(probe_labels, start=1):
        cname = f'probe_{label}'
        cmd.set_color(cname, list(pal[label]))
        color_names.append(cname)
        values.append(i)

    ramp_name = f'ramp_best_probe_{best_probe_map}'
    cmd.ramp_new(ramp_name, best_probe_map, values, color_names)
    cmd.set('surface_color', ramp_name, rec_name)
    cmd.set('surface_ramp_above_mode', 0)
    cmd.set('surface_quality', 1)
    cmd.show('surface', rec_name)

    return ramp_name

@cmd.extend
def extract_pose_clusters(ens_name: str, scoredata:str):
    '''
    Extract clusters of poses contained in a scoredata file
    from a PyMOL-loaded multistate ensemble and display in a
    PyMOL visualization session.
    '''
    df = pd.read_csv(scoredata)

    if 'model_num' not in df.columns or 'cluster_id' not in df.columns:
        print("Error: CSV file must contain 'model_num' and 'cluster_id' columns")
        return
    
    n_states = cmd.count_states(ens_name)
    if n_states != df['model_num'].max():
        print(f"Warning: Ensemble has {n_states} states but scoredata contains {df['model_num'].max()} model samples")

    cluster_ids = df['cluster_id'].dropna().unique()
    if len(cluster_ids) == 0:
        print("No clusters found in the scoredata file")
        return
    
    cluster_dict = {}
    for cluster_id in cluster_ids:
        if pd.isna(cluster_id) or cluster_id == '':
            continue

        cluster_name = f'cluster_{int(cluster_id)}'
        cluster_models = df[df['cluster_id'] == cluster_id]['model_num'].tolist()

        cluster_dict[cluster_name] = cluster_models
        for model in cluster_models:
            cmd.create(cluster_name,
                       ens_name,
                       source_state=int(model),
                       target_state=-1)
            
    print('Cluster: Models')
    for key, val in cluster_dict.items():
        print(f'{key}: {val}')

    return cluster_dict