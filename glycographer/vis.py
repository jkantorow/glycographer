'''
Functions for creating and rendering volume map visualizations in PyMOL.

Each function can also be imported and used within a PyMOL session via the
PyMOL run command:
> run /path/to/glycographer/vis.py
'''

import colorsys
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from pymol import cmd
from pymol import stored

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
    Load and display receptor as grey surface:
    '''
    rec_name = os.path.basename(receptor_pdb).replace('.pdb', '')
    cmd.load(receptor_pdb, rec_name)
    cmd.hide('cartoon', rec_name)
    cmd.show('surface', rec_name)
    cmd.color('grey80', rec_name)

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
        cmd.color(snfg_colors[res], f'{lig_name} and resn {res}')

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
def draw_contour(map_name: str, level: float):
    '''
    Draw an isomesh contour from a volume density map at a specified level.
    '''
    contour_name = '_'.join(['contour', map_name, str(level)])
    cmd.isomesh(contour_name, map_name, level)

    return contour_name

@cmd.extend
def draw_mapped_surface(rec_name: str, map_name: str,
                        ramp_min=0, ramp_max=1,
                        color_min='white', color_max='red'):
    '''
    Project a mapped molecular surface of probe score density onto the receptor.
    '''
    ramp_name = f'ramp_{ramp_min}_{ramp_max}_{map_name}'
    cmd.ramp_new(ramp_name, map_name, f'[{ramp_min}, {ramp_max}]', f'[{color_min}, {color_max}]')

    cmd.set('surface_color', ramp_name, rec_name)
    cmd.set('surface_ramp_above_mode', 1)
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

#### SNFG Coloring ####

snfg_colors = {
    "MAN": "green",
    "BMA": "green", 
    "GLC": "blue",
    "GAL": "yellow",
    "A2G": "yellow",
    "GUL": "orange",
    "NAG": "blue",
    "NDG": "blue",
    "FUC": "red",
    "SIA": "magenta",
    "XYL": "orange",
    "ALL": "purple",
    "ALT": "pink",
    "RHA": "pink",
    "ARA": "green",
    "RIB": "blue"
}

# Note: coloring by ramp would be much simpler
def color_by_magnitude(base_color: str, score: float, score_range, 
                       negative_is_better: bool = True) -> Tuple[float, float, float]:
    '''
    Return the rgb values of a residue name's SNFG base color
    scaled by saturation/lightness depending on a corresponding
    input score value's position within a score range.
    '''
    # Take the pymol color name and translate it into hls:
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Scale the saturation value by the relative score magnitude:
    score_min = score_range[0]
    score_max = score_range[1]
    ratio = (score_max - score) / (score_max - score_min)

    if negative_is_better:
        s = s * ratio
        l = 0.90 - (l * ratio)
    else:
        s = s * (1 - ratio)
        l = 0.10 + (l * ratio)
        
    # Convert the saturation-scaled color back to rgb format:
    r_scaled, g_scaled, b_scaled = colorsys.hls_to_rgb(h, l, s)

    return (r_scaled, g_scaled, b_scaled)