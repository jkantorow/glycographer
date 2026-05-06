#!/usr/bin/env python3

'''
Visualize ensemble / map data from an input file containing the name
and desired base color scheme of each ensemble / map using pymol.
'''

from typing import List, Dict, Optional, Tuple
import pymol
from pymol import cmd
import numpy as np
import sys
import re
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

from scripts.vis.glycolors import (
    get_snfg_color,
    color_by_magnitude,
    glycolor_by_magnitude,
    atomcolor_by_magnitude
)
from glycographer.mapping import VolMap

def parse_config(configfile: str) -> Dict:
    ''' Get plotting data from input config file. '''
    configdat = {}

    with open(configfile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            configdat[parts[0]] = parts[1]
    
    return configdat

def get_data_from_filename(filename: str) -> Dict:
    ''' Extract structure metadata from the filename. '''
    gly_pdbs = ('MAN', 'BMA', 'GLC', 'GAL', 'A2G', 'GUL', 'NAG',
                'NDG', 'FUC', 'SIA', 'XYL', 'ALL', 'ALT', 'RHA',
                'ARA', 'RIB')
    gly_elems = ('O', '1O', '2O', 'C', 'N')
    
    structdat = {}

    path, ext = os.path.splitext(filename)
    basename = os.path.basename(path)
    parts = basename.split('_')
    rec, lig = parts[0].split('-')
    lig_parts = re.findall(r'[a-zA-z]+|[0-9]+', lig)
    resnames = [item.upper() for item in lig_parts if item.upper() in gly_pdbs]
    
    if ext == '.dx':
        maptype = 'energy' if 'energy' in parts else 'count'
        atomname = parts[2] if parts[2].startswith(gly_elems) else None
    else:
        maptype = atomname = None

    structdat['basename'] = basename
    structdat['recname'] = rec
    structdat['ligname'] = lig
    structdat['resnames'] = resnames
    structdat['maptype'] = maptype
    structdat['atomname'] = atomname

    return structdat

def format_background():
    '''
    Standardize the viewing environment for the output in pymol or vmd.
    '''
    cmd.set('bg_rgb', 'white')
    cmd.set('depth_cue', 0)
    cmd.set('ray_shadow', 0)
    cmd.set('surface_quality', 2)

def vis_receptor(receptor_pdb: str):
    '''
    Load and display receptor as grey surface:
    '''
    cmd.load(receptor_pdb, 'rec')
    cmd.hide('cartoon', 'rec')
    cmd.show('surface', 'rec')
    cmd.color('grey80', 'rec')

def vis_crystal_ligand(lig_pdb: str, mode: str = 'pymol'):
    '''
    Load and display known crystal glycoligand pose as licorice
    representation colored by residue in standard SNFG format.
    '''
    lig_name = cmd.get_unused_name('crystal_lig_')
    cmd.load(lig_pdb, lig_name)
    cmd.show('sticks', lig_name)

    # Get all unique residue names:
    from pymol import stored
    stored.res_types = []
    cmd.iterate(f'{lig_name} and name c1', 'stored.res_types.append(resn)')
    unique_res = set(stored.res_types)

    # Color by SNFG standard:
    for res in unique_res:
        res_color = get_snfg_color(res, mode)
        cmd.color(res_color, f'{lig_name} and resn {res}')

def load_volmap_pymol(map_dx: str) -> str:
    '''
    Load a volume map file as a pymol object.
    '''
    map_name = os.path.basename(map_dx).replace('.dx', '')
    cmd.load(map_dx, map_name)

    return map_name

def draw_isocontour_pymol(map_name: str, level: float = -1.0,
                          style: str = 'mesh') -> str:
    '''
    Draw an isosurface, isomesh, or isodot contour at
    a specified isolevel from an input pymol map object.
    '''
    contour_name = '_'.join([map_name, str(np.round(level, 3))])

    if style == 'mesh':
        cmd.isomesh(contour_name, map_name, level)
    elif style == 'dots':
        cmd.isodot(contour_name, map_name, level)
    elif style == 'surface':
        cmd.isosurface(contour_name, map_name, level)
        cmd.set('transparency', 0.5, contour_name)

    return contour_name

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize output glycan binding affinity maps in the context' \
        'of the receptor target and any known crystal ligand poses.'
    )

    parser.add_argument('config', type=str,
                        help='Input file containing each structure file and color scheme for plotting.')
    parser.add_argument('-path', '--path', type=str, default='.',
                        help='Path to where structure files in config are stored.')
    parser.add_argument('-rec', '--receptor', type=str, required=True,
                        help='Receptor structure in pdb format around which the volume files were generated.')
    parser.add_argument('-lig', '--crystal-ligand', type=str, required=False,
                        help='Crystal glycoligand pose to display for reference if one is known.')
    parser.add_argument('-lvls', '--isolevels', nargs='+', required=False,
                        help='Option to manually specify values at which to plot isocontours if known prior (determined internally if not specified).')
    parser.add_argument('-nlvls', '--n-isolevels', type=int, default=5,
                        help='Number of isolevels to plot for each map if discrete values are not given (default: 5).')
    parser.add_argument('--style', type=str, choices=['mesh', 'dot', 'surface'], default='mesh',
                        help='Draw style in which to show the isocontours (default: mesh).')
    parser.add_argument('-o', '--outprefix', type=str, default='map_vis',
                        help='Prefix to name each output file.')

    args = parser.parse_args()

    configdat = parse_config(args.config)
        
    pymol.finish_launching(['pymol', '-c'])

    # Format GUI environment:
    format_background()

    # Load and visualize receptor:
    try:
        print(f'Adding receptor: {args.receptor}')
        vis_receptor(args.receptor)
    except Exception as e:
        print(f'Unable to visualize receptor {args.receptor}: {e}')
        import traceback
        traceback.print_exc()
        return 1
    
    # Load and visualize ligand if provided:
    if args.crystal_ligand:
        try:
            print(f'Adding crystal ligand: {args.crystal_ligand}')
            vis_crystal_ligand(args.crystal_ligand)
        except Exception as e:
            print(f'Unable to visualize crystal ligand {args.crystal_ligand}: {e}')
            import traceback
            traceback.print_exc()
            return 1
        
    # Load and visualize each volmap:
    for file, color in configdat.items():
        file = os.path.join(args.path, file)

        # Extract map info from filename:
        structdat = get_data_from_filename(file)
        
        # Import the map data as a class instance for processing:
        volmap = VolMap(
            name=structdat['basename'],
            recname=structdat['recname'],
            maptype=structdat['maptype'],
            resnames=structdat['resnames']
        )
        volmap.read_dx(file)

        # Also, import the map as a pymol object:
        map_name_pml = load_volmap_pymol(file)

        # Extract data for visualizing:
        map_min = volmap.get_min_voxel_val()
        map_max = volmap.get_max_voxel_val()

        if not args.isolevels:
            # Plot n most significant isovals:
            map_avg = volmap.get_avg_voxel_val()
            if map_avg < 0:
                levels = np.linspace(map_min, map_avg, args.n_isolevels)
                lvl_range = [map_min, map_avg]
                negative_is_better = True
            elif map_avg > 0:
                levels = np.linspace(map_avg, map_max, args.n_isolevels)
                lvl_range = [map_avg, map_max]
                negative_is_better = False
            else:
                print('Warning: Average of all voxel values is 0.')
        else:
            levels = args.isolevels

        for lvl in levels:
            # Draw isocontours for each isolevel:
            contour_name = draw_isocontour_pymol(map_name=map_name_pml, level=lvl, style=args.style)
            
            # Color each isocontour by level magnitude:
            scaled_colorname = f'{structdat['ligname']}_magcol_{np.round(lvl, 3)}'
            scaled_rgb = color_by_magnitude(color, lvl, lvl_range, negative_is_better)
            cmd.set_color(scaled_colorname, scaled_rgb)
            cmd.color(scaled_colorname, contour_name)

    session_file = f'{args.outprefix}.pse'
    cmd.save(session_file)
    print(f'PyMOL visualization saved to {os.path.abspath(session_file)}')
    cmd.quit()

if __name__ == '__main__':
    exit(main())