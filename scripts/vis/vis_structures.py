#!/usr/bin/env python3

'''
Visualize ensemble / map data from an input file containing the name
and desired base color scheme of each ensemble / map using pymol.
'''

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

from typing import Dict
import sys
import re
import os

import pymol
from pymol import cmd
import numpy as np

from glycographer.map import VolMap
from glycographer.vis import (
    format_background,
    vis_receptor,
    vis_grid,
    vis_crystal_ligand,
    load_volmap_from_dx,
    draw_contour,
    draw_mapped_surface,
    color_by_magnitude
)

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
        maptype = 'intengmin' if 'intengmin' in parts else 'count'
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
    parser.add_argument('-grid', '--grid', type=str, required=False,
                        help='Option to visualize the sampling grid generated above the receptor surface.')
    parser.add_argument('-lig', '--ligand-pose', type=str, required=False,
                        help='Glycoligand pose to display for reference.')
    parser.add_argument('-lvls', '--isolevels', nargs='+', required=False,
                        help='Option to manually specify values at which to plot isocontours if known prior (determined internally if not specified).')
    parser.add_argument('-nlvls', '--n-isolevels', type=int, default=5,
                        help='Number of isolevels to plot for each map if discrete values are not given (default: 5).')
    parser.add_argument('-o', '--outprefix', type=str, default='map_vis',
                        help='Prefix to name each output file.')
    parser.add_argument('--bg-rgb', type=str, default='white',
                        help='Output visualization background color (Default: white).')

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
    
    # Load and visualize the grid if provided:
    if args.grid:
        try:
            print(f'Adding sampling grid: {args.grid}')
            vis_grid(args.grid)
        except Exception as e:
            print(f'Unable to load and visualize grid {args.grid}: {e}')
            import traceback
            traceback.print_exc()
            return 1

    # Load and visualize a ligand pose if provided:
    if args.ligand_pose:
        try:
            print(f'Adding ligand: {args.ligand_pose}')
            vis_crystal_ligand(args.ligand_pose)
        except Exception as e:
            print(f'Unable to visualize crystal ligand {args.ligand_pose}: {e}')
            import traceback
            traceback.print_exc()
            return 1
        
    # Load and visualize each volmap:
    for file, color in configdat.items():
        file = os.path.join(args.path, file)

        # Extract map info from filename:
        structdat = get_data_from_filename(file)
        
        # Import the map data as a class instance for processing:
        volmap = VolMap.from_dx(file)

        # Load the map into PyMOL:
        map_name_pml = load_volmap_from_dx(file)

        # Extract data for visualizing:
        occupied = volmap.values[volmap.values != 0]
        map_min = np.min(occupied) if occupied.any() else None
        map_max = np.max(occupied) if occupied.any() else None

        if not args.isolevels:
            # Plot n most significant isovals:
            map_avg = np.mean(occupied) if occupied.any() else None
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
            contour_name = draw_contour(map_name_pml, lvl)
            
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