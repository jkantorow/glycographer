#!/usr/bin/env python3

'''
Main script for outputting a pymol and/or vmd visualization file
of input glycan volume maps and the target receptor they describe.

Also allows for visualizing any available input glycan crystal pose
for map validation and quality control.
'''

from typing import List, Dict, Optional, Tuple
import pymol
from pymol import cmd
import numpy as np
import subprocess
import argparse
import glob
import sys
import os

# Add the parent directory of 'scripts' to sys.path for custom imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

from scripts.vis.glycolors import (
    get_snfg_color,
    glycolor_by_magnitude,
    atomcolor_by_magnitude
)
from dat.mapping import VolMap

def map_data_from_filename(filename: str) -> Dict:
    '''
    Extract map metadata from its filename
    '''
    gly_pdbs = ('MAN', 'BMA', 'GLC', 'GAL', 'A2G', 'GUL', 'NAG',
                'NDG', 'FUC', 'SIA', 'XYL', 'ALL', 'ALT', 'RHA',
                'ARA', 'RIB')
    gly_elems = ('O', '1O', '2O', 'C', 'N')
    map_types = ('energy', 'count')

    map_metadata = {}

    basename = os.path.basename(os.path.splitext(filename)[0]).replace('-', '_')
    parsed = basename.split('_')

    resnames = [item.upper() for item in parsed if item.upper() in gly_pdbs]
    maptype = [item for item in parsed if item in map_types]
    if len(maptype) > 1:
        raise Warning(f'Warning: multiple maptype keywords detected in {os.path.abspath(filename)}.')
    atomname = [item for item in parsed if item.startswith(gly_elems)] if maptype[0] == 'count' else None

    map_metadata['name'] = parsed[0]
    map_metadata['resnames'] = resnames if len(resnames) > 0 else None
    map_metadata['maptype'] = maptype[0] if len(maptype) > 0 else None
    map_metadata['atomname'] = atomname[0] if maptype[0] == 'count' else None

    return map_metadata

def format_background(mode: str = 'pymol', tcl_file: Optional[str] = None):
    '''
    Standardize the viewing environment for the output in pymol or vmd.
    '''
    if mode == 'pymol':
        cmd.set('bg_rgb', 'white')
        cmd.set('depth_cue', 0)
    elif mode == 'vmd':
        if tcl_file is None:
            tcl_file = os.path.join(os.path.dirname(__file__), 'format_background_vmd.tcl')
        subprocess.run(['vmd', '-dispdev', 'text', '-e', tcl_file])

def vis_receptor(receptor_pdb: str, mode: str = 'pymol',
                 tcl_file: Optional[str] = None):
    '''
    Load and display receptor as grey surface:
    '''
    if mode == 'pymol':
        cmd.load(receptor_pdb, 'rec')
        cmd.hide('cartoon', 'rec')
        cmd.show('surface', 'rec')
        cmd.color('grey80', 'rec')
    
    elif mode == 'vmd':
        if tcl_file is None:
            tcl_file = os.path.join(os.path.dirname(__file__), 'vis_receptor_vmd.tcl')
        subprocess.run(['vmd', '-dispdev', 'text', '-e',
                        tcl_file, '-args', receptor_pdb])
        
def vis_crystal_ligand(lig_pdb: str, mode: str = 'pymol', 
                       tcl_file: Optional[str] = None):
    '''
    Load and display known crystal glycoligand pose as licorice
    representation colored by residue in standard SNFG format.
    '''
    if mode == 'pymol':
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

    elif mode == 'vmd':
        if tcl_file is None:
            tcl_file = os.path.join(os.path.dirname(__file__), 'vis_crystal_ligand_vmd.tcl')
        subprocess.run(['vmd', '-dispdev', 'text', '-e',
                        tcl_file, '-args', lig_pdb])

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

# Function for drawing multiple isocontours?
    # At what point should I seriously consider
    # translating this into class/generator
    # objects?

# Separate vmd functionality into its own script?


def main():

    parser = argparse.ArgumentParser(
        description='Visualize output glycan binding affinity maps in the context' \
        'of the receptor target and any known crystal ligand poses.'
    )

    parser.add_argument('map_files', type=str,
                        help='Glob pattern for each dx volume file to visualize.')
    parser.add_argument('-rec', '--receptor', type=str, required=True,
                        help='Receptor structure in pdb format around which the volume files were generated.')
    parser.add_argument('-lig', '--crystal-ligand', type=str, required=False,
                        help='Crystal glycoligand pose to display for reference if one is known.')
    parser.add_argument('-lvls', '--isolevels', nargs='+', required=False,
                        help='Option to manually specify values at which to plot isocontours if known prior (determined internally if not specified).')
    parser.add_argument('--nlvls', type=int, default=5,
                        help='Number of isolevels to plot for each map if discrete values are not given (default: 5).')
    parser.add_argument('--style', type=str, choices=['mesh', 'dot', 'surface'], default='mesh',
                        help='Draw style in which to show the isocontours (default: mesh).')
    parser.add_argument('-o', '--outprefix', type=str, default='map_vis',
                        help='Prefix to name each output file.')

    args = parser.parse_args()

    map_files = sorted(glob.glob(args.map_files))
    if not map_files:
        print(f'No map files found matching pattern {args.map_files}')
        return 1
    
    recname = os.path.basename(os.path.splitext(args.receptor)[0])
        
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
    for file in map_files:

        # Extract map info from filename:
        map_metadata = map_data_from_filename(file)
        
        # Import the map data as a class instance for processing:
        volmap = VolMap(
            name=map_metadata['name'],
            recname=recname,
            maptype=map_metadata['maptype'],
            resnames=map_metadata['resnames']
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
                levels = np.linspace(map_min, map_avg, args.nlvls)
                lvl_range = [map_min, map_avg]
                negative_is_better = True
            elif map_avg > 0:
                levels = np.linspace(map_avg, map_max, args.nlvls)
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
            if map_metadata['maptype'] == 'energy':
                colortag = map_metadata['resnames'][0] if map_metadata['resnames'] else map_metadata['name']
                scaled_colorname = f'{colortag}_colorbymag_{np.round(lvl, 3)}'
                scaled_rgb = glycolor_by_magnitude(resname=map_metadata['resnames'][0], score=lvl,
                                                   score_range=lvl_range, negative_is_better=negative_is_better)
            elif map_metadata['maptype'] == 'count':
                colortag = map_metadata['atomname'] if map_metadata['atomname'] else map_metadata['name']
                scaled_colorname = f'{colortag}_colorbymag_{np.round(lvl, 3)}'
                elem = [char.upper() for char in map_metadata['atomname'] if char.isalpha()][0]
                scaled_rgb = atomcolor_by_magnitude(element=elem, score=lvl, score_range=lvl_range,
                                                    negative_is_better=negative_is_better)
            cmd.set_color(scaled_colorname, scaled_rgb)
            cmd.color(scaled_colorname, contour_name)

    session_file = f'{args.outprefix}.pse'
    cmd.save(session_file)
    print(f'PyMOL visualization saved to {os.path.abspath(session_file)}')
    cmd.quit()

if __name__ == '__main__':
    exit(main())
            




