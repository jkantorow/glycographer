#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import pymol
from pymol import cmd
import argparse
import sys
import time
from typing import Optional, Tuple, List

'''
Use 3d mesh grid sampling techniques to generate a uniform set
of coordinates within n Angstroms of the surface of a receptor.

Designed to be adaptive to binding energy hotspots and arbitrary
binding sites.
'''

def get_surface_samples(surface_stl, grid_res=1.0, bb_min=None, bb_max=None,
                        surf_shell_min=1.0, surf_shell_max=4.0, padding=5.0,
                        chunk_size=100000, filter_cryptic=False, cryptic_radius=4.0):
    '''
    Generate a cartesian grid of sampling points within a shell above a
    putative binding surface.
    '''
    mesh = o3d.io.read_triangle_mesh(surface_stl)

    # Extract gridbox bounds:
    if bb_min and bb_max:
        xmin, ymin, zmin = bb_min[0], bb_min[1], bb_min[2]
        xmax, ymax, zmax = bb_max[0], bb_max[1], bb_max[2]
    else:
        # Generate sampling points across entire surface:
        xmin, ymin, zmin = mesh.get_min_bound()
        xmax, ymax, zmax = mesh.get_max_bound()

        xmin -= padding
        ymin -= padding
        zmin -= padding
        xmax += padding
        ymax += padding
        zmax += padding

        print(f"Using mesh bounds with padding: [{xmin:.2f}, {xmax:.2f}] x [{ymin:.2f}, {ymax:.2f}] x [{zmin:.2f}, {zmax:.2f}]")

    # Generate grid coordinates
    xs = np.arange(xmin, xmax, grid_res)
    ys = np.arange(ymin, ymax, grid_res)
    zs = np.arange(zmin, zmax, grid_res)
    
    total_points = len(xs) * len(ys) * len(zs)
    print(f"Generating {total_points:,} grid points...")
    
    # Create raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    
    # Process in chunks to manage memory
    filtered_points = []
    
    for i in range(0, len(zs), max(1, chunk_size // (len(xs) * len(ys)))):
        z_chunk = zs[i:i + max(1, chunk_size // (len(xs) * len(ys)))]
        
        X, Y, Z = np.meshgrid(xs, ys, z_chunk, indexing='ij')
        chunk_pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        
        # Compute signed distances
        query_pts = o3d.core.Tensor(chunk_pts.astype(np.float32))
        signed_d = scene.compute_signed_distance(query_pts).numpy()
        
        # Filter points within surface shell (surf_min < distance < surf_cutoff)
        mask = (signed_d > surf_shell_min) & (signed_d < surf_shell_max)
        filtered_chunk = chunk_pts[mask]
        
        # Apply cryptic pocket filtering if requested
        if filter_cryptic and len(filtered_chunk) > 0:
            filtered_chunk = _filter_cryptic_pockets(filtered_chunk, scene, cryptic_radius)
        
        if len(filtered_chunk) > 0:
            filtered_points.append(filtered_chunk)
        
        if i % 10 == 0:
            progress = min(100, (i / len(zs)) * 100)
            print(f"Processing chunk {i//max(1, chunk_size // (len(xs) * len(ys))) + 1}, progress: {progress:.1f}%")
    
    if filtered_points:
        result = np.vstack(filtered_points)
        print(f"Found {len(result):,} surface points within a shell between {surf_shell_min} and {surf_shell_max}Å above receptor surface")
        return result
    else:
        print("No points found within the specified criteria")
        return np.array([]).reshape(0, 3)

def _filter_cryptic_pockets(points, scene, cryptic_radius):
    """
    Filter out points that are in cryptic pockets (enclosed within surface cavities).
    
    Uses raycasting in multiple directions to detect if a point is accessible
    from the outside surface. Points that can't be reached without hitting
    the surface are considered cryptic and filtered out.
    
    Args:
        points: Array of 3D coordinates to filter
        scene: Open3D raycasting scene
        cryptic_radius: Radius for accessibility testing
    
    Returns:
        Filtered array of accessible points
    """
    if len(points) == 0:
        return points
    
    # Generate rays in multiple directions from each point outward
    directions = np.array([
        [1, 0, 0], [-1, 0, 0],  # X axis
        [0, 1, 0], [0, -1, 0],  # Y axis  
        [0, 0, 1], [0, 0, -1],  # Z axis
        [1, 1, 1], [-1, -1, -1],  # Diagonal
        [1, -1, 1], [-1, 1, -1],  # Other diagonals
        [1, 1, -1], [-1, -1, 1]
    ])
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    accessible_mask = np.zeros(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        accessible_count = 0
        
        for direction in directions:
            # Cast ray from point outward
            ray_origin = point
            ray_direction = direction
            
            # Create ray
            rays = o3d.core.Tensor([ray_origin.tolist() + ray_direction.tolist()], 
                                  dtype=o3d.core.float32)
            
            # Check if ray hits surface within cryptic_radius
            result = scene.cast_rays(rays)
            distances = result['t_hit'].numpy()
            
            # If ray travels further than cryptic_radius without hitting surface,
            # this direction is accessible
            if len(distances) == 0 or distances[0] > cryptic_radius:
                accessible_count += 1
        
        # Point is accessible if at least 3 directions are clear
        # (allows for some surface contact while filtering deep pockets)
        accessible_mask[i] = accessible_count >= 3
    
    filtered_points = points[accessible_mask]
    removed_count = len(points) - len(filtered_points)
    
    if removed_count > 0:
        print(f"  Filtered out {removed_count} cryptic pocket points")
    
    return filtered_points

def save_points_to_xyz(points, filename, comments=[]):
    """Save points to XYZ format for visualization."""
    with open(filename, 'w') as f:
        f.write(f"{len(points)}\n")
        for comment in comments:
            f.write(f"{comment}\n")
        for point in points:
            f.write(f"X {point[0]:.3f} {point[1]:.3f} {point[2]:.3f}\n")
    print(f"Saved {len(points)} points to {filename}")

def save_points_to_pdb(points, filename, comments=[]):
    """Save points as pseudoatoms in PDB format for Rosetta StartFrom mover."""
    with open(filename, 'w') as f:
        f.write(f"HEADER    GLYCOGRAPHER GRIDBOX\n")
        f.write(f"REMARK    Generated surface sampling points: {len(points)}\n")
        for comment in comments:
            f.write(f"REMARK    {comment}\n")        
        for i, point in enumerate(points, 1):
            # Format as ATOM record with chain P, oxygen atoms
            f.write(f"ATOM  {i:5d}  O   PSD P{i:4d}    "
                   f"{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}"
                   f"  1.00 20.00           O  \n")
        
        f.write("END\n")
    print(f"Saved {len(points)} points to PDB file: {filename}")


def generate_stl_from_pymol(pdb_file, output_stl=None, selection="all", 
                           surface_quality=1, surface_solvent=True):
    """
    Generate STL file from PDB using PyMOL with correct coordinate frame.
    
    Args:
        pdb_file: Input PDB file
        output_stl: Output STL filename (default: input_surface.stl)
        selection: PyMOL selection for surface generation
        surface_quality: Surface quality (0-4, higher is better)
        surface_solvent: Whether to generate solvent-accessible surface
    
    Returns:
        Path to generated STL file
    """
    if output_stl is None:
        output_stl = pdb_file.replace('.pdb', '_surface.stl')
    
    try:
        # Initialize PyMOL in command-line mode
        pymol.finish_launching(['pymol', '-c'])
        
        # Load the structure
        cmd.load(pdb_file, "protein")
        
        # Set surface quality
        cmd.set("surface_quality", surface_quality)
        
        # Generate surface
        if surface_solvent:
            cmd.set("surface_mode", 1)  # Solvent accessible surface
        else:
            cmd.set("surface_mode", 0)  # Van der Waals surface
            
        cmd.show("surface", selection)
        
        # Important: Reset view to ensure coordinates are in model frame
        cmd.reset()
        cmd.center(selection)
        
        # Export as STL using model coordinates
        cmd.save(output_stl, f"({selection}) and surface", format="stl")
        
        cmd.delete("all")
        print(f"Generated STL file: {output_stl}")
        
        return output_stl
        
    except Exception as e:
        print(f"Error generating STL with PyMOL: {e}")
        print("Consider using ChimeraX for STL generation as backup")
        return None
    
def main():

    parser = argparse.ArgumentParser(
        description='Generate grid of sampling points along an input putative receptor surface.')
    parser.add_argument('receptor', type=str,
                        help='Receptor structure from which to generate sampling points. (PDB, STL)')
    parser.add_argument('-res', '--gridres', type=float, default=2.0,
                        help='Uniform spacing between each sampling point in Angstroms (default: 1.0)')
    parser.add_argument('--bb-min', type=float, nargs=3, metavar=('xmin', 'ymin', 'zmin'),
                        help='Minimum vertices for sampling bounding box.')
    parser.add_argument('--bb-max', type=float, nargs=3, metavar=('xmax', 'ymax', 'zmax'),
                        help='Maximum vertices for sampling bounding box.')
    parser.add_argument('--shell-min', type=float, default=1.0,
                        help='Lower surface shell bound in which to populate sampling points in Angstroms above the surface (default: 1.0)')
    parser.add_argument('--shell-max', type=float, default=4.0,
                        help='Upper surface shell bound in which to populate sampling points in Angstroms above the surface (default: 4.0)')
    parser.add_argument('--padding', type=float, default=1.0,
                        help='Padding distance to include around specified gridbox boundaries in Angstroms (default: 5.0)')
    parser.add_argument('--filter-cryptic', action='store_true',
                        help='Filter out sampling points embedded into pockets too small for glycan residues to fit.')
    parser.add_argument('--cryptic-radius', type=float, default=4.0,
                        help='Radius defining cryptic pocket size in Angstroms (default: 4.0)')
    parser.add_argument('--chunk-size', type=int, default=100000,
                        help='Chunk size for memory management during processing (default: 100000)')
    parser.add_argument('-o', '--outprefix', type=str, required=False,
                        help='Output file prefix.')
    parser.add_argument('--write-xyz', action='store_true',
                        help='Output sampling points as a text file in xyz format.')
    parser.add_argument('--write-pdb', action='store_true', default=True,
                        help='Output sampling points as pseudoatoms in pdb format.')
    
    args = parser.parse_args()

    if not args.bb_min and args.bb_max:
        parser.error('Both maximum and minimum vertices must be specified to define a custom gridbox area (the entire receptor will be considered if neither are specified).')

    # Generate stl file of receptor surface if PDB is provided:
    if str(args.receptor).endswith('.pdb'):
        recname = str(args.receptor).replace('.pdb', '')
        stlfile = generate_stl_from_pymol(args.receptor)
    elif str(args.receptor).endswith('.stl'):
        recname = str(args.receptor).replace('.stl', '')
        recname = recname.replace('_surface', '')
        stlfile = args.receptor
    else:
        raise ValueError(f'Receptor input structure type not recognized: {args.receptor}')

    # Generate sampling points:
    points = get_surface_samples(
        stlfile,
        grid_res=args.gridres,
        bb_min=args.bb_min,
        bb_max=args.bb_max,
        surf_shell_min=args.shell_min,
        surf_shell_max=args.shell_max,
        padding=args.padding,
        chunk_size=args.chunk_size,
        filter_cryptic=args.filter_cryptic,
        cryptic_radius=args.cryptic_radius
    )

    if args.outprefix:
        outfilename = f'{args.outprefix}_gridbox_res{str(args.gridres).replace('.', 'p')}_sh{str(args.shell_min).replace('.', 'p')}-{str(args.shell_max).replace('.', 'p')}_pad{str(args.padding).replace('.', 'p')}'
    else:
        outfilename = f'{recname}_gridbox_res{str(args.gridres).replace('.', 'p')}_sh{str(args.shell_min).replace('.', 'p')}-{str(args.shell_max).replace('.', 'p')}_pad{str(args.padding).replace('.', 'p')}'

    # Append gridbox parameters as comments included in output file:
    comments = [
        f'receptor {args.receptor}',
        f'grid_res {args.gridres}',
        f'bb_min {args.bb_min}',
        f'bb_max {args.bb_max}',
        f'shell_min {args.shell_min}',
        f'shell_max {args.shell_max}',
        f'padding {args.padding}',
        f'chunk_size {args.chunk_size}',
        f'filter_cryptic {args.filter_cryptic if args.filter_cryptic else False}',
        f'cryptic_radius {args.cryptic_radius if args.filter_cryptic else None}'
    ]

    if len(points) > 0:
        if args.write_xyz:
            save_points_to_xyz(points, f'{outfilename}.xyz', comments=comments)
        if args.write_pdb:
            save_points_to_pdb(points, f'{outfilename}.pdb', comments=comments)
    else:
        print('No sampling points generated within gridbox: Check input parameters.')
        sys.exit(1)

if __name__ == '__main__':
    exit(main())