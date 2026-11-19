#!/usr/bin/env python3

'''
Build the sampling grid GlycanDock starts probes from: a point cloud in a
shell above a receptor's surface, written as PSD pseudo-atoms on chain P.

This is a thin CLI over glycographer.grid. The whole surface, or any part of
it, can be sampled:

    # everything -- 1pmh_rec_grid_global.pdb
    build_sampling_grid.py 1pmh_rec.pdb

    # one site, named by PyMOL selection
    build_sampling_grid.py tetra.pdb -g crd \
        --site 'chain A and resn CA' --site-radius 14

    # two sites at once; nothing is sampled in between
    build_sampling_grid.py tetra.pdb -g crd-ad \
        --site 'chain A and resn CA' --site 'chain D and resn CA'

    # an explicit box, as the old build_gridbox.py took
    build_sampling_grid.py rec.pdb -g cleft --box -4 -4 -4 38 40 54

The grid file is <receptor>_grid_<grid_id>.pdb, and <grid_id> becomes the
trailing field of every run_id built from it (<receptor>-<ligand>_<grid_id>),
so it must be letters, digits and '-' only. A JSON sidecar records every
parameter the grid was built with.

Use --dry-run to check a selection and the size of the job without building
anything. Nothing here needs PyRosetta.
'''

import argparse
import os
import sys

# The HPC env is not guaranteed to have the editable install, so put the repo
# root (one directory up from <repo>/scripts/) on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from glycographer.grid import (
    DEFAULT_FIT_RADIUS, DEFAULT_PROBE_RADIUS, DEFAULT_SHELL, DEFAULT_SITE_RADIUS,
    DEFAULT_SPACING, DEFAULT_SURFACE_SELECTION, GridError, MAX_VOXELS,
    SamplingRegion, build_sampling_grid, grid_filename, load_receptor,
    plan_lattice, write_grid,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Build a GlycanDock sampling grid over a receptor surface.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''examples:
  # whole surface
  build_sampling_grid.py 1pmh_rec.pdb -o grids/

  # the Ca2+ carbohydrate-recognition domain of one chain
  build_sampling_grid.py tetra.pdb -g crd -o grids/ \\
      --site 'chain A and resn CA' --site-radius 14

  # two separated patches in one grid (nothing sampled between them)
  build_sampling_grid.py tetra.pdb -g crd-ad -o grids/ \\
      --site 'chain A and resn CA' --site 'chain D and resn CA'

  # a known cleft, by residue
  build_sampling_grid.py 1pmh_rec.pdb -g cleft -o grids/ \\
      --site 'chain A and resi 110+112+135+157' --site-radius 12

  # check what a selection matches, without building
  build_sampling_grid.py tetra.pdb --site 'chain A and resn CA' --dry-run
''')

    parser.add_argument('receptor', type=str,
                        help='Receptor PDB. Use the prepared receptor (the one '
                             'that goes into docking): prepare_complexes.py '
                             'renames chains and may renumber residues.')

    where = parser.add_argument_group(
        'where to sample',
        'With none of these, the whole receptor surface is sampled. A --site '
        'and a --box can be combined; a point must then satisfy both.')
    where.add_argument('--site', type=str, action='append', default=None,
                       metavar='SEL',
                       help='PyMOL selection of receptor atoms to sample '
                            'around, e.g. \'chain A and resi 365+366\'. Repeat '
                            'for several sites: the kept volume is the union '
                            'of spheres around every selected atom, so distant '
                            'sites give disjoint patches with nothing sampled '
                            'in between.')
    where.add_argument('--site-radius', type=float, default=DEFAULT_SITE_RADIUS,
                       metavar='R',
                       help=f'How far from the --site atoms to sample, in A '
                            f'(default: {DEFAULT_SITE_RADIUS:g}). Points are '
                            f'still confined to the surface shell.')
    where.add_argument('--box', type=float, nargs=6, default=None,
                       metavar=('XMIN', 'YMIN', 'ZMIN', 'XMAX', 'YMAX', 'ZMAX'),
                       help='Explicit bounding box, as the old build_gridbox.py '
                            '--bb-min/--bb-max took.')
    where.add_argument('--bb-min', type=float, nargs=3, default=None,
                       metavar=('XMIN', 'YMIN', 'ZMIN'),
                       help='Box minimum corner (use with --bb-max).')
    where.add_argument('--bb-max', type=float, nargs=3, default=None,
                       metavar=('XMAX', 'YMAX', 'ZMAX'),
                       help='Box maximum corner (use with --bb-min).')

    geom = parser.add_argument_group('grid geometry')
    geom.add_argument('-s', '--spacing', type=float, default=DEFAULT_SPACING,
                      help=f'Distance between neighbouring sampling points, in '
                           f'A (default: {DEFAULT_SPACING:g}).')
    geom.add_argument('--shell-min', type=float, default=DEFAULT_SHELL[0],
                      help=f'Lower edge of the sampling shell, in A above the '
                           f'molecular surface (default: {DEFAULT_SHELL[0]:g}).')
    geom.add_argument('--shell-max', type=float, default=DEFAULT_SHELL[1],
                      help=f'Upper edge of the sampling shell, in A above the '
                           f'molecular surface (default: {DEFAULT_SHELL[1]:g}). '
                           f'A grid point is where the ligand\'s centre of mass '
                           f'starts, not where it contacts.')
    geom.add_argument('--surface-sel', type=str,
                      default=DEFAULT_SURFACE_SELECTION, metavar='SEL',
                      help=f'PyMOL selection of the atoms that make up the '
                           f'surface (default: {DEFAULT_SURFACE_SELECTION!r}). '
                           f'Keep metal ions in: they are part of the binding '
                           f'site in a C-type lectin.')
    geom.add_argument('--probe-radius', type=float,
                      default=DEFAULT_PROBE_RADIUS,
                      help=f'Radius of the probe that rolls over the atoms to '
                           f'define the surface, in A (default: '
                           f'{DEFAULT_PROBE_RADIUS:g}, a water molecule).')
    geom.add_argument('--fit-radius', type=float, default=DEFAULT_FIT_RADIUS,
                      help=f'Drop points that only a sphere smaller than this '
                           f'can reach from bulk solvent -- narrow grooves and '
                           f'sealed cavities where no sugar ring fits, in A '
                           f'(default: {DEFAULT_FIT_RADIUS:g}).')
    geom.add_argument('--no-fit-filter', action='store_true',
                      help='Keep every point in the shell, however tight the '
                           'space (equivalent to --fit-radius 0).')
    geom.add_argument('--radius', type=str, action='append', default=None,
                      metavar='ELEM=R',
                      help='Override one element\'s radius, e.g. --radius '
                           'CA=2.31. Metal ions default to Shannon ionic radii; '
                           'see glycographer.grid.ELEMENT_RADII.')

    out = parser.add_argument_group('output')
    out.add_argument('-g', '--grid-id', type=str, default=None,
                     help='Name of this grid, used in the filename and as the '
                          'trailing field of every run_id built from it. '
                          'Letters, digits and "-" only. Defaults to "global" '
                          'when the whole surface is sampled; required '
                          'otherwise.')
    out.add_argument('-o', '--outdir', type=str, default='.',
                     help='Directory for the grid files (default: .).')
    out.add_argument('--out', type=str, default=None, metavar='PATH',
                     help='Write the grid PDB here instead of '
                          '<outdir>/<receptor>_grid_<grid_id>.pdb.')
    out.add_argument('--write-xyz', action='store_true',
                     help='Also write the points as an .xyz file.')
    out.add_argument('--no-manifest', action='store_true',
                     help='Skip the .json sidecar of build parameters.')
    out.add_argument('--overwrite', action='store_true',
                     help='Overwrite an existing grid file (default: refuse).')

    adv = parser.add_argument_group('advanced')
    adv.add_argument('--lattice-spacing', type=float, default=None, metavar='A',
                     help='Pitch of the lattice the surface geometry is '
                          'computed on; must divide --spacing. Default: the '
                          'largest divisor of --spacing that is <= 1 A.')
    adv.add_argument('--max-voxels', type=int, default=MAX_VOXELS,
                     help=f'Refuse a geometry lattice larger than this '
                          f'(default: {MAX_VOXELS:,}; ~50 bytes each).')
    adv.add_argument('--dry-run', action='store_true',
                     help='Report what the selections match and how big the '
                          'job is, then exit without building the grid.')
    return parser


def parse_radius_overrides(items, parser):
    out = {}
    for item in items or []:
        if '=' not in item:
            parser.error(f'--radius expects ELEM=VALUE, got {item!r}')
        elem, _, value = item.partition('=')
        try:
            out[elem.strip().upper()] = float(value)
        except ValueError:
            parser.error(f'--radius {item!r}: {value!r} is not a number')
    return out


def resolve_region(args, receptor, parser):
    '''Turn the --site/--box arguments into a SamplingRegion.'''
    box = None
    if args.box is not None:
        if args.bb_min or args.bb_max:
            parser.error('use either --box or --bb-min/--bb-max, not both.')
        box = (args.box[:3], args.box[3:])
    elif args.bb_min or args.bb_max:
        if not (args.bb_min and args.bb_max):
            parser.error('--bb-min and --bb-max must both be given.')
        box = (args.bb_min, args.bb_max)

    region = SamplingRegion()
    if args.site:
        coords = np.vstack([receptor.site_coords(sel) for sel in args.site])
        region = SamplingRegion.around(coords, args.site_radius,
                                       labels=args.site)
    if box is not None:
        region = region.with_box(*box)
    return region


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.receptor):
        parser.error(f'receptor not found: {args.receptor}')
    if args.no_fit_filter:
        args.fit_radius = None

    overrides = parse_radius_overrides(args.radius, parser)
    try:
        receptor = load_receptor(args.receptor, args.surface_sel,
                                 radius_overrides=overrides)
        region = resolve_region(args, receptor, parser)
    except GridError as e:
        raise SystemExit(f'error: {e}')

    grid_id = args.grid_id
    if grid_id is None:
        if not region.is_global:
            raise SystemExit(
                'error: --grid-id is required for a grid that covers only part '
                'of the receptor. It names the grid file and becomes the last '
                'field of every run_id built from it, so two grids over the '
                'same receptor must not share one (e.g. -g crd, -g cleft).')
        grid_id = 'global'

    try:
        out_path = args.out or os.path.join(
            args.outdir, grid_filename(receptor.stem, grid_id))
    except ValueError as e:
        raise SystemExit(f'error: {e}')
    if os.path.exists(out_path) and not args.overwrite and not args.dry_run:
        raise SystemExit(f'error: {out_path} already exists (--overwrite to '
                         f'replace it).')

    if args.dry_run:
        try:
            plans = plan_lattice(
                receptor, region, spacing=args.spacing,
                shell=(args.shell_min, args.shell_max),
                probe_radius=args.probe_radius, fit_radius=args.fit_radius,
                lattice_spacing=args.lattice_spacing)
        except GridError as e:
            raise SystemExit(f'error: {e}')
        print(f'Region: {region.describe()}')
        if len(plans) > 1:
            print(f'  {len(plans)} separated patches, one lattice each')
        for n, plan in enumerate(plans, 1):
            tag = f'  patch {n}:' if len(plans) > 1 else ' '
            nx, ny, nz = plan['shape']
            print(f'{tag} bounds  ' + '  '.join(
                f'{a:.1f}..{b:.1f}' for a, b in zip(plan['lo'], plan['hi'])))
            print(f'{tag} surface lattice {nx} x {ny} x {nz} = '
                  f'{plan["n_voxels"]:,} voxels at {plan["lattice_spacing"]:g} A'
                  + ('  (OVER --max-voxels)'
                     if plan['n_voxels'] > args.max_voxels else ''))
        print(f'  sampling nodes  {sum(p["n_nodes"] for p in plans):,} '
              f'before the shell and fit filters')
        print(f'  would write     {out_path}')
        print('dry run, nothing built.')
        return 0

    try:
        grid = build_sampling_grid(
            receptor, region=region, spacing=args.spacing,
            shell=(args.shell_min, args.shell_max),
            probe_radius=args.probe_radius, fit_radius=args.fit_radius,
            lattice_spacing=args.lattice_spacing, max_voxels=args.max_voxels)
    except GridError as e:
        raise SystemExit(f'error: {e}')

    if grid.n_points == 0:
        return 1

    write_grid(grid, out_path, xyz=args.write_xyz,
               manifest=not args.no_manifest)
    print(f'\nDock against it with:\n'
          f'  run_glycandock.py <complex>.pdb --grid {out_path} '
          f'--run-id <receptor>-<probe>_{grid_id}')
    return 0


if __name__ == '__main__':
    exit(main())
