#!/usr/bin/env python3

'''
Build a PyMOL visualization for a set of glycan binding maps, driven by the
map manifest (JSON or YAML) written by map_ensembles.py. This replaces
hand-processing each .dx in an interactive session when a run produces tens of
maps.

The default scene follows "surface = consensus, mesh = per-probe":
  * each probe's map is drawn as auto-leveled isocontours (analysis picks the
    levels; see --contour-mode), colored per-probe from a colorblind-safe
    categorical palette so probes are distinguishable in overlay;
  * one consensus map (best_probe by default, else consensus_min/mean) is
    projected onto the receptor surface for the "compare all probes at once"
    view.

Because consensus is per map-tag (a boltzmann@0.5 map is never compared to a
boltzmann@1.0 map), a single tag is visualized at a time; pass --tag to choose,
otherwise the first tag in the manifest is used and the rest are listed.

With --hotspots, each per-probe map is also segmented and ranked, and (with the
receptor) attributed to its lining residues; the tables are written as CSVs.

Examples
--------
  # Best_probe surface + per-probe contour overlay, saved to a .pse:
  vis_maps.py maps/map_manifest.json --receptor 1o7v_receptor.pdb -o 1o7v_vis

  # Pick a tag, use the anti-bleed contour levels, write hotspot tables, render:
  vis_maps.py maps/map_manifest.json --receptor rec.pdb --tag boltzmann_b0p5 \
      --contour-mode components --hotspots --render -o 1o7v_vis
'''

import os
import sys
import json
import argparse

# The glycographer package isn't pip-installed (in-development repo); put the
# repo root on sys.path. This script lives at <repo>/scripts/vis/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pymol
from pymol import cmd

from glycographer.colors import probe_palette
from glycographer.map import VolMap
from glycographer.analysis import find_hotspots, attribute_hotspots_to_residues
from glycographer.vis import (
    format_background,
    vis_receptor,
    vis_grid,
    vis_crystal_ligand,
    load_volmap_from_dx,
    draw_map_contours,
    draw_mapped_surface,
    draw_best_probe_surface,
)


# Consensus reductions that make sense as a colored receptor surface, in the
# order we prefer when the user does not name one.
SURFACE_PREFERENCE = ['best_probe', 'min', 'mean', 'support', 'selectivity']


def load_manifest(path):
    '''
    Read the map manifest into (grid, maps, consensus). Supports both JSON and
    YAML (matching map_ensembles.py's config handling); the format is chosen by
    file extension, defaulting to JSON.
    '''
    with open(path) as f:
        if path.endswith(('.yaml', '.yml')):
            import yaml
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return (data.get('grid', {}),
            data.get('maps', []),
            data.get('consensus', []))


def tags_in(maps):
    '''Ordered unique map tags present in the manifest maps list.'''
    seen = []
    for m in maps:
        if m['tag'] not in seen:
            seen.append(m['tag'])
    return seen


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('manifest',
                        help='Map manifest (JSON or YAML) written by '
                             'map_ensembles.py; format chosen by extension.')
    parser.add_argument('-rec', '--receptor', required=True,
                        help='Receptor PDB the maps were built around.')
    parser.add_argument('--tag',
                        help='Which map tag to visualize (e.g. boltzmann_b0p5). '
                             'Default: the first tag in the manifest.')
    parser.add_argument('-grid', '--grid',
                        help='Optional sampling-grid PDB to display.')
    parser.add_argument('-lig', '--ligand-pose',
                        help='Optional crystal glycoligand pose to display.')

    # Contour (per-probe mesh) options.
    parser.add_argument('--contour-mode', default='absolute',
                        choices=['absolute', 'quantile', 'components'],
                        help='How per-probe isocontour levels are chosen '
                             '(default: absolute).')
    parser.add_argument('-n', '--n-levels', type=int, default=4,
                        help='Contours per probe map (default: 4).')
    parser.add_argument('--step', type=float, default=1.0,
                        help='REU step for absolute contour mode (default: 1.0).')
    parser.add_argument('--smooth', type=float, default=None,
                        help='Gaussian sigma (voxels) to smooth maps for '
                             'display/segmentation (default: none).')

    # Consensus (surface) options.
    parser.add_argument('--surface', default=None,
                        help='Consensus reduction to show as the receptor '
                             'surface (best_probe, min, mean, support, '
                             'selectivity), or "none". Default: the first '
                             'available in preference order.')

    # Hotspot / attribution tables.
    parser.add_argument('--hotspots', action='store_true',
                        help='Also segment/rank each probe map and attribute '
                             'hotspots to lining residues; write CSV tables.')
    parser.add_argument('--radius', type=float, default=4.0,
                        help='Lining-residue distance cutoff in A (default: 4.0).')
    parser.add_argument('--min-voxels', type=int, default=3,
                        help='Minimum voxels for a hotspot (default: 3).')

    parser.add_argument('-o', '--outprefix', default='map_vis',
                        help='Output file prefix (default: map_vis).')
    parser.add_argument('--outdir', default='.',
                        help='Output directory (default: cwd).')
    parser.add_argument('--render', action='store_true',
                        help='Also ray-trace a PNG (needs OSMesa/libGL when '
                             'headless on HPC).')
    parser.add_argument('--bg-rgb', default='white',
                        help='Background color (default: white).')

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    grid, maps, consensus = load_manifest(args.manifest)
    if not maps:
        raise SystemExit(f'No maps listed in {args.manifest}.')

    # Choose the tag to visualize (keeps probes comparable).
    all_tags = tags_in(maps)
    tag = args.tag or all_tags[0]
    if tag not in all_tags:
        raise SystemExit(f'Tag {tag!r} not in manifest; available: {all_tags}.')
    if not args.tag and len(all_tags) > 1:
        print(f'Multiple tags present {all_tags}; visualizing {tag!r}. '
              'Pass --tag to choose another.')

    probe_maps = [m for m in maps if m['tag'] == tag]
    probe_labels = [m['ensemble'] for m in probe_maps]
    palette = probe_palette(probe_labels)

    pymol.finish_launching(['pymol', '-cq'])
    format_background(args.bg_rgb)

    # Receptor, optional grid and ligand.
    rec_name = vis_receptor(args.receptor)
    if args.grid:
        vis_grid(args.grid)
    if args.ligand_pose:
        vis_crystal_ligand(args.ligand_pose)

    # ---- Per-probe contour meshes (colored per probe) -------------------
    for m in probe_maps:
        label = m['ensemble']
        dx = m['dx']
        if not os.path.isfile(dx):
            print(f'Warning: {label} map missing on disk ({dx}); skipping.')
            continue
        names, levels = draw_map_contours(
            dx, base_color=palette[label], n=args.n_levels,
            mode=args.contour_mode, step=args.step, smooth_sigma=args.smooth)
        if names:
            cmd.group(f'contours_{label}', ' '.join(names))

    print('Per-probe contour colors (probe -> RGB): '
          + ', '.join(f'{k}={tuple(round(c, 2) for c in v)}'
                      for k, v in palette.items()))

    # ---- Consensus surface ----------------------------------------------
    cons_for_tag = [c for c in consensus if c.get('tag') == tag]
    surf = args.surface
    if surf is None:
        available = {c['reduction'] for c in cons_for_tag}
        surf = next((r for r in SURFACE_PREFERENCE if r in available), 'none')

    if surf != 'none':
        entry = next((c for c in cons_for_tag if c['reduction'] == surf), None)
        if entry is None:
            print(f'No consensus "{surf}" map for tag {tag!r}; skipping surface. '
                  f'(Available: {[c["reduction"] for c in cons_for_tag]})')
        elif not os.path.isfile(entry['dx']):
            print(f'Consensus "{surf}" map missing on disk; skipping surface.')
        else:
            surf_map = load_volmap_from_dx(entry['dx'])
            if surf == 'best_probe':
                draw_best_probe_surface(rec_name, surf_map, entry['probes'])
                print(f'best_probe surface; probe index order: {entry["probes"]}')
            else:
                draw_mapped_surface(rec_name, surf_map, map_dx=entry['dx'])
                print(f'consensus "{surf}" surface on {rec_name}.')

    # ---- Hotspot / attribution tables -----------------------------------
    if args.hotspots:
        for m in probe_maps:
            label, dx = m['ensemble'], m['dx']
            if not os.path.isfile(dx):
                continue
            vm = VolMap.from_dx(dx)
            hs = find_hotspots(vm, min_voxels=args.min_voxels,
                               smooth_sigma=args.smooth)
            hs_path = os.path.join(args.outdir, f'{label}_{tag}_hotspots.csv')
            hs.to_csv(hs_path, index=False)
            att = attribute_hotspots_to_residues(
                vm, args.receptor, min_voxels=args.min_voxels,
                radius=args.radius, smooth_sigma=args.smooth)
            att_path = os.path.join(args.outdir, f'{label}_{tag}_lining.csv')
            att.to_csv(att_path, index=False)
            print(f'[{label}] {len(hs)} hotspots -> {hs_path}; '
                  f'lining residues -> {att_path}')

    # ---- Save session (+ optional render) -------------------------------
    pse = os.path.join(args.outdir, f'{args.outprefix}.pse')
    cmd.save(pse)
    print(f'Saved PyMOL session to {os.path.abspath(pse)}')

    if args.render:
        png = os.path.join(args.outdir, f'{args.outprefix}.png')
        try:
            cmd.ray(1600, 1200)
            cmd.png(png, dpi=300)
            print(f'Rendered {os.path.abspath(png)}')
        except Exception as e:
            print(f'Render failed ({e}); the .pse is still saved. Headless '
                  'ray-tracing needs OSMesa/libGL.')

    cmd.quit()


if __name__ == '__main__':
    main()
