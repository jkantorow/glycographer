#!/usr/bin/env python

'''
Generate volumetric maps -- and optional cross-probe consensus maps -- for one
or more GlycanDock pose ensembles docked against the *same* receptor.

Everything runs in a single process so that all maps share one grid: this is
what makes different probes comparable and lets consensus maps be built without
a second pass.

Two ways to drive it, both of which normalize to the same internal job list
(a global config + a list of per-ensemble jobs, each carrying its own list of
map specs):

  1. Flat mode (same maps for every ensemble, or a parameter sweep). List-valued
     flags take the cartesian product, so this covers "one boltzmann per
     ensemble", "a beta sweep", and "boltzmann + intengmin per ensemble":

       map_ensembles.py posedir1 posedir2 posedir3 \
           --map-type boltzmann --beta 0.3 0.5 1.0 \
           --grid-pdb receptor.pdb --outdir maps --consensus mean best_probe

  2. Config mode (heterogeneous per-ensemble/per-map parameters). Point at a
     JSON (or YAML, if pyyaml is installed) spec:

       map_ensembles.py --config jobs.json

     jobs.json:
       {
         "global": {"grid_pdb": "receptor.pdb", "voxel_size": 1.0,
                    "padding": 6.0, "outdir": "maps",
                    "consensus": ["mean", "best_probe", "support"]},
         "ensembles": [
           {"posedir": "runs/diSia", "label": "diSia",
            "maps": [{"type": "boltzmann", "beta": 0.5},
                     {"type": "boltzmann", "beta": 1.0},
                     {"type": "intengmin"}]},
           {"posedir": "runs/LacNAc", "label": "LacNAc",
            "maps": [{"type": "boltzmann", "beta": 0.5}]}
         ]
       }

Consensus maps are built per *map tag* (type + parameters): only maps that share
an identical tag across ensembles are combined -- e.g. boltzmann@0.5 across every
ensemble that produced one. Mismatched betas are never mixed (their absolute
scales differ), and singletons are skipped with a note.
'''

import os
import sys
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# The glycographer package isn't pip-installed (in-development repo), so the
# repo root must be on sys.path. This script lives at <repo>/scripts/utils/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the sibling util's pose-collection logic rather than duplicating it.
_UTIL_DIR = os.path.dirname(os.path.abspath(__file__))
if _UTIL_DIR not in sys.path:
    sys.path.insert(0, _UTIL_DIR)
from process_glycandock_output import build_pose_list

from glycographer.dock import GlycanDockEnsemble
from glycographer.map import (
    GridSpec, ConsensusMap,
    BoltzmannMapper, ResidueBoltzmannMapper,
    IntEngMinMapper, IntEngAvgMapper,
)


# Registry of available map types -> Mapper class.
MAP_TYPES = {
    'boltzmann': BoltzmannMapper,
    'residue_boltzmann': ResidueBoltzmannMapper,
    'intengmin': IntEngMinMapper,
    'intengavg': IntEngAvgMapper,
}
# Types that take a beta parameter.
BOLTZMANN_TYPES = {'boltzmann', 'residue_boltzmann'}
# Types that need the per-residue REF15 table parsed from the poses.
RESIDUE_TYPES = {'residue_boltzmann'}

# Consensus reductions -> ConsensusMap method.
CONSENSUS_REDUCTIONS = {
    'mean': lambda cm: cm.consensus_mean(),
    'min': lambda cm: cm.consensus_min(),
    'support': lambda cm: cm.support_count(),
    'best_probe': lambda cm: cm.best_probe(),
    'selectivity': lambda cm: cm.selectivity_entropy(),
}
DEFAULT_CONSENSUS = ['mean', 'best_probe', 'support']


def _fmt_num(x):
    '''Filename-safe number: 0.5 -> 0p5, -1.0 -> m1p0.'''
    return str(x).replace('.', 'p').replace('-', 'm')


@dataclass
class MapSpec:
    '''A single map to produce: a type and (for Boltzmann types) a beta.'''
    type: str
    beta: float = 0.5

    def __post_init__(self):
        if self.type not in MAP_TYPES:
            raise ValueError(
                f'Unknown map type {self.type!r}; choose from '
                f'{sorted(MAP_TYPES)}.')

    @property
    def tag(self):
        '''Stable identifier used for filenames and consensus grouping.'''
        if self.type in BOLTZMANN_TYPES:
            return f'{self.type}_b{_fmt_num(self.beta)}'
        return self.type

    @property
    def needs_energies(self):
        return self.type in RESIDUE_TYPES

    def make_mapper(self, ensemble, grid):
        cls = MAP_TYPES[self.type]
        if self.type in BOLTZMANN_TYPES:
            return cls(ensemble, grid=grid, beta=self.beta)
        return cls(ensemble, grid=grid)


@dataclass
class EnsembleJob:
    '''One ensemble (a pose directory) plus the maps to produce for it.'''
    posedir: str
    label: str = None
    poserange: Optional[Tuple[int, int]] = None
    maps: List[MapSpec] = field(default_factory=list)

    def __post_init__(self):
        if not self.label:
            self.label = os.path.basename(os.path.normpath(self.posedir))

    @property
    def needs_energies(self):
        return any(m.needs_energies for m in self.maps)


# ---------------------------------------------------------------------------
# Building the job list from either input mode
# ---------------------------------------------------------------------------
def load_config(path):
    '''Load a JSON (or YAML) job spec into (global_cfg, [EnsembleJob]).'''
    with open(path) as f:
        if path.endswith(('.yaml', '.yml')):
            import yaml
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    global_cfg = dict(data.get('global', {}))
    jobs = []
    for entry in data.get('ensembles', []):
        maps = [MapSpec(**m) for m in entry.get('maps', [])]
        if not maps:
            raise ValueError(
                f"Ensemble {entry.get('posedir')!r} has no 'maps' listed.")
        rng = entry.get('poserange')
        jobs.append(EnsembleJob(
            posedir=entry['posedir'],
            label=entry.get('label'),
            poserange=tuple(rng) if rng else None,
            maps=maps,
        ))
    if not jobs:
        raise ValueError(f'No ensembles defined in {path}.')
    return global_cfg, jobs


def build_flat(args):
    '''Build (global_cfg, [EnsembleJob]) from flat CLI args.'''
    # Cartesian product of requested types x betas, shared by every ensemble.
    specs = []
    for t in args.map_type:
        if t in BOLTZMANN_TYPES:
            specs.extend(MapSpec(type=t, beta=b) for b in args.beta)
        else:
            specs.append(MapSpec(type=t))

    rng = tuple(args.poserange) if args.poserange else None
    jobs = [EnsembleJob(posedir=d, poserange=rng, maps=list(specs))
            for d in args.posedirs]

    global_cfg = {
        'grid_pdb': args.grid_pdb,
        'grid_selection': args.grid_selection,
        'include_receptor': args.include_receptor,
        'voxel_size': args.voxel_size,
        'padding': args.padding,
        'outdir': args.outdir,
        'consensus': args.consensus,
        'write_json': not args.no_json,
        'manifest_format': args.manifest_format,
    }
    return global_cfg, jobs


# ---------------------------------------------------------------------------
# Execution engine (shared by both modes)
# ---------------------------------------------------------------------------
def run(global_cfg, jobs):
    voxel_size = global_cfg.get('voxel_size', 1.0)
    padding = global_cfg.get('padding', 5.0)
    outdir = global_cfg.get('outdir', '.')
    write_json = global_cfg.get('write_json', True)
    os.makedirs(outdir, exist_ok=True)

    # 1. Build every ensemble (parse per-residue energies only when needed).
    ensembles = []
    for job in jobs:
        pose_list = build_pose_list(job.posedir, job.poserange)
        if not pose_list:
            rng = f' in range {job.poserange}' if job.poserange else ''
            raise SystemExit(f'No poses found in {job.posedir}{rng}.')
        ens = GlycanDockEnsemble.from_poses(
            pose_list=pose_list, run_id=job.label,
            parse_energies=job.needs_energies)
        ens.to_pdb(os.path.join(
            outdir, f'{job.label}_{ens._n_poses}p_ensemble.pdb'))
        ensembles.append(ens)
        print(f'[{job.label}] built ensemble: {ens._n_poses} poses '
              f'(parse_energies={job.needs_energies})')

    # 2. One shared grid for all ensembles.
    grid_pdb = global_cfg.get('grid_pdb')
    if grid_pdb:
        grid = GridSpec.from_pdb(
            grid_pdb, voxel_size=voxel_size, padding=padding,
            selection=global_cfg.get('grid_selection', 'not name *H*'))
        print(f'Shared grid from {grid_pdb}: shape {grid.shape}')
    else:
        grid = GridSpec.from_ensembles(
            ensembles, voxel_size=voxel_size, padding=padding,
            include_receptor_pdb=global_cfg.get('include_receptor'))
        print(f'Shared grid from union of {len(ensembles)} ensembles: '
              f'shape {grid.shape}')

    # 3. Produce every map; group VolMaps by tag for consensus.
    by_tag = {}
    manifest = {
        'grid': {'origin': grid.origin.tolist(),
                 'shape': list(grid.shape),
                 'voxel_size': grid.voxel_size},
        'maps': [], 'consensus': [],
    }
    for ens, job in zip(ensembles, jobs):
        for spec in job.maps:
            vmap = spec.make_mapper(ens, grid).map()
            dx = os.path.join(outdir, f'{job.label}_{spec.tag}.dx')
            vmap.to_dx(dx)
            if write_json:
                vmap.to_json(os.path.join(
                    outdir, f'{job.label}_{spec.tag}_metadata.json'))
            by_tag.setdefault(spec.tag, []).append((job.label, vmap))
            manifest['maps'].append({'ensemble': job.label, 'tag': spec.tag,
                                     'dx': os.path.abspath(dx)})
            print(f'[{job.label}] wrote {spec.tag} -> {dx}')

    # 4. Consensus maps per shared tag.
    reductions = global_cfg.get('consensus') or []
    for tag, entries in by_tag.items():
        if not reductions:
            break
        if len(entries) < 2:
            print(f'Consensus: skipping {tag} (only {len(entries)} map).')
            continue
        labels = [lab for lab, _ in entries]
        vmaps = [vm for _, vm in entries]
        cm = ConsensusMap(vmaps, probe_labels=labels)
        for red in reductions:
            if red not in CONSENSUS_REDUCTIONS:
                raise SystemExit(
                    f'Unknown consensus reduction {red!r}; choose from '
                    f'{sorted(CONSENSUS_REDUCTIONS)}.')
            cvm = CONSENSUS_REDUCTIONS[red](cm)
            dx = os.path.join(outdir, f'consensus_{tag}_{red}.dx')
            cvm.to_dx(dx)
            manifest['consensus'].append({'tag': tag, 'reduction': red,
                                          'probes': labels,
                                          'dx': os.path.abspath(dx)})
            print(f'Consensus [{tag}] {red} over {labels} -> {dx}')

    fmt = str(global_cfg.get('manifest_format') or 'json').lower()
    if fmt in ('yaml', 'yml'):
        import yaml
        manifest_path = os.path.join(outdir, 'map_manifest.yaml')
        with open(manifest_path, 'w') as f:
            yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    else:
        manifest_path = os.path.join(outdir, 'map_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    print(f'Wrote manifest to {manifest_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate volumetric (and consensus) maps for one or more '
                    'GlycanDock ensembles docked against the same receptor.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Flat mode covers same-maps-for-all and sweeps; use --config '
               'for heterogeneous per-ensemble map parameters. Put --consensus '
               'last (or after posedirs) so it does not swallow positional '
               'pose directories.')

    parser.add_argument('posedirs', nargs='*',
                        help='One or more directories of GlycanDock output '
                             'poses (all vs the same receptor). Omit if using '
                             '--config.')
    parser.add_argument('--config',
                        help='JSON (or YAML) job spec for heterogeneous '
                             'per-ensemble/per-map parameters. When given, flat '
                             'map/grid args are ignored.')

    # Flat-mode map specification:
    parser.add_argument('--map-type', nargs='+', default=['boltzmann'],
                        choices=sorted(MAP_TYPES),
                        help='Map type(s) to produce for every ensemble '
                             '(default: boltzmann).')
    parser.add_argument('--beta', nargs='+', type=float, default=[0.5],
                        help='Beta value(s) for Boltzmann maps; multiple '
                             'values produce a sweep (default: 0.5).')
    parser.add_argument('-range', '--poserange', type=int, nargs=2,
                        metavar=('START', 'STOP'),
                        help='Inclusive pose-id range applied to every '
                             'posedir (default: all poses).')

    # Global grid / output:
    parser.add_argument('--grid-pdb',
                        help='Structure defining the shared grid frame '
                             '(receptor or gridbox pdb). Default: enclose the '
                             'union of all ensembles (guarantees every pose is '
                             'inside every map).')
    parser.add_argument('--grid-selection', default='not name *H*',
                        help='MDAnalysis selection used with --grid-pdb '
                             "(default: 'not name *H*').")
    parser.add_argument('--include-receptor',
                        help='Receptor pdb to also enclose when the grid is '
                             'built from the ensemble union.')
    parser.add_argument('--voxel-size', type=float, default=1.0,
                        help='Voxel edge length in Angstroms (default: 1.0).')
    parser.add_argument('--padding', type=float, default=5.0,
                        help='Padding around the grid in Angstroms '
                             '(default: 5.0).')
    parser.add_argument('--outdir', default='.',
                        help='Output directory (default: cwd).')
    parser.add_argument('--consensus', nargs='*', default=None,
                        metavar='REDUCTION',
                        help='Also write consensus maps across ensembles for '
                             'each shared map tag. Reductions: '
                             + ', '.join(sorted(CONSENSUS_REDUCTIONS))
                             + '. With no value given, defaults to: '
                             + ' '.join(DEFAULT_CONSENSUS) + '.')
    parser.add_argument('--no-json', action='store_true',
                        help='Do not write per-map JSON metadata sidecars.')
    parser.add_argument('--manifest-format', choices=['json', 'yaml'],
                        default='json',
                        help='Format for the run manifest (default: json). '
                             'yaml requires pyyaml.')

    args = parser.parse_args()

    if args.config:
        global_cfg, jobs = load_config(args.config)
        global_cfg.setdefault('voxel_size', 1.0)
        global_cfg.setdefault('padding', 5.0)
        global_cfg.setdefault('outdir', '.')
        global_cfg.setdefault('manifest_format', 'json')
    else:
        if not args.posedirs:
            parser.error('Provide one or more pose directories, or --config.')
        # `--consensus` with no values -> the sensible default set.
        if args.consensus is not None and len(args.consensus) == 0:
            args.consensus = list(DEFAULT_CONSENSUS)
        global_cfg, jobs = build_flat(args)

    run(global_cfg, jobs)


if __name__ == '__main__':
    exit(main())
