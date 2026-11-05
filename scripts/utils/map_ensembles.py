#!/usr/bin/env python

'''
Generate volumetric maps -- and optional cross-probe consensus maps -- for one
or more GlycanDock pose ensembles docked against the *same* receptor.

Everything runs in a single process so that all maps share one grid: this is
what makes different probes comparable and lets consensus maps be built without
a second pass.

Pose sources
------------
Each source may be a directory of loose decoy PDBs, a directory of block
archives (<run_id>_blockNNN.tar.xz, as written by run_glycandock.py), or a
single merged archive (<run_id>.tar.xz / .tar.gz). Archive members are read in
memory; nothing is extracted.

Two ways to drive it, both of which normalize to the same internal job list
(a global config + a list of per-ensemble jobs, each carrying its own list of
map specs):

  1. Flat mode (same maps for every ensemble, or a parameter sweep). List-valued
     flags take the cartesian product, so this covers "one boltzmann per
     ensemble" and "a beta sweep":

       map_ensembles.py runs/1o7v-gal_global.tar.xz runs/1o7v-nag_global/ \
           --beta 0.3 0.5 1.0 1.62 -range 1 1000 \
           --grid-pdb 1o7v_rec.pdb --outdir maps

  2. Config mode (heterogeneous per-ensemble/per-map parameters). Point at a
     JSON (or YAML, if pyyaml is installed) spec:

       map_ensembles.py --config jobs.json

     jobs.json:
       {
         "global": {"grid_pdb": "receptor.pdb", "voxel_size": 1.0,
                    "padding": 6.0, "outdir": "maps"},
         "ensembles": [
           {"posedir": "runs/1o7v-sia_global.tar.xz", "poserange": [1, 1000],
            "maps": [{"type": "boltzmann", "beta": 0.5},
                     {"type": "boltzmann", "beta": 1.62}]},
           {"posedir": "runs/1o7v-sia26nag_global_5000_poses",
            "run_id": "1o7v-sia26nag_global",
            "maps": [{"type": "boltzmann", "beta": 0.5}]}
         ]
       }

Naming (see glycographer.naming)
--------------------------------
Each ensemble's run_id is, in order of preference: "run_id" in the config
(or "label", its older name); the name of the archive(s) it was read from,
minus any _blockNNN suffix; or the pose directory's name. Legacy "_<N>_poses"
and "_<N>p" suffixes are dropped, and the pose count actually used is
appended once, giving <run_id>_<N>p. Outputs in --outdir:

  <run_id>_<N>p_ensemble.pdb, <run_id>_<N>p_scores.csv
  <run_id>_<N>p_<map_type>[_b<beta>].dx / .json        one per map
  <manifest_id>_map_manifest.json (or .yaml) and .csv  one per run

manifest_id is --manifest-id if given, else built from the fields every map
in the run shares, e.g. 1o7v-gal_global_1000p_boltzmann for a beta sweep over
one ensemble, or 1o7v_global_1000p_boltzmann_b0p5 for six probes at one beta.
The CSV has one row per map (receptor, ligand, grid_id, n_poses, map_type,
beta, paths, value statistics); concatenate many with
glycographer.map.load_map_manifests().

Grid
----
All maps in one run share one grid: by default the union of the ensembles'
ligand atoms (+ padding), or a structure via grid_pdb. To add maps to a set
made earlier, reuse that set's grid with grid_from (its map manifest, which
stores the origin at full precision, or one of its .dx maps); grid_expand
grows it by whole voxels if new poses fall outside, keeping every old voxel
aligned. Boltzmann values depend only on each ensemble's own poses, so which
other ensembles share a run never changes a map's values -- only the grid.

Map types
---------
boltzmann (default), residue_boltzmann (needs per-residue interface energies:
a precomputed parquet via "interface_energies" / --interface-energies, or
PyRosetta to compute them), intengavg. intengmin is deprecated: use boltzmann
with a large beta (e.g. 10), which converges to it.

Consensus maps (on hiatus) are built per map tag (type + beta) across the
ensembles that share it, with --consensus.
'''

import os
import sys
import csv
import json
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# The glycographer package isn't pip-installed (in-development repo), so the
# repo root must be on sys.path. This script lives at <repo>/scripts/utils/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from glycographer import naming
from glycographer.archive import split_pose_path
from glycographer.utils import build_pose_list
from glycographer.dock import GlycanDockEnsemble
from glycographer.map import (
    GridSpec, ConsensusMap,
    BoltzmannMapper, ResidueBoltzmannMapper, IntEngAvgMapper,
)


# Registry of available map types -> Mapper class.
MAP_TYPES = {
    'boltzmann': BoltzmannMapper,
    'residue_boltzmann': ResidueBoltzmannMapper,
    'intengavg': IntEngAvgMapper,
}
DEPRECATED_TYPES = {
    'intengmin': 'use boltzmann with a large beta (e.g. --beta 10), which '
                 'converges to the minimum-energy map as beta -> infinity',
}
# Types that take a beta parameter.
BOLTZMANN_TYPES = {'boltzmann', 'residue_boltzmann'}
DEFAULT_BETA = 0.5
# Types that need per-residue interface energies (analysis.extract_interface_energies).
INTERFACE_TYPES = {'residue_boltzmann'}

# Consensus reductions -> ConsensusMap method.
CONSENSUS_REDUCTIONS = {
    'mean': lambda cm: cm.consensus_mean(),
    'min': lambda cm: cm.consensus_min(),
    'support': lambda cm: cm.support_count(),
    'best_probe': lambda cm: cm.best_probe(),
    'selectivity': lambda cm: cm.selectivity_entropy(),
}
DEFAULT_CONSENSUS = ['mean', 'best_probe', 'support']

# Column order of the CSV manifest (one row per map).
CSV_COLUMNS = [
    'map_id', 'receptor', 'ligand', 'grid_id', 'n_poses', 'map_type', 'beta',
    'run_id', 'ensemble_id', 'map_tag',
    'n_occupied', 'n_favorable', 'min_value', 'max_value', 'mean_value',
    'std_value', 'n_voxels', 'voxel_size',
    'dx_path', 'json_path', 'ensemble_pdb', 'scores_csv', 'source',
    'poserange', 'created',
]


@dataclass
class MapSpec:
    '''A single map to produce: a type and (for Boltzmann types) a beta.'''
    type: str
    beta: Optional[float] = None

    def __post_init__(self):
        if self.type in DEPRECATED_TYPES:
            raise ValueError(f'Map type {self.type!r} is deprecated: '
                             f'{DEPRECATED_TYPES[self.type]}.')
        if self.type not in MAP_TYPES:
            raise ValueError(
                f'Unknown map type {self.type!r}; choose from '
                f'{sorted(MAP_TYPES)}.')
        if self.type in BOLTZMANN_TYPES:
            self.beta = float(DEFAULT_BETA if self.beta is None else self.beta)
        else:
            self.beta = None

    @property
    def tag(self):
        '''<type>[_b<beta>]: used in filenames and for consensus grouping.'''
        return naming.map_tag(self.type, self.beta)

    @property
    def needs_interface_energies(self):
        return self.type in INTERFACE_TYPES

    def make_mapper(self, ensemble, grid):
        cls = MAP_TYPES[self.type]
        if self.type in BOLTZMANN_TYPES:
            return cls(ensemble, grid=grid, beta=self.beta)
        return cls(ensemble, grid=grid)


@dataclass
class EnsembleJob:
    '''One pose source (directory or archive) plus the maps to produce for it.'''
    posedir: str
    run_id: Optional[str] = None
    poserange: Optional[Tuple[int, int]] = None
    maps: List[MapSpec] = field(default_factory=list)
    interface_energies: Optional[str] = None

    def __post_init__(self):
        # The same map requested twice would write the same file twice.
        unique = {}
        for m in self.maps:
            unique.setdefault(m.tag, m)
        self.maps = list(unique.values())

    @property
    def needs_interface_energies(self):
        return any(m.needs_interface_energies for m in self.maps)

    def resolve_run_id(self, pose_list):
        '''Explicit run_id, else the source archive name, else the directory name.'''
        if self.run_id:
            return naming.normalize_run_id(self.run_id)
        archives = {split_pose_path(p)[0] for p in pose_list}
        if None not in archives:
            names = {naming.normalize_run_id(a) for a in archives}
            if len(names) == 1:
                return names.pop()
        return naming.normalize_run_id(self.posedir)


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
            run_id=entry.get('run_id') or entry.get('label'),
            poserange=tuple(rng) if rng else None,
            maps=maps,
            interface_energies=entry.get('interface_energies'),
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

    ie = args.interface_energies or []
    if ie and len(ie) != len(args.posedirs):
        raise SystemExit(f'--interface-energies takes one parquet per pose '
                         f'source ({len(args.posedirs)}); got {len(ie)}.')
    if args.run_id and len(args.run_id) != len(args.posedirs):
        raise SystemExit(f'--run-id takes one id per pose source '
                         f'({len(args.posedirs)}); got {len(args.run_id)}.')

    rng = tuple(args.poserange) if args.poserange else None
    jobs = [EnsembleJob(posedir=d, poserange=rng, maps=list(specs),
                        run_id=args.run_id[i] if args.run_id else None,
                        interface_energies=ie[i] if ie else None)
            for i, d in enumerate(args.posedirs)]

    global_cfg = {
        'grid_pdb': args.grid_pdb,
        'grid_from': args.grid_from,
        'grid_expand': args.grid_expand,
        'grid_selection': args.grid_selection,
        'include_receptor': args.include_receptor,
        'voxel_size': args.voxel_size,
        'padding': args.padding,
        'outdir': args.outdir,
        'consensus': args.consensus,
        'write_json': not args.no_json,
        'manifest_format': args.manifest_format,
        'manifest_id': args.manifest_id,
    }
    return global_cfg, jobs


# ---------------------------------------------------------------------------
# Execution engine (shared by both modes)
# ---------------------------------------------------------------------------
def _attach_interface_energies(ens, job):
    '''Populate ens.interface_energies for residue-level map types.'''
    if job.interface_energies:
        import pandas as pd
        ens.interface_energies = pd.read_parquet(job.interface_energies)
        return
    try:
        from glycographer.analysis import extract_interface_energies
        extract_interface_energies(ens)
    except ImportError as e:
        raise SystemExit(
            f'[{ens.run_id}] residue_boltzmann needs per-residue interface '
            f'energies: pass a precomputed parquet ("interface_energies" in '
            f'the config, or --interface-energies), or run where PyRosetta is '
            f'installed ({e}).')


def _write_manifest_csv(path, records):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        w.writeheader()
        for r in records:
            w.writerow({k: ('' if r.get(k) is None else r.get(k))
                        for k in CSV_COLUMNS})


def run(global_cfg, jobs):
    voxel_size = global_cfg.get('voxel_size', 1.0)
    padding = global_cfg.get('padding', 5.0)
    outdir = global_cfg.get('outdir', '.')
    write_json = global_cfg.get('write_json', True)
    os.makedirs(outdir, exist_ok=True)
    created = datetime.now().isoformat(timespec='seconds')

    # 1. Resolve every job's poses and name before any heavy work, so a
    # naming collision fails fast instead of after the first maps are written.
    resolved = []
    for job in jobs:
        pose_list = build_pose_list(job.posedir, job.poserange)
        if not pose_list:
            rng = f' in range {job.poserange}' if job.poserange else ''
            raise SystemExit(f'No poses found in {job.posedir}{rng}.')
        run_id = job.resolve_run_id(pose_list)
        ens_id = naming.ensemble_id(run_id, len(pose_list))
        resolved.append((job, pose_list, run_id, ens_id))

    planned = [naming.map_id(ens_id, s.type, s.beta)
               for job, _, _, ens_id in resolved for s in job.maps]
    dupes = sorted({m for m in planned if planned.count(m) > 1})
    if dupes:
        raise SystemExit(
            f'These maps would be written more than once: {dupes}. Two pose '
            f'sources resolve to the same run_id and pose count; set "run_id" '
            f'(config) or --run-id to tell them apart.')

    # 2. Build every ensemble.
    ensembles, ens_records = [], []
    for job, pose_list, run_id, ens_id in resolved:
        ens = GlycanDockEnsemble.from_poses(
            pose_list=pose_list, run_id=ens_id, parse_energies=False)
        if job.needs_interface_energies:
            _attach_interface_energies(ens, job)
        ens_pdb = os.path.abspath(
            ens.to_pdb(os.path.join(outdir, f'{ens_id}_ensemble.pdb')))
        scores_csv = ens.scores_to_csv(
            os.path.join(outdir, f'{ens_id}_scores.csv'))
        ens_records.append({
            'ensemble_id': ens_id, 'run_id': run_id,
            'n_poses': ens._n_poses,
            'source': os.path.abspath(job.posedir),
            'poserange': list(job.poserange) if job.poserange else None,
            'ensemble_pdb': ens_pdb,
            'scores_csv': os.path.abspath(scores_csv),
        })
        ensembles.append(ens)
        print(f'[{ens_id}] built ensemble: {ens._n_poses} poses from '
              f'{job.posedir}')

    # 3. One shared grid for all ensembles.
    grid_pdb = global_cfg.get('grid_pdb')
    grid_from = global_cfg.get('grid_from')
    if grid_from and grid_pdb:
        raise SystemExit('Give grid_from or grid_pdb, not both.')
    if grid_from:
        grid = GridSpec.from_reference(grid_from)
        print(f'Shared grid reused from {grid_from}: origin '
              f'{grid.origin.round(3).tolist()}, shape {grid.shape}')
        if global_cfg.get('grid_expand'):
            grown = grid.expanded_to_cover(ensembles)
            if grown.shape != grid.shape:
                print(f'Expanded by whole voxels to cover every pose: origin '
                      f'{grown.origin.round(3).tolist()}, shape {grown.shape} '
                      f'(same lattice; overlapping voxels align exactly).')
            grid = grown
    elif grid_pdb:
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

    # 4. Produce every map; group VolMaps by tag for consensus.
    by_tag = {}
    map_records = []
    for ens, (job, *_), ens_rec in zip(ensembles, resolved, ens_records):
        for spec in job.maps:
            vmap = spec.make_mapper(ens, grid).map()
            vmap.to_dx(os.path.join(outdir, f'{vmap.map_id}.dx'))
            rec = vmap.metadata()
            rec.pop('origin'), rec.pop('shape')   # shared; in manifest['grid']
            rec['json_path'] = (os.path.abspath(vmap.to_json())
                                if write_json else None)
            rec.update(
                map_tag=spec.tag,
                ensemble_pdb=ens_rec['ensemble_pdb'],
                scores_csv=ens_rec['scores_csv'],
                source=ens_rec['source'],
                poserange=('-'.join(map(str, job.poserange))
                           if job.poserange else None),
                created=created,
            )
            map_records.append(rec)
            by_tag.setdefault(spec.tag, []).append((rec, vmap))
            print(f'[{ens_rec["ensemble_id"]}] wrote {vmap.map_id}.dx')

    # 5. Consensus maps per shared tag (on hiatus; opt-in via --consensus).
    reductions = global_cfg.get('consensus') or []
    for red in reductions:
        if red not in CONSENSUS_REDUCTIONS:
            raise SystemExit(f'Unknown consensus reduction {red!r}; choose '
                             f'from {sorted(CONSENSUS_REDUCTIONS)}.')
    cons_records = []
    for tag, entries in by_tag.items():
        if not reductions:
            break
        if len(entries) < 2:
            print(f'Consensus: skipping {tag} (only {len(entries)} map).')
            continue
        recs = [r for r, _ in entries]
        labels = [r['ensemble_id'] for r in recs]
        cm = ConsensusMap([vm for _, vm in entries], probe_labels=labels)
        stem = naming.common_id(recs)
        for red in reductions:
            cvm = CONSENSUS_REDUCTIONS[red](cm)
            cvm.map_id = f'{stem}_consensus_{red}'
            cvm.to_dx(os.path.join(outdir, f'{cvm.map_id}.dx'))
            cons_records.append({'map_id': cvm.map_id, 'map_tag': tag,
                                 'reduction': red, 'probes': labels,
                                 'dx_path': cvm.dx_path})
            print(f'Consensus [{tag}] {red} over {labels} -> {cvm.dx_path}')

    # 6. Manifests: structured (JSON/YAML) + flat table (CSV), same run tag.
    manifest_id = (global_cfg.get('manifest_id')
                   or naming.common_id(map_records))
    manifest = {
        'manifest_id': manifest_id,
        'created': created,
        'config': {k: global_cfg.get(k) for k in (
            'grid_from', 'grid_expand',
            'grid_pdb', 'grid_selection', 'include_receptor', 'voxel_size',
            'padding', 'consensus')},
        'grid': {'origin': grid.origin.tolist(),
                 'shape': list(grid.shape),
                 'voxel_size': grid.voxel_size},
        'ensembles': ens_records,
        'maps': map_records,
        'consensus': cons_records,
    }
    stem = os.path.join(outdir, f'{manifest_id}_map_manifest')
    fmt = str(global_cfg.get('manifest_format') or 'json').lower()
    if fmt in ('yaml', 'yml'):
        import yaml
        manifest_path = f'{stem}.yaml'
        with open(manifest_path, 'w') as f:
            yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    else:
        manifest_path = f'{stem}.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    _write_manifest_csv(f'{stem}.csv', map_records)
    print(f'Wrote manifest to {manifest_path} (+ {stem}.csv)')
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description='Generate volumetric (and consensus) maps for one or more '
                    'GlycanDock ensembles docked against the same receptor.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Flat mode covers same-maps-for-all and sweeps; use --config '
               'for heterogeneous per-ensemble map parameters. Put --consensus '
               'last (or after the pose sources) so it does not swallow them.')

    parser.add_argument('posedirs', nargs='*',
                        help='One or more pose sources (all vs the same '
                             'receptor): a directory of decoy PDBs and/or '
                             'block archives, or a single .tar.xz/.tar.gz '
                             'archive. Omit if using --config.')
    parser.add_argument('--config',
                        help='JSON (or YAML) job spec for heterogeneous '
                             'per-ensemble/per-map parameters. When given, flat '
                             'map/grid args are ignored.')

    # Flat-mode map specification:
    parser.add_argument('--map-type', nargs='+', default=['boltzmann'],
                        choices=sorted(MAP_TYPES),
                        help='Map type(s) to produce for every ensemble '
                             '(default: boltzmann).')
    parser.add_argument('--beta', nargs='+', type=float, default=[DEFAULT_BETA],
                        help='Beta value(s) for Boltzmann maps; multiple '
                             f'values produce a sweep (default: {DEFAULT_BETA}).')
    parser.add_argument('-range', '--poserange', type=int, nargs=2,
                        metavar=('START', 'STOP'),
                        help='Inclusive pose-id range applied to every '
                             'source (default: all poses).')
    parser.add_argument('--run-id', nargs='+',
                        help='run_id per pose source, overriding the name '
                             'inferred from the archive/directory '
                             '(<receptor>-<ligand>[_<grid_id>]).')
    parser.add_argument('--interface-energies', nargs='+', metavar='PARQUET',
                        help='Precomputed interface-energy parquet per pose '
                             'source, for residue_boltzmann (else computed '
                             'with PyRosetta).')

    # Global grid / output:
    parser.add_argument('--grid-from', metavar='MANIFEST_OR_DX',
                        help='Reuse the grid of earlier maps so new maps are '
                             'voxel-aligned with them (e.g. adding probes to '
                             'a mapped set). Give their map manifest (.json/'
                             '.yaml; full-precision origin, preferred) or a '
                             '.dx map.')
    parser.add_argument('--grid-expand', action='store_true',
                        help='With --grid-from: grow that grid by whole voxels '
                             'if any pose falls outside it (keeps the '
                             'lattice, so overlapping voxels still align).')
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
                        help='(On hiatus.) Also write consensus maps across '
                             'ensembles for each shared map tag. Reductions: '
                             + ', '.join(sorted(CONSENSUS_REDUCTIONS))
                             + '. With no value given, defaults to: '
                             + ' '.join(DEFAULT_CONSENSUS) + '.')
    parser.add_argument('--no-json', action='store_true',
                        help='Do not write per-map <map_id>.json sidecars.')
    parser.add_argument('--manifest-format', choices=['json', 'yaml'],
                        default='json',
                        help='Format of the structured run manifest (default: '
                             'json; yaml requires pyyaml). A CSV table is '
                             'always written alongside it.')
    parser.add_argument('--manifest-id',
                        help='Stem for <manifest_id>_map_manifest.{json,csv}. '
                             'Default: built from the fields all maps share.')

    args = parser.parse_args()

    try:
        if args.config:
            global_cfg, jobs = load_config(args.config)
            global_cfg.setdefault('voxel_size', 1.0)
            global_cfg.setdefault('padding', 5.0)
            global_cfg.setdefault('outdir', '.')
            global_cfg.setdefault('manifest_format', 'json')
        else:
            if not args.posedirs:
                parser.error('Provide one or more pose sources, or --config.')
            # `--consensus` with no values -> the sensible default set.
            if args.consensus is not None and len(args.consensus) == 0:
                args.consensus = list(DEFAULT_CONSENSUS)
            global_cfg, jobs = build_flat(args)
    except ValueError as e:
        parser.error(str(e))

    run(global_cfg, jobs)


if __name__ == '__main__':
    exit(main())
