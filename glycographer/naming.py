'''
Standard names for glycographer runs, ensembles, maps and manifests.

Every identifier is built from (and can be parsed back into) the same fields,
so a filename alone says what produced it:

    run_id       <receptor>-<ligand>[_<grid_id>]              1o7v-gal_global
    ensemble_id  <run_id>_<n_poses>p                          1o7v-gal_global_1000p
    map_tag      <map_type>[_b<beta>]                         boltzmann_b0p5
    map_id       <ensemble_id>_<map_tag>                      1o7v-gal_global_1000p_boltzmann_b0p5

Files are named after the identifier they hold:

    <ensemble_id>_ensemble.pdb, <ensemble_id>_scores.csv
    <map_id>.dx, <map_id>.json                (map + its metadata sidecar)
    <manifest_id>_map_manifest.{json,csv}     (one per map_ensembles.py run)

run_id is the docking run's id (sample.run_block / run_glycandock.py
--run-id; by default the input complex's basename). The pose count is NOT part
of run_id: it is appended exactly once, from the number of poses actually
used, so a 1000-pose subset of a 5000-pose run is named ..._1000p.

Numbers in names use 'p' for the decimal point and a leading 'm' for minus
(0.5 -> 0p5, 1.0 -> 1p0, -1.5 -> m1p5).
'''

import os
import re
from typing import Iterable, Optional

from glycographer.archive import strip_archive_ext

# A trailing pose-count tag ("_1000p") or the legacy pose-directory suffix
# ("_5000_poses", "_poses"). Stripped from run ids so the count is never
# repeated and never stale.
_SIZE_TAG_RE = re.compile(r'_\d+p$')
_POSEDIR_TAG_RE = re.compile(r'(?:_\d+)?_poses$')
_BLOCK_RE = re.compile(r'_block\d+$')

_RUN_ID_RE = re.compile(
    r'^(?P<receptor>[^-_]+)-(?P<ligand>[^_]+)(?:_(?P<grid_id>.+))?$')
_MAP_ID_RE = re.compile(
    r'^(?P<ensemble_id>(?P<run_id>.+)_(?P<n_poses>\d+)p)'
    r'_(?P<map_type>[A-Za-z][A-Za-z0-9_]*?)'
    r'(?:_b(?P<beta>m?\d+(?:p\d+)?))?$')


# ---- numbers ---------------------------------------------------------------

def fmt_num(x) -> str:
    '''Filename-safe number: 0.5 -> '0p5', 1 -> '1p0', -1.5 -> 'm1p5'.'''
    return str(float(x)).replace('-', 'm').replace('.', 'p')


def parse_num(s: str) -> float:
    ''''0p5' -> 0.5, 'm1p5' -> -1.5 (inverse of fmt_num).'''
    return float(s.replace('m', '-').replace('p', '.'))


# ---- building --------------------------------------------------------------

def normalize_run_id(name: str) -> str:
    '''
    Reduce a label, directory or archive name to a bare run_id by removing a
    path, an archive extension, a "_blockNNN" suffix, a legacy
    "_<N>_poses"/"_poses" suffix and a trailing "_<N>p" size tag.

        runs/1o7v-gal_global_block003.tar.xz -> 1o7v-gal_global
        1o7v-gal_global_5564_poses           -> 1o7v-gal_global
        1o7v-gal_global_1000p                -> 1o7v-gal_global
    '''
    name = strip_archive_ext(os.path.basename(os.path.normpath(str(name))))
    for pattern in (_BLOCK_RE, _POSEDIR_TAG_RE, _SIZE_TAG_RE):
        name = pattern.sub('', name)
    return name


def ensemble_id(run_id: str, n_poses: int) -> str:
    '''<run_id>_<n>p. Any size tag already on run_id is replaced, not repeated.'''
    return f'{_SIZE_TAG_RE.sub("", run_id)}_{int(n_poses)}p'


def map_tag(map_type: str, beta: Optional[float] = None) -> str:
    '''<map_type>[_b<beta>]; beta is included only when given.'''
    return map_type if beta is None else f'{map_type}_b{fmt_num(beta)}'


def map_id(ens_id: str, map_type: str, beta: Optional[float] = None) -> str:
    '''<ensemble_id>_<map_tag>.'''
    return f'{ens_id}_{map_tag(map_type, beta)}'


# ---- parsing ---------------------------------------------------------------

def parse_run_id(run_id: str) -> dict:
    '''
    Split a run_id into receptor, ligand and grid_id. Fields that cannot be
    recovered (a run_id not of the form <receptor>-<ligand>[_<grid>]) are None.
    '''
    m = _RUN_ID_RE.match(run_id or '')
    out = {'run_id': run_id, 'receptor': None, 'ligand': None, 'grid_id': None}
    if m:
        out.update(m.groupdict())
    return out


def parse_map_id(mid: str) -> dict:
    '''
    Split a map_id (or a .dx filename) into its fields: run_id, receptor,
    ligand, grid_id, ensemble_id, n_poses, map_type, beta. Returns {} if mid
    does not follow the standard pattern.
    '''
    mid = os.path.basename(mid)
    if mid.endswith('.dx'):
        mid = mid[:-3]
    m = _MAP_ID_RE.match(mid)
    if not m:
        return {}
    d = m.groupdict()
    out = parse_run_id(d['run_id'])
    out.update(ensemble_id=d['ensemble_id'], n_poses=int(d['n_poses']),
               map_type=d['map_type'],
               beta=parse_num(d['beta']) if d['beta'] else None)
    return out


def common_id(records: Iterable[dict], default: str = 'glycographer') -> str:
    '''
    Name for a set of maps, built from the fields they all share:

        <receptor>[-<ligand>][_<grid_id>][_<n>p][_<map_type>][_b<beta>]

    A field is kept only if every record agrees on it. One map gives back its
    full map_id; six probes at 1000 poses and one beta give
    "1o7v_global_1000p_boltzmann_b0p5". Falls back to `default` when the
    records share no receptor.
    '''
    records = list(records)

    def shared(key):
        vals = {r.get(key) for r in records}
        return vals.pop() if len(vals) == 1 else None

    receptor = shared('receptor')
    if not records or receptor is None:
        return default
    name = receptor
    ligand = shared('ligand')
    if ligand:
        name += f'-{ligand}'
    grid_id = shared('grid_id')
    if grid_id:
        name += f'_{grid_id}'
    n_poses = shared('n_poses')
    if n_poses:
        name += f'_{n_poses}p'
    map_type = shared('map_type')
    if map_type:
        name += f'_{map_type}'
        beta = shared('beta')
        if beta is not None:
            name += f'_b{fmt_num(beta)}'
    return name
