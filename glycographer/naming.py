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

Glycan probe names
------------------
The <ligand> field of a run_id is the probe's name, and a probe model is
written as <probe_name>.pdb. glycan_name() derives it from a Rosetta
simplified-IUPAC sequence: every residue becomes its lowercase PDB 3-letter
code (which already encodes the anomer, D/L, ring form and modifications), and
every linkage becomes its two locants with the punctuation dropped:

    a-D-Neup5Ac-(2->3)-b-D-Galp-(1->3)-b-D-GalpNAc          sia23gal13nga
    b-D-Galp-(1->4)-[a-L-Fucp-(1->3)]-b-D-GlcpNAc           gal14{fuc13}nag
    a-D-Manp-(1->3)-[a-D-Manp-(1->6)]-b-D-Manp              man16{man13}bma

Branches are written in Rosetta's own order (a branch sits directly before
the residue it is attached to), with {} in place of []. Square brackets are
glob characters, so a name containing them would silently break glob.glob()
on any path built from it (build_pose_list, run directories); braces are
literal to Python's glob and, with no comma inside, to bash.

The name is canonical: the same glycan written with its branches in a
different order gives the same name. At each branch point the child with the
largest subtree continues the main chain; on a tie, the child linked at the
higher-numbered position does. Probe names never contain '_' (the run_id field
separator) -- see validate_probe_name().
'''

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

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


# ---- glycan probes ---------------------------------------------------------

BRANCH_OPEN, BRANCH_CLOSE = '{', '}'

# Probe names become filenames, the <ligand> field of a run_id, and SLURM /
# PyMOL labels: letters, digits, '-' and the branch braces only. No '_' (it
# separates run_id fields) and no '.' (extensions).
_PROBE_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9{}\-]*$')

_TOKEN_RE = re.compile(r'(\(\s*\d+\s*->\s*\d+\s*\)|\[|\])')
_LINK_RE = re.compile(r'^\(\s*(\d+)\s*->\s*(\d+)\s*\)$')
_RESIDUE_RE = re.compile(
    r'^(?:(?P<anomer>alpha|beta|a|b)-)?'
    r'(?:(?P<stereo>[DL])-)?'
    r'(?P<root>[A-Z][a-z]{2})'
    r'(?P<ring>[pfs])?'
    r'(?P<suffix>[A-Za-z0-9]*)$')
_MOD_RE = re.compile(r'(\d*)(NAc|Ac|Gc|N|A)')

# Suffix affix -> (default position, pdb_sugar.codes modification name).
# Uronic acid ('A') has no position.
_MODS = {
    'NAc': (2, 'AcNH'),
    'N': (2, 'NH3+'),
    'Ac': (5, 'Ac'),
    'Gc': (5, 'HOAc'),
}

# Rosetta's saccharide-sequence parser (core/pose/annotated_sequence.cc)
# assumes alpha when the anomer is omitted. D/L, when omitted, comes from the
# sugar database; D is assumed here except for these roots. Neu takes no D/L
# prefix at all.
_DEFAULT_STEREO = {'Fuc': 'L', 'Rha': 'L'}
_NO_STEREO = {'Neu'}

# Curated from Rosetta's database/input_output/3-letter_codes/pdb_sugar.codes
# -- the table -alternate_3_letter_codes pdb_sugar writes output with:
#   code  root  anomer  D/L  ring  modifications
_CODE_TABLE = [
    ('GLC', 'Glc', 'alpha', 'D', 'p', ()),
    ('BGC', 'Glc', 'beta', 'D', 'p', ()),
    ('NDG', 'Glc', 'alpha', 'D', 'p', ('2-AcNH',)),
    ('NAG', 'Glc', 'beta', 'D', 'p', ('2-AcNH',)),
    ('PA1', 'Glc', 'alpha', 'D', 'p', ('2-NH3+',)),
    ('GCS', 'Glc', 'beta', 'D', 'p', ('2-NH3+',)),
    ('GCU', 'Glc', 'alpha', 'D', 'p', ('uronic_acid',)),
    ('BDP', 'Glc', 'beta', 'D', 'p', ('uronic_acid',)),
    ('MAN', 'Man', 'alpha', 'D', 'p', ()),
    ('BMA', 'Man', 'beta', 'D', 'p', ()),
    ('BM3', 'Man', 'alpha', 'D', 'p', ('2-AcNH',)),
    ('GLA', 'Gal', 'alpha', 'D', 'p', ()),
    ('GAL', 'Gal', 'beta', 'D', 'p', ()),
    ('A2G', 'Gal', 'alpha', 'D', 'p', ('2-AcNH',)),
    ('NGA', 'Gal', 'beta', 'D', 'p', ('2-AcNH',)),
    ('ADA', 'Gal', 'alpha', 'D', 'p', ('uronic_acid',)),
    ('GTR', 'Gal', 'beta', 'D', 'p', ('uronic_acid',)),
    ('FUC', 'Fuc', 'alpha', 'L', 'p', ()),
    ('FUL', 'Fuc', 'beta', 'L', 'p', ()),
    ('RAM', 'Rha', 'alpha', 'L', 'p', ()),
    ('XYS', 'Xyl', 'alpha', 'D', 'p', ()),
    ('XYP', 'Xyl', 'beta', 'D', 'p', ()),
    ('LXC', 'Xyl', 'beta', 'L', 'p', ()),
    ('ARA', 'Ara', 'alpha', 'L', 'p', ()),
    ('ARB', 'Ara', 'beta', 'L', 'p', ()),
    ('64K', 'Ara', 'alpha', 'D', 'p', ()),
    ('AHR', 'Ara', 'alpha', 'L', 'f', ()),
    ('FUB', 'Ara', 'beta', 'L', 'f', ()),
    ('BXY', 'Ara', 'alpha', 'D', 'f', ()),
    ('BXX', 'Ara', 'beta', 'D', 'f', ()),
    ('RIB', 'Rib', 'alpha', 'D', 'f', ()),
    ('RIP', 'Rib', 'alpha', 'D', 'p', ()),
    ('ALL', 'All', 'beta', 'D', 'p', ()),
    ('G6D', 'Qui', 'alpha', 'D', 'p', ()),
    ('ABE', 'Abe', 'alpha', 'D', 'p', ()),
    ('MUR', 'Mur', 'beta', 'D', 'p', ()),
    ('FRU', 'Fru', 'beta', 'D', 'f', ()),
    ('SOE', 'Sor', 'alpha', 'L', 'p', ()),
    ('KDO', 'Kdo', 'alpha', 'D', 'p', ()),
    ('SIA', 'Neu', 'alpha', None, 'p', ('5-Ac',)),
    ('SLB', 'Neu', 'beta', None, 'p', ('5-Ac',)),
    ('NGC', 'Neu', 'alpha', None, 'p', ('5-HOAc',)),
    ('NGE', 'Neu', 'beta', None, 'p', ('5-HOAc',)),
]
PDB_SUGAR_CODES = {(root, anomer, stereo, ring, frozenset(mods)): code
                   for code, root, anomer, stereo, ring, mods in _CODE_TABLE}


class GlycanSequenceError(ValueError):
    '''A sequence that is not valid Rosetta simplified IUPAC.'''


class UnknownResidueError(ValueError):
    '''A residue with no entry in PDB_SUGAR_CODES, so it cannot be auto-named.'''


@dataclass(eq=False)
class GlycanResidue:
    '''
    One monosaccharide of a parsed sequence. `link` is its linkage to its
    parent, (donor carbon, acceptor position), e.g. (1, 4) for (1->4); None
    for the reducing end.
    '''
    token: str
    root: str
    anomer: str
    stereo: Optional[str]
    ring: Optional[str]
    mods: frozenset
    link: Optional[Tuple[int, int]] = None
    children: List['GlycanResidue'] = field(default_factory=list)

    @property
    def pdb_code(self) -> str:
        key = (self.root, self.anomer, self.stereo, self.ring, self.mods)
        try:
            return PDB_SUGAR_CODES[key]
        except KeyError:
            mods = ', '.join(sorted(self.mods)) or 'none'
            raise UnknownResidueError(
                f'no PDB code known for {self.token!r} ({self.anomer}, '
                f'{self.stereo or "no D/L"}, ring {self.ring or "none"}, '
                f'modifications: {mods}). Give this probe an explicit name, '
                f'or add the residue to naming._CODE_TABLE from Rosetta\'s '
                f'pdb_sugar.codes.') from None

    def walk(self):
        '''This residue and every residue above it (toward the non-reducing ends).'''
        yield self
        for child in self.children:
            yield from child.walk()

    def size(self) -> int:
        return sum(1 for _ in self.walk())


def _parse_residue(token: str, seq: str) -> GlycanResidue:
    m = _RESIDUE_RE.match(token)
    if not m:
        raise GlycanSequenceError(
            f'cannot read residue {token!r} in {seq!r}. Expected e.g. '
            f'b-D-GlcpNAc; two residues need a (x->y) linkage between them.')
    root = m['root']
    anomer = {'a': 'alpha', 'b': 'beta'}.get(m['anomer'], m['anomer']) or 'alpha'
    if root in _NO_STEREO:
        stereo = None
    else:
        stereo = m['stereo'] or _DEFAULT_STEREO.get(root, 'D')

    mods, suffix, pos = set(), m['suffix'], 0
    for mm in _MOD_RE.finditer(suffix):
        if mm.start() != pos:
            break
        pos = mm.end()
        if mm[2] == 'A':
            mods.add('uronic_acid')
        else:
            default_pos, patch = _MODS[mm[2]]
            mods.add(f'{mm[1] or default_pos}-{patch}')
    if pos != len(suffix):
        raise UnknownResidueError(
            f'unrecognized modification suffix {suffix!r} on {token!r}; give '
            f'this probe an explicit name.')

    return GlycanResidue(token=token, root=root, anomer=anomer, stereo=stereo,
                         ring=m['ring'], mods=frozenset(mods))


def _tokenize(seq: str) -> list:
    tokens = []
    for part in _TOKEN_RE.split(''.join(seq.split())):
        if part in ('[', ']'):
            tokens.append(part)
            continue
        link = _LINK_RE.match(part)
        if link:
            tokens.append(('link', int(link[1]), int(link[2])))
            continue
        part = part.strip('-')
        if part:
            tokens.append(('res', part))
    return tokens


def _parse_chain(tokens: list, i: int, seq: str, depth: int):
    '''
    Read one chain (the whole sequence, or the inside of a [...] branch)
    starting at tokens[i]. A residue's children are the residue that precedes
    it at the same level with a linkage, plus the top residue of each [...]
    branch immediately before it.

    Returns (reducing-end residue, next index) for the top level, and
    (dangling residues, next index) for a branch.
    '''
    pending, last = [], None
    while i < len(tokens):
        tok = tokens[i]
        if tok == '[':
            if last is not None:
                raise GlycanSequenceError(
                    f'missing linkage before "[" after {last.token!r} in {seq!r}.')
            dangling, i = _parse_chain(tokens, i + 1, seq, depth + 1)
            if len(dangling) != 1:
                raise GlycanSequenceError(f'malformed branch in {seq!r}.')
            pending.extend(dangling)
            continue
        if tok == ']':
            if depth == 0:
                raise GlycanSequenceError(f'unbalanced "]" in {seq!r}.')
            if last is not None or not pending:
                raise GlycanSequenceError(
                    f'a [...] branch must end with the linkage to the residue '
                    f'after it, e.g. [a-L-Fucp-(1->3)], in {seq!r}.')
            return pending, i + 1
        if tok[0] == 'res':
            if last is not None:
                raise GlycanSequenceError(
                    f'missing linkage between {last.token!r} and {tok[1]!r} '
                    f'in {seq!r}.')
            last = _parse_residue(tok[1], seq)
            last.children, pending = pending, []
        else:
            if last is None:
                raise GlycanSequenceError(
                    f'linkage ({tok[1]}->{tok[2]}) does not follow a residue '
                    f'in {seq!r}.')
            last.link = (tok[1], tok[2])
            pending.append(last)
            last = None
        i += 1
    if depth:
        raise GlycanSequenceError(f'unclosed "[" in {seq!r}.')
    if last is None:
        raise GlycanSequenceError(
            f'{seq!r} must end with its reducing-end residue, not a linkage.')
    return last, i


def parse_glycan_sequence(seq: str) -> GlycanResidue:
    '''
    Parse a Rosetta simplified-IUPAC sequence (non-reducing end first, branches
    in [...] directly before the residue they attach to) into a residue tree.
    Returns the reducing-end residue. Pure Python: no PyRosetta needed.

    Defaults follow Rosetta's parser: an omitted anomer is alpha, so
    'NeupAc-(2->3)-...' is alpha-Neu5Ac (SIA). An omitted D/L is assumed D
    (L for Fuc and Rha); write it explicitly for anything else.
    '''
    if not seq or not seq.strip():
        raise GlycanSequenceError('empty glycan sequence.')
    root, _ = _parse_chain(_tokenize(seq), 0, seq, 0)
    return root


def _linkage(res: GlycanResidue) -> str:
    return f'{res.link[0]}{res.link[1]}'


def _render(res: GlycanResidue) -> str:
    if not res.children:
        return res.pdb_code.lower()
    rendered = [(c, _render(c)) for c in res.children]
    # Main chain: largest subtree, then the higher-numbered acceptor position.
    rendered.sort(key=lambda cr: (-cr[0].size(), -cr[0].link[1], cr[1]))
    (main, main_str), branches = rendered[0], rendered[1:]
    out = main_str + _linkage(main)
    for child, child_str in branches:
        out += BRANCH_OPEN + child_str + _linkage(child) + BRANCH_CLOSE
    return out + res.pdb_code.lower()


def glycan_name(seq: str) -> str:
    '''
    Canonical probe name for a Rosetta simplified-IUPAC sequence (see the
    module docstring), e.g. sia23gal13nga. Raises UnknownResidueError when a
    residue has no known PDB code.
    '''
    return _render(parse_glycan_sequence(seq))


def glycan_pdb_codes(seq: str) -> Counter:
    '''PDB 3-letter codes the sequence's model should contain, with counts.'''
    return Counter(r.pdb_code for r in parse_glycan_sequence(seq).walk())


def validate_probe_name(name: str) -> str:
    '''Return `name` if it is usable as a probe name, else raise ValueError.'''
    if not _PROBE_NAME_RE.match(name or ''):
        raise ValueError(
            f'invalid probe name {name!r}: use letters, digits, "-" and "{{}}" '
            f'only. "_" separates run_id fields (<receptor>-<ligand>_<grid_id>), '
            f'so it cannot appear in a probe name.')
    return name


def probe_names(sequences: Sequence[str],
                names: Optional[Sequence[Optional[str]]] = None) -> List[str]:
    '''
    Names for a batch of probe sequences: each explicit name (validated) where
    given, else glycan_name(). Raises ValueError, listing every problem at
    once, if any name cannot be made or two sequences end up with the same
    name -- e.g. the same glycan listed twice, perhaps with its branches
    written in a different order.
    '''
    names = list(names) if names is not None else [None] * len(sequences)
    if len(names) != len(sequences):
        raise ValueError('sequences and names differ in length')

    out, problems = [], []
    for i, (seq, name) in enumerate(zip(sequences, names), 1):
        try:
            out.append(validate_probe_name(name) if name else glycan_name(seq))
        except ValueError as e:
            out.append(None)
            problems.append(f'  #{i} {seq}: {e}')

    seen = {}
    for i, name in enumerate(out, 1):
        if name is not None:
            seen.setdefault(name, []).append(i)
    for name, idx in seen.items():
        if len(idx) > 1:
            problems.append(
                f'  name {name!r} is shared by sequences '
                f'{", ".join(f"#{i}" for i in idx)}; drop the duplicate or '
                f'give one an explicit name.')

    if problems:
        raise ValueError('cannot name probe sequences:\n' + '\n'.join(problems))
    return out
