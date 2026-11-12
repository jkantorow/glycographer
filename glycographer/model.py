'''
Tools to set up atomistic glycoligand (probe) models.

De novo free-glycan probes are built from a Rosetta simplified-IUPAC sequence
with pose_from_saccharide_sequence and relaxed with Rosetta's GlycanSampler,
then written as <probe_name>.pdb, named by glycographer.naming.

    from glycographer.model import init_modeling, read_sequence_file, \
        probe_specs, write_probe_models

    init_modeling(seed=1)
    specs = probe_specs(read_sequence_file('probes.iupac'))
    write_probe_models(specs, outdir='probes')

Rosetta takes its options once per process, and modeling runs under
config/glycan_modeling.init, not the GlycanDock flags -- so in a full pipeline
probe modeling must be its own process (as Stage 0 prepacking is), not a
function call inside a docking job.

PyRosetta is imported lazily inside the functions that need it; reading and
naming sequences works without it.
'''

import csv
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from glycographer import naming
from glycographer.utils import config_path, init_rosetta

MODELING_OPTIONS = config_path('glycan_modeling.init')

MANIFEST_FIELDS = ['name', 'sequence', 'pdb', 'status', 'n_residues',
                   'total_score', 'code_check', 'error']


@dataclass(frozen=True)
class ProbeSpec:
    '''One probe to model: its simplified-IUPAC sequence and output name.'''
    sequence: str
    name: str

    def pdb_path(self, outdir: str = '.') -> str:
        return os.path.join(outdir, f'{self.name}.pdb')


# --- input ------------------------------------------------------------------
def read_sequence_file(path: str) -> List[Tuple[str, Optional[str]]]:
    '''
    Read a probe sequence file: one Rosetta simplified-IUPAC sequence per
    line, optionally followed by whitespace and an explicit probe name.
    Blank lines and '#' comments (whole-line or trailing) are ignored.

        # Lewis X, auto-named gal14{fuc13}nag
        b-D-Galp-(1->4)-[a-L-Fucp-(1->3)]-b-D-GlcpNAc
        a-Neup5Ac-(2->3)-b-D-Galp-(1->4)-b-D-Glcp    3sl

    Returns [(sequence, name or None), ...] in file order.
    '''
    entries = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            fields = line.split('#', 1)[0].split()
            if not fields:
                continue
            if len(fields) > 2:
                raise ValueError(
                    f'{path}:{lineno}: expected "<sequence> [name]", got '
                    f'{len(fields)} fields. Sequences may not contain spaces.')
            entries.append((fields[0], fields[1] if len(fields) == 2 else None))
    if not entries:
        raise ValueError(f'{path}: no sequences found.')
    return entries


def probe_specs(entries: Iterable[Tuple[str, Optional[str]]]) -> List[ProbeSpec]:
    '''
    Name a batch of (sequence, name or None) entries -- e.g. from
    read_sequence_file -- with naming.probe_names. Every naming problem in
    the batch is raised together, before any modeling starts.
    '''
    entries = list(entries)
    seqs = [s for s, _ in entries]
    names = naming.probe_names(seqs, [n for _, n in entries])
    return [ProbeSpec(s, n) for s, n in zip(seqs, names)]


# --- Rosetta ----------------------------------------------------------------
def init_modeling(options_file: str = MODELING_OPTIONS, *,
                  seed: Optional[int] = None, extra_flags: str = '',
                  mute: bool = False) -> str:
    '''
    Start this process's Rosetta session for probe modeling. GlycanSampler
    starts from a randomized conformation, so pass `seed` for reproducible
    models. Returns the flag string, for provenance.
    '''
    return init_rosetta(options_file, seed=seed, extra_flags=extra_flags,
                        mute=mute)


def make_sampler(randomize_first: bool = True):
    '''A GlycanSampler configured as for every glycographer probe model.'''
    from pyrosetta.rosetta.protocols.carbohydrates import GlycanSampler
    sampler = GlycanSampler()
    sampler.set_randomize_first(randomize_first)
    return sampler


def model_glycan(sequence: str, *, name: Optional[str] = None,
                 minimize: bool = True, sampler=None):
    '''
    Build a free-glycan pose from a simplified-IUPAC sequence and, unless
    minimize=False, relax it with GlycanSampler. Rosetta must already be
    initialized (init_modeling).

    Raises ValueError if the sequence builds more than one glycan tree, which
    means a linkage was written wrong rather than that the glycan is branched.
    '''
    from pyrosetta.rosetta.core.pose import pose_from_saccharide_sequence

    pose = pose_from_saccharide_sequence(sequence)
    if name:
        pose.pdb_info().name(name)

    tree_set = pose.glycan_tree_set()
    if tree_set.n_trees() > 1:
        raise ValueError(
            f'{sequence!r} built {tree_set.n_trees()} separate glycan trees; '
            f'check that every residue is joined by a (x->y) linkage.')

    # A monosaccharide has no glycosidic torsions to sample.
    if minimize and tree_set.get_largest_glycan_tree_length() > 1:
        (sampler or make_sampler()).apply(pose)
    return pose


def pdb_residue_codes(pdb_path: str) -> Counter:
    '''Residue names written in a PDB file, counted once per residue.'''
    seen, codes = set(), Counter()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('ATOM  ', 'HETATM')):
                key = (line[21], line[22:27])
                if key not in seen:
                    seen.add(key)
                    codes[line[17:20].strip()] += 1
    return codes


def check_codes(spec: ProbeSpec, pdb_path: str) -> str:
    '''
    Compare the residue codes Rosetta wrote with those the name was built
    from. 'ok', 'skipped' (sequence has residues naming cannot code), or a
    description of the mismatch -- which means naming._CODE_TABLE disagrees
    with Rosetta and the auto-generated name is wrong.
    '''
    try:
        expected = naming.glycan_pdb_codes(spec.sequence)
    except ValueError:
        return 'skipped'
    written = pdb_residue_codes(pdb_path)
    if written == expected:
        return 'ok'
    fmt = lambda c: ' '.join(f'{k}x{v}' for k, v in sorted(c.items()))
    return f'mismatch: wrote {fmt(written)}, expected {fmt(expected)}'


def write_probe_models(specs: Iterable[ProbeSpec], outdir: str = '.', *,
                       minimize: bool = True, overwrite: bool = False,
                       manifest: Optional[str] = None) -> List[Dict]:
    '''
    Model each probe and write it to <outdir>/<name>.pdb. Rosetta must
    already be initialized (init_modeling).

    A failing probe is recorded and the batch carries on. Existing PDBs are
    left alone (status 'exists') unless overwrite=True. If `manifest` is
    given, one CSV row per probe (name, sequence, pdb, status, score, code
    check) is written there.

    Returns the per-probe records; status is 'ok', 'exists' or 'failed'.
    '''
    from pyrosetta import get_fa_scorefxn

    os.makedirs(outdir, exist_ok=True)
    sampler = make_sampler() if minimize else None
    scorefxn = get_fa_scorefxn()
    records = []

    for spec in specs:
        pdb = spec.pdb_path(outdir)
        rec = dict.fromkeys(MANIFEST_FIELDS, '')
        rec.update(name=spec.name, sequence=spec.sequence, pdb=pdb)

        if os.path.exists(pdb) and not overwrite:
            rec['status'] = 'exists'
            print(f'[exists]  {spec.name}: {pdb} (--overwrite to remodel)')
            records.append(rec)
            continue

        try:
            pose = model_glycan(spec.sequence, name=spec.name,
                                minimize=minimize, sampler=sampler)
            pose.dump_pdb(pdb)
            rec.update(status='ok', n_residues=pose.size(),
                       total_score=round(scorefxn(pose), 3),
                       code_check=check_codes(spec, pdb))
            print(f'[ok]      {spec.name}: {pdb} ({pose.size()} residues, '
                  f'score {rec["total_score"]})')
            if rec['code_check'].startswith('mismatch'):
                print(f'  WARNING {spec.name}: residue code {rec["code_check"]}. '
                      f'The probe name may not describe this model.',
                      file=sys.stderr)
        except Exception as e:  # PyRosetta raises RuntimeError for bad residues
            msg = str(e).strip()
            rec.update(status='failed',
                       error=msg.splitlines()[-1] if msg else type(e).__name__)
            print(f'[failed]  {spec.name}: {spec.sequence}\n{msg}',
                  file=sys.stderr)
        records.append(rec)

    if manifest:
        write_manifest(records, manifest)
    return records


def write_manifest(records: List[Dict], path: str) -> str:
    '''Write probe records (from write_probe_models) as CSV.'''
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return path
