'''
Tools for assisting the construction of and interaction between glycographer objects.

Also home to the generic Rosetta session helpers (`init_rosetta`, the option
file locations under config/) shared by every PyRosetta-facing module --
sample.py (docking) and model.py (probe modeling). PyRosetta is imported lazily
inside init_rosetta(), so this module stays importable without it.
'''

import os
import re
import glob
from collections import Counter
from typing import Optional

from glycographer.archive import is_archive, list_archive_poses, pose_name

# Pose files are named like "<run_id>_0001.pdb"; capture the trailing
# integer id that sits between the final underscore and the .pdb suffix.
# The run_id itself may contain underscores, so anchor on the *last* one.
POSE_NUM_RE = re.compile(r'_(\d+)\.pdb$')

# --- option file locations -------------------------------------------------
# config/ currently lives at the repo root, beside the package. Under the
# editable install this resolves to the live source tree; see the note in
# pyproject.toml about eventually moving it inside the package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, 'config')


def config_path(name: str) -> str:
    '''Path of an options file in config/, e.g. config_path('glycan_modeling.init').'''
    return os.path.join(CONFIG_DIR, name)


def build_pose_list(posedir, poserange=None):
    '''
    Collect output poses from `posedir`, optionally trimmed to a numeric id
    range.

    Parameters
    ----------
    posedir : str
        Directory containing the GlycanDock output poses -- loose ``*.pdb``
        files and/or tar archives of them, such as the per-block
        ``<run_id>_blockNNN.tar.xz`` -- or the path of a single archive.
    poserange : (int, int), optional
        Inclusive ``(start, stop)`` range of pose ids to keep, matched
        against the trailing number in each filename (e.g. the ``2`` in
        ``run_id_0002.pdb``). If None, every ``.pdb`` in `posedir` is used.

    Returns
    -------
    list of str
        Pose paths, sorted by pose id. Poses inside an archive are given as
        ``<archive>::<member>`` (see glycographer.archive).
    '''
    if is_archive(posedir) and os.path.isfile(posedir):
        all_poses = list_archive_poses(posedir)
    else:
        all_poses = glob.glob(os.path.join(posedir, '*.pdb'))
        if os.path.isdir(posedir):
            for name in sorted(os.listdir(posedir)):
                if is_archive(name):
                    all_poses.extend(
                        list_archive_poses(os.path.join(posedir, name)))

    # The same decoy in two places (e.g. a merged archive next to the block
    # archives it was made from) would silently enter the ensemble twice.
    dupes = [n for n, c in Counter(map(pose_name, all_poses)).items() if c > 1]
    if dupes:
        raise ValueError(
            f'{len(dupes)} pose(s) found more than once under {posedir}, e.g. '
            f'{sorted(dupes)[0]}. Remove the duplicate archive or directory.')

    if poserange is None:
        return sorted(all_poses, key=pose_name)

    start, stop = poserange
    selected = []
    for path in all_poses:
        match = POSE_NUM_RE.search(pose_name(path))
        if match is None:
            # Skip files that don't carry a numeric pose id (e.g. an
            # input complex or grid pdb that happens to live alongside
            # the output poses).
            continue
        pose_num = int(match.group(1))
        if start <= pose_num <= stop:
            selected.append(path)

    # Sort by the parsed pose id so the ensemble model order is numeric
    # rather than lexicographic (matters once ids exceed the zero-pad width).
    return sorted(selected,
                  key=lambda p: int(POSE_NUM_RE.search(pose_name(p)).group(1)))


# --- Rosetta session --------------------------------------------------------
def seed_flags(seed: Optional[int]) -> str:
    '''
    Rosetta flags pinning the RNG seed.

    `-run:jran` only takes effect together with `-run:constant_seed`; without
    it Rosetta derives a seed from the system clock / urandom and ignores jran.

    VERIFY ONCE ON THE CLUSTER: run the same block twice with the same seed and
    confirm the decoys are identical, then run two blocks with different seeds
    and confirm they differ. Silent seed collisions across array tasks would
    duplicate decoys without raising anything, and would not be obvious in the
    output -- an energy map built from duplicated samples just looks sharper
    than it should.

    Pass seed=None to fall back to Rosetta's default non-deterministic seeding.
    '''
    if seed is None:
        return ''
    return f'-run:constant_seed -run:jran {int(seed)}'


def read_option_flags(options_file: str) -> set:
    '''Flag names (the leading -token of each line) declared in a Rosetta options file.'''
    flags = set()
    with open(options_file) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line.startswith('-'):
                flags.add(line.split()[0])
    return flags


def init_rosetta(options_file: str, complex_pdb: Optional[str] = None, *,
                 native: Optional[str] = None,
                 n_cycles: Optional[int] = None,
                 cst_file: Optional[str] = None,
                 seed: Optional[int] = None,
                 extra_flags: str = '',
                 mute: bool = False) -> str:
    '''
    Start a Rosetta session from an options file (typically one in config/).

    A process gets exactly one of these: Rosetta reads its options once, at
    init. Steps that need different flags -- Stage 0 prepacking (-ex3 -ex4
    -ex1aro -ex2aro) vs Stage 1/2 refinement (-ex1 -ex2), or probe modeling
    (config/glycan_modeling.init) vs docking -- therefore have to run as
    separate processes, not as separate function calls in one process.

    Parameters
    ----------
    options_file : str
        Rosetta flags file -- e.g. sample.STAGE_0_OPTIONS,
        sample.STAGE_1_2_OPTIONS, model.MODELING_OPTIONS, or a caller-supplied
        path.
    complex_pdb : str, optional
        Passed as -in:file:s. Redundant with an explicit pose_from_pdb() call,
        but some option-system defaults key off it, and it records the input in
        the Rosetta log.
    native : str, optional
        Reference complex for RMSD/Fnat scoring (-in:file:native). Without it
        Fnat, heavy_Lrmsd, ring_Lrmsd and the other native-relative scores come
        out as NA.
    n_cycles : int, optional
        GlycanDock Stage 2 outer ramping cycles. Applied here rather than on
        the mover because GlycanDockProtocol exposes no setter for it; see
        sample._SETTERS.
    cst_file : str, optional
        Constraint file (-cst_fa_file), e.g. AtomPair restraints holding a
        known glycan in its site during refinement. Not used for probe
        sampling, where restraining the probe would defeat the scan.
    seed : int, optional
        RNG seed; see seed_flags. Use sample.block_seed() to derive one per
        docking block.
    extra_flags : str
        Any further flags, appended before the options file.
    mute : bool
        Add -mute all.

    Returns
    -------
    str : the full flag string passed to pyrosetta.init(), for provenance.
    '''
    from pyrosetta import init

    flags = []
    if complex_pdb:
        flags.append(f'-in:file:s {complex_pdb}')
    if native:
        flags.append(f'-in:file:native {native}')
    if n_cycles is not None:
        flags.append(f'-n_cycles {int(n_cycles)}')
    if cst_file:
        flags.append(f'-cst_fa_file {cst_file}')
    # -nstruct never controls how many structures glycographer makes -- the
    # callers' own loops do. It is supplied because Rosetta's job machinery
    # expects it to be set.
    flags.append('-nstruct 1')
    seed_str = seed_flags(seed)
    if seed_str:
        flags.append(seed_str)
    if mute:
        flags.append('-mute all')
    if extra_flags:
        flags.append(extra_flags)
    flags.append(f'@{options_file}')

    flag_string = ' '.join(flags)
    init(flag_string)
    return flag_string
