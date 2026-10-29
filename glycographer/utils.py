'''
Tools for assisting the construction of and interaction between glycographer objects.
'''

import os
import re
import glob
from collections import Counter

from glycographer.archive import is_archive, list_archive_poses, pose_name

# Pose files are named like "<run_id>_0001.pdb"; capture the trailing
# integer id that sits between the final underscore and the .pdb suffix.
# The run_id itself may contain underscores, so anchor on the *last* one.
POSE_NUM_RE = re.compile(r'_(\d+)\.pdb$')


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


def init_glycandock(complex, nstruct=1, n_cycles=1, native=None, options=None):
    '''
    Start Rosetta session with necessary structure and
    options input.
    '''
    in_flags = f'''
    -in:file:s {complex}
    -nstruct {nstruct}
    -n_cycles {n_cycles}
    '''
    if native is not None:
        in_flags = in_flags + f'-in:file:native {native}'

    # Lazy import init so that utils.py can be imported without having pyrosetta installed
    from pyrosetta import init

    if not options:
        options = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'glycandock_stage_1_stage_2.init')
    
    init(f'{in_flags} @{options}')