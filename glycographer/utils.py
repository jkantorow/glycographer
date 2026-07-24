'''
Tools for assisting the construction of and interaction between glycographer objects.
'''

import os
import re
import glob

# Pose files are named like "<run_id>_0001.pdb"; capture the trailing
# integer id that sits between the final underscore and the .pdb suffix.
# The run_id itself may contain underscores, so anchor on the *last* one.
POSE_NUM_RE = re.compile(r'_(\d+)\.pdb$')


def build_pose_list(posedir, poserange=None):
    '''
    Collect output pose files from `posedir`, optionally trimmed to a
    numeric id range.

    Parameters
    ----------
    posedir : str
        Directory containing the GlycanDock output poses (``*.pdb``).
    poserange : (int, int), optional
        Inclusive ``(start, stop)`` range of pose ids to keep, matched
        against the trailing number in each filename (e.g. the ``2`` in
        ``run_id_0002.pdb``). If None, every ``.pdb`` in `posedir` is used.

    Returns
    -------
    list of str
        Paths to the selected pose files, sorted by pose id.
    '''
    all_poses = glob.glob(os.path.join(posedir, '*.pdb'))

    if poserange is None:
        return sorted(all_poses)

    start, stop = poserange
    selected = []
    for path in all_poses:
        match = POSE_NUM_RE.search(os.path.basename(path))
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
                  key=lambda p: int(POSE_NUM_RE.search(os.path.basename(p)).group(1)))