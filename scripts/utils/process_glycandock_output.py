#!/usr/bin/env python

'''
Load poses from a GlycanDock output, perform various useful processes on the ensemble, and output relevant data.
'''

import os
import re
import sys
import glob
import argparse

# The glycographer package isn't pip-installed (it's an in-development repo),
# so it's only importable when the repo root is on sys.path. This script lives
# at <repo>/scripts/utils/, so the repo root is three directories up. Adding it
# here lets the script run from any cwd (and from the SLURM wrapper).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from glycographer.dock import GlycanDockEnsemble

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


def main():

    parser = argparse.ArgumentParser(
        description='Load poses from a GlycanDock output, perform various useful processes on the ensemble, and output relevant data'
    )
    parser.add_argument('posedir', type=str, default='.',
                        help='Directory where output poses are located (default: cwd)')
    parser.add_argument('-range', '--poserange', type=int, nargs=2, required=False,
                        metavar=('START', 'STOP'),
                        help='Inclusive numeric range of pose ids to use for constructing the ensemble (defaults to loading all poses detected within posedir)')
    parser.add_argument('-o', '--outprefix', type=str, default=None,
                        help='Filename prefix proceeding each specified output (defaults to posedir basename)')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory to write extracted/processed data (default: cwd)')
    parser.add_argument('--no-cluster', action='store_true',
                        help='Specify not to cluster ensemble poses')
    parser.add_argument('--no-write-scoredata', action='store_true',
                        help='Specify not to output extracted scoredata')
    parser.add_argument('--no-parse-energies', action='store_true',
                        help='Specify not to read or dump residue energies per term from output poses')
    parser.add_argument('--cluster-cutoff', type=float, default=2.0,
                        help='RMSD cutoff for assigning poses to a cluster in Angstroms (defalut: 2.0)')
    parser.add_argument('--min-cluster-size', type=int, default=4,
                        help='Minimum number of clustered poses to be considered significant')

    args = parser.parse_args()

    # The prefix defaults to the posedir basename. We can't express this in
    # the argparse default (posedir isn't known until parse time), so resolve
    # it here. normpath strips any trailing slash so basename isn't empty.
    outprefix = args.outprefix or os.path.basename(os.path.normpath(args.posedir))

    pose_list = build_pose_list(args.posedir, args.poserange)
    if not pose_list:
        if args.poserange:
            raise SystemExit(
                f'No poses found in {args.posedir} with ids in range '
                f'{args.poserange[0]}-{args.poserange[1]}.')
        raise SystemExit(f'No .pdb poses found in {args.posedir}.')

    os.makedirs(args.outdir, exist_ok=True)

    ens = GlycanDockEnsemble.from_poses(
        pose_list=pose_list,
        run_id=outprefix,
        parse_energies=not args.no_parse_energies
    )

    ens.to_pdb(os.path.join(args.outdir, '_'.join([outprefix, f'{ens._n_poses}p', 'ensemble.pdb'])))
    ens.scale_inteng()

    if not args.no_parse_energies:
        ens.energies_to_parquet(os.path.join(args.outdir, '_'.join([outprefix, 'residue_energies.parquet'])))

    if not args.no_cluster:
        ens.cluster_poses(
            rmsd_cutoff=args.cluster_cutoff,
            min_cluster_size=args.min_cluster_size
        )

    if not args.no_write_scoredata:
        ens.scores_to_csv(os.path.join(args.outdir, '_'.join([outprefix, 'scoredata.csv'])))


if __name__ == '__main__':
    exit(main())
