#!/usr/bin/env python

'''
Load poses from a GlycanDock output, cluster the docked ligand poses, and dump
the per-pose scores, cluster summary, ensemble PDB, and residue energies.

Two input modes:

  1. From raw poses (the normal post-docking step). Parses every pose once and
     clusters straight from the in-memory coordinates:

       process_glycandock_output.py runs/1k9i_full --outdir processed \\
           --cluster-cutoff 2.0 --min-cluster-size 6

  2. Re-cluster previously processed output (e.g. to try other cutoffs)
     without re-parsing thousands of pose files:

       process_glycandock_output.py \\
           --ensemble-pdb processed/1k9i_full_5000p_ensemble.pdb \\
           --scores-csv processed/1k9i_full_scoredata.csv \\
           --outdir processed/c1p5 --cluster-cutoff 1.5

Outputs (prefix defaults to the posedir / ensemble basename):
  <prefix>_<N>p_ensemble.pdb        multimodel ligand ensemble (mode 1 only)
  <prefix>_scoredata.csv            per-pose scores + cluster_id,
                                    cluster_center, cluster_rmsd_to_center
                                    (+ ref_rmsd with --reference)
  <prefix>_clusters.csv             one row per cluster (size, center/best
                                    pose, score stats, and RMSD stats vs the
                                    native if the run used --native and/or
                                    vs --reference)
  <prefix>_residue_energies.parquet per-residue REF15 terms (mode 1 only)

The ensemble PDB and scoredata CSV are the inputs for PyMOL cluster
visualization (scripts/vis/vis_clusters.py).
'''

import os
import sys
import argparse

# The glycographer package isn't pip-installed (it's an in-development repo),
# so it's only importable when the repo root is on sys.path. This script lives
# at <repo>/scripts/utils/, so the repo root is three directories up. Adding it
# here lets the script run from any cwd (and from the SLURM wrapper).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from glycographer.utils import build_pose_list
from glycographer.dock import GlycanDockEnsemble
from glycographer.analysis import cluster_ensemble_poses, rmsd_to_reference


def _load_ensemble(args, parser):
    '''Build the ensemble from raw poses or from previously dumped files.'''
    if args.posedir:
        if args.ensemble_pdb or args.scores_csv:
            parser.error('Give either posedir or --ensemble-pdb/--scores-csv, '
                         'not both.')
        pose_list = build_pose_list(args.posedir, args.poserange)
        if not pose_list:
            if args.poserange:
                raise SystemExit(
                    f'No poses found in {args.posedir} with ids in range '
                    f'{args.poserange[0]}-{args.poserange[1]}.')
            raise SystemExit(f'No .pdb poses found in {args.posedir}.')
        outprefix = args.outprefix or os.path.basename(os.path.normpath(args.posedir))
        ens = GlycanDockEnsemble.from_poses(
            pose_list=pose_list,
            run_id=outprefix,
            parse_energies=not args.no_parse_energies,
        )
        return ens, outprefix, True

    if not (args.ensemble_pdb and args.scores_csv):
        parser.error('Provide a pose directory, or both --ensemble-pdb and '
                     '--scores-csv to re-cluster processed output.')
    outprefix = args.outprefix or (
        os.path.basename(args.scores_csv).replace('_scoredata.csv', '')
        .replace('.csv', ''))
    ens = GlycanDockEnsemble.from_files(
        run_id=outprefix,
        ensemble_pdb=args.ensemble_pdb,
        scores_csv=args.scores_csv,
        run_type='full',
    )
    return ens, outprefix, False


def main():

    parser = argparse.ArgumentParser(
        description='Load poses from a GlycanDock output, cluster the docked '
                    'ligand poses, and dump scores, clusters, and energies.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Clustering uses in-place (no superposition) heavy-atom ligand '
               'RMSD, since decoys share the fixed receptor frame.')

    # Input
    parser.add_argument('posedir', type=str, nargs='?', default=None,
                        help='Directory where output poses are located. Omit '
                             'when re-clustering with --ensemble-pdb/--scores-csv.')
    parser.add_argument('-range', '--poserange', type=int, nargs=2, required=False,
                        metavar=('START', 'STOP'),
                        help='Inclusive numeric range of pose ids to use for constructing the ensemble (defaults to loading all poses detected within posedir)')
    parser.add_argument('--ensemble-pdb', type=str, default=None,
                        help='Re-cluster mode: multimodel ensemble PDB written by a previous run')
    parser.add_argument('--scores-csv', type=str, default=None,
                        help='Re-cluster mode: scoredata CSV written by a previous run (same pose order as --ensemble-pdb)')

    # Output
    parser.add_argument('-o', '--outprefix', type=str, default=None,
                        help='Filename prefix proceeding each specified output (defaults to posedir basename)')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory to write extracted/processed data (default: cwd)')
    parser.add_argument('--no-write-ensemble', action='store_true',
                        help='Specify not to write the multimodel ensemble PDB (not needed for clustering, but used for PyMOL visualization)')
    parser.add_argument('--no-write-scoredata', action='store_true',
                        help='Specify not to output extracted scoredata')
    parser.add_argument('--no-parse-energies', action='store_true',
                        help='Specify not to read or dump residue energies per term from output poses')

    # Clustering
    parser.add_argument('--no-cluster', action='store_true',
                        help='Specify not to cluster ensemble poses')
    parser.add_argument('--cluster-method', choices=['gromos', 'leader'], default='gromos',
                        help='gromos: neighbor-count clustering, largest cluster first, order independent (default). '
                             'leader: leader-follower, best-scoring poses become leaders first.')
    parser.add_argument('--cluster-cutoff', type=float, default=2.0,
                        help='RMSD cutoff for assigning poses to a cluster in Angstroms (default: 2.0)')
    parser.add_argument('--min-cluster-size', type=int, default=6,
                        help='Minimum number of clustered poses to be considered significant (default: 6)')
    parser.add_argument('--cluster-score', type=str, default='interaction_energy',
                        help='Score used to break ties, order leaders, and summarize clusters; lower is better (default: interaction_energy)')
    parser.add_argument('--include-hydrogens', action='store_true',
                        help='Include hydrogens in the clustering and reference RMSDs (default: heavy atoms only)')

    # Reference pose
    parser.add_argument('--reference', type=str, default=None,
                        help='PDB with a known ligand pose (e.g. a crystal probe pose) in the receptor frame. '
                             'Adds in-place ref_rmsd to scoredata and the cluster summary. Use when the run '
                             'had no --native; otherwise GlycanDock already reports heavy_Lrmsd.')
    parser.add_argument('--reference-selection', type=str, default=None,
                        help='MDAnalysis selection for the ligand within --reference (default: chain X if present, else all atoms)')
    parser.add_argument('--reference-match', choices=['resid', 'order'], default='resid',
                        help='Pair reference atoms by residue number + atom name (resid, default) or by residue order + atom name (order)')
    parser.add_argument('--reference-allow-missing', action='store_true',
                        help='Compute over the atoms that pair instead of failing when some do not')
    parser.add_argument('--reference-superimpose', action='store_true',
                        help='Also write ref_rmsd_fit, the RMSD after superimposing each pose on the reference (conformation only)')

    args = parser.parse_args()

    ens, outprefix, from_poses = _load_ensemble(args, parser)
    os.makedirs(args.outdir, exist_ok=True)

    def out(name):
        return os.path.join(args.outdir, f'{outprefix}_{name}')

    print(f'[{outprefix}] loaded {ens._n_poses} poses')

    if from_poses and not args.no_write_ensemble:
        path = ens.to_pdb(out(f'{ens._n_poses}p_ensemble.pdb'))
        print(f'[{outprefix}] wrote ensemble -> {path}')

    ens.scale_inteng()

    if from_poses and not args.no_parse_energies:
        path = ens.energies_to_parquet(out('residue_energies.parquet'))
        print(f'[{outprefix}] wrote residue energies -> {path}')

    if args.reference:
        for superimpose in ([False, True] if args.reference_superimpose else [False]):
            rmsd = rmsd_to_reference(
                ens, args.reference,
                superimpose=superimpose,
                reference_selection=args.reference_selection,
                match=args.reference_match,
                allow_missing=args.reference_allow_missing,
                heavy_only=not args.include_hydrogens,
            )
            print(f'[{outprefix}] {rmsd.name} to {args.reference} over '
                  f'{rmsd.attrs["n_atoms"]} atoms: min {rmsd.min():.2f}, '
                  f'median {rmsd.median():.2f} A')

    if not args.no_cluster:
        summary = cluster_ensemble_poses(
            ens,
            rmsd_cutoff=args.cluster_cutoff,
            min_cluster_size=args.min_cluster_size,
            method=args.cluster_method,
            score_name=args.cluster_score,
            heavy_only=not args.include_hydrogens,
        )
        path = out('clusters.csv')
        summary.to_csv(path, index=False)
        a = summary.attrs
        print(f'[{outprefix}] {len(summary)} clusters ({a["method"]}, '
              f'cutoff {a["rmsd_cutoff"]} A, min size {a["min_cluster_size"]}); '
              f'{a["n_clustered"]}/{a["n_poses"]} poses clustered -> {path}')
        if not summary.empty:
            print(summary.head(10).to_string(index=False))

    if not args.no_write_scoredata:
        path = ens.scores_to_csv(out('scoredata.csv'))
        print(f'[{outprefix}] wrote scoredata -> {path}')


if __name__ == '__main__':
    exit(main())
