#!/usr/bin/env python

'''
Load poses from a GlycanDock output, perform various useful processes on the ensemble, and output relevant data. 
'''

import os
import glob
import argparse

from glycographer.dock import GlycanDockEnsemble

def main():

    parser = argparse.ArgumentParser(
        description='Load poses from a GlycanDock output, perform various useful processes on the ensemble, and output relevant data'
    )
    parser.add_argument('posedir', type=str, default='.',
                        help='Directory where output poses are located (default: cwd)')
    parser.add_argument('-range', '--poserange', type=int, nargs=2, required=False,
                        help='A numeric range representing the pose ids to use for contructing the ensemble (defaults to loading all poses detected within posedir)')
    parser.add_argument('-o', '--outprefix', type=str, default=os.path.basename(os.getcwd()),
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
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--dump_pdb', action='store_true',
                        help='Dump PDB with glycan info')

    args = parser.parse_args()

    if args.poserange:
        start, stop = args.poserange[0], args.poserange[1]
        pose_list = glob.glob(os.path.join(args.posedir, 'GLOB PATTERN FOR START AND STOP'))
    else:
        pose_list = glob.glob(os.path.join(args.posedir, '*.pdb'))

    ens = GlycanDockEnsemble.from_poses(
        pose_list=pose_list,
        run_id=args.outprefix,
        parse_energies=False if args.no_parse_energies else True
    )

    ens.to_pdb(os.path.join(args.outdir, '_'.join(args.prefix, f'{ens._n_poses}p', 'ensemble.pdb')))
    ens.scale_inteng()

    if args.no_parse_energies:
        pass
    else:
        ens.energies_to_parquet(os.path.join(args.outdir, '_'.join(args.outprefix, 'residue_energies.parquet')))

    if args.no_cluster:
        pass
    else:
        ens.cluster_poses(
            rmsd_cutoff=args.cluster_cutoff,
            min_cluster_size=args.min_cluster_size
        )
    
    if args.no_write_scoredata:
        pass
    else:
        ens.scores_to_csv(os.path.join(args.outdir, '_'.join(args.prefix, 'scoredata.csv')))


if __name__ == '__main__':
    exit(main())