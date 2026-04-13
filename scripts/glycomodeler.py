#!/usr/bin/env python3

'''
Generate energetically minimized atomistic free glycan models from
a glycan sequence in simplified IUPAC format using Rosetta's
GlycanSampler mover.
'''

from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.core.pose import pose_from_saccharide_sequence
from pyrosetta.rosetta.protocols.carbohydrates import GlycanSampler
import argparse

def init_rosetta(options=None):
    '''
    Initialize CLI Rosetta environment and pass input options.
    Default options for free glycan modeling are passed if none
    are provided.
    '''
    if not options:
        options = '''
        -beta
        -include_sugars
        -auto_detect_glycan_connections
        -alternate_3_letter_codes pdb_sugar
        -maintain_links
        -write_pdb_link_records
        -write_glycan_pdb_codes
        -ignore_unrecognized_res
        -ignore_zero_occupancy false
        -load_PDB_components false
        -no_fconfig
        '''
        init(' '.join(options.split('\n')))
    else:
        init(f'@{options}')

def parse_iupac_file(iupac_file: str) -> list:
    '''
    Take an input standard text file containing
    glycan simplified IUPAC sequences per each line
    and return them as a list.
    '''
    with open(iupac_file, 'r') as f:
        lines = f.readlines()

    return [line.strip() for line in lines if line and not line.startswith('#')]

def main():

    parser = argparse.ArgumentParser(description='Output minimized free glycan models from simplified IUPAC notation to PDB.')
    parser.add_argument('iupac', type=str,
                        help='Simplified glycan IUPAC as raw string or .iupac file containing each sequence per line.')
    parser.add_argument('-o', '--outprefix', type=str, default='glycan_model',
                        help='Output prefix for each file.')
    parser.add_argument('--options', type=str, default=None,
                        help='Option flags passed to Rosetta for initiation (default: Handled internally).')
    args = parser.parse_args()

    # Get a list of all sequences to construct:
    if args.iupac.endswith(('.iupac', '.txt')):
        sequences = parse_iupac_file(args.iupac)
    else:
        sequences = [args.iupac]
    
    # Initialize Rosetta and the GlycanSampler:
    init_rosetta(args.options)
    minimizer = GlycanSampler()
    minimizer.set_randomize_first(True)

    for i, seq in enumerate(sequences):

        # Generate raw pose from iupac sequence:
        pose = pose_from_saccharide_sequence(seq)
        pose.pdb_info().name('_'.join([args.outprefix, str(i)]))

        # Check Rosetta glycan tree information:
        tree_set = pose.glycan_tree_set()
        n_trees = tree_set.n_trees()
        if n_trees > 1:
            raise Warning(f'Warning: Input sequence {i} ({seq}) resulted in a model consisting of more than 1 glycan tree; ensure syntax specifies correct linkages.')
        
        # Only minimize poses consisting of more than 1 residue:
        gly_len = tree_set.get_largest_glycan_tree_length()
        if gly_len > 1:
            minimizer.apply(pose)

        # Dump the pose as a pdb file:
        outpdb = f'{args.outprefix}_{i}.pdb'
        pose.dump_pdb(outpdb)

if __name__ == '__main__':
    exit(main())