#!/usr/bin/env python

from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.protocols.analysis import GlycanInfoMover
import argparse

'''
Example usage in bash:

$ python get_glycan_info.py <structure.pdb> --options <options.init> --verbose --dump_pdb
'''

def glycan_info_from_pdb(pdb_input, options=None, verbose=False, dump_pdb=False):

    '''
    Intended to take a pdb containing glycans and output
    Rosetta-specific information about how the glycans
    are being processed (tree set, glycosidic linkages,
    root definition, etc.)
    '''

    if not options:
        options = '''
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
        init(" ".join(options.split('\n')))
    else:
        init(f'@{options}')

    pose = pose_from_pdb(pdb_input)

    tree_set = pose.glycan_tree_set()

    n_trees = tree_set.n_trees()
    largest_tree_lengh = tree_set.get_largest_glycan_tree_length()

    print('\n \n')
    print(f'Number of trees found in pose: {n_trees}')
    print(f'Number of residues in largest tree found: {largest_tree_lengh} \n')
    
    print('POSE GLYCAN INFO OVERVIEW:')
    gly_info = GlycanInfoMover()
    gly_info.apply(pose)
    
    if verbose:

        print('\n')
        print('TREES FROM EACH STARTING RESIDUE... \n')

        for i, st in enumerate(tree_set.get_start_points()):

            rosetta_start_res_name = pose.residue_type(st).name3()
            tree = tree_set.get_tree(st)
            tree_length = tree.size()
            root = tree.get_root()

            print('\n')
            print(f'TREE_{i+1}: \n')
            print(f'Starting residue {st} ({rosetta_start_res_name}) (PDB code {pose.pdb_info().pose2pdb(st)})')
            print(f'Number of glycan residues: {tree_length}')
            print(f'Glycan root: {root} (0 if no residue is defined as root)')

            if root:
                root_name = pose.residue_type(root).name3()
                root_pdb = pose.pdb_info().pose2pdb(root)
                print(f'Root glycosylated at {root_name} {root} (PDB code {root_pdb})')

            for gly_res in tree.get_residues():

                res, chain, ext = pose.pdb_info().pose2pdb(gly_res).split(' ')
                full_name = pose.residue_type(gly_res).carbohydrate_info().full_name()
                res_info = pose.residue(gly_res)

                print('\n')
                print(f'RESIDUE {gly_res} OF TREE_{i+1} (tree start res {st})\n')
                print(f'(PDB residue {res}, chain {chain})')
                print(f'Full IUPAC name: {full_name} \n')
                print('Verbose info:')
                print(res_info)
    
    if dump_pdb:
        pose.dump_pdb('gly_info_out.pdb')

def main():

    parser = argparse.ArgumentParser(description='Get glycan information from a PDB file.')
    parser.add_argument('pdb_input', type=str, help='Input PDB file')
    parser.add_argument('--options', type=str, default=None, help='Rosetta options file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--dump_pdb', action='store_true', help='Dump PDB with glycan info')

    args = parser.parse_args()

    glycan_info_from_pdb(args.pdb_input, args.options, args.verbose, args.dump_pdb)

if __name__ == '__main__':
    main()