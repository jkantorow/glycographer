#!/usr/bin/env python

'''
Generate complexes between input receptors and glycoligands and prepare
them for input to the GlycanDockProtocol mover.
'''

import pymol
from pymol import cmd
import argparse
import os

def prep_receptor(receptor_pdb):
    '''
    Load and prepare input receptor for docking.
    '''
    rec_name = os.path.basename(receptor_pdb.replace('.pdb', ''))
    cmd.load(receptor_pdb, rec_name)
    cmd.remove(f'resn HOH and {rec_name}')
    cmd.alter(rec_name, "chain='A'")

    return rec_name

def prep_glycoligand(gly_pdb):
    '''
    Load and prepare the glycoligand structure to be appended
    to each receptor.
    '''
    gly_name = os.path.basename(gly_pdb.replace('.pdb', ''))
    cmd.load(gly_pdb, gly_name)
    cmd.alter(gly_name, "chain='X'")

    return gly_name

def cat_gly_to_rec(rec_name, gly_name):
    '''
    Append a loaded glycoligand structure to the end of
    a loaded receptor structure and output the complex
    as a pdb.
    '''
    complex_name = f'{rec_name}_{gly_name}'
    cmd.create(complex_name, rec_name, zoom=0)
    cmd.copy_to(complex_name, gly_name, zoom=0)

    return complex_name

def main():

    parser = argparse.ArgumentParser(description='Create receptor-glycoligand complexes suited for glycan docking via the Rosetta GlycanDockProtocol mover.')
    parser.add_argument('-rec', '--receptor-structures', type=str, nargs='+', required=True,
                        help='Receptor structures to process in PDB format.')
    parser.add_argument('-lig', '--glycoligand-structures', type=str, nargs='+', required=True,
                        help='Glycoligand structures to append to each provided receptor in PDB format.')
    parser.add_argument('-o', '--outprefix', type=str, default=None,
                        help='Optional prefix to append to each output complex.')
    
    args = parser.parse_args()

    pymol.finish_launching(['pymol', '-c'])

    for rec_pdb in args.receptor_structures:
        rec_name = prep_receptor(rec_pdb)

        for lig_pdb in args.glycoligand_structures:
            lig_name = prep_glycoligand(lig_pdb)
            
            complex_name = cat_gly_to_rec(rec_name=rec_name,
                                          gly_name=lig_name)
            
            if args.outprefix:
                out_pdb = f'{args.outprefix}_{complex_name}.pdb'
            else:
                out_pdb = f'{complex_name}.pdb'

            cmd.save(out_pdb, complex_name)

if __name__ == '__main__':
    exit(main())
else:
    cmd.extend('prep_receptor', prep_receptor)
    cmd.extend('prep_glycoligand', prep_glycoligand)
    cmd.extend('cat_gly_to_rec', cat_gly_to_rec)