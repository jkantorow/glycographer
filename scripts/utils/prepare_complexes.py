#!/usr/bin/env python

'''
Prepare receptor-glycoligand complexes for input to the GlycanDockProtocol
mover.

Two input modes, one clean-up:

  probe mode    -rec R1 [R2 ...] -lig L1 [L2 ...]
                Append every glycoligand to every receptor (probe docking,
                or any binder posed in a separate file).

  complex mode  -c C1 [C2 ...] [--lig-sel 'chain X']
                Clean up complexes that were assembled elsewhere, e.g. a
                native binder already posed in a receptor model. The ligand
                is picked out with a PyMOL selection; everything else is the
                receptor.

The receptor clean-up exists because Rosetta (with -ignore_unrecognized_res)
silently drops any residue it does not recognize. Receptors that come out of
MD/AMBER tooling name residues by protonation or bonding state (HIE, HID, HIP,
CYX, ...), and losing them leaves holes in the receptor with no error raised.
For every receptor this script:

  - removes solvent and keeps only the first alternate location
  - renames force-field residue names to standard PDB names (RESN_ALIASES).
    Rosetta re-derives what they encoded: disulfides are re-detected from
    SG-SG geometry, and the packer samples His tautomers itself
  - strips hydrogens (unless --keep-hydrogens); Rosetta rebuilds them with
    its own naming
  - sets occupancy to 1.0 (MD exports often write 0.00, which Rosetta
    ignores by default)
  - renumbers residues sequentially if collapsing a multimer onto chain A
    would collide residue numbers, writing a <out>_resmap.csv back to the
    original chain/residue numbering
  - sets the receptor to chain A and the glycoligand to chain X
  - orders protein first, then hetero residues (metal ions etc.), then the
    glycoligand, with a TER at every protein chain break. Without the TERs
    Rosetta would join separate monomers with a peptide bond, and ions
    written between monomers hide the next monomer's N-terminus

Residue names still unknown after clean-up are reported, because Rosetta
will either fail on them or (with -ignore_unrecognized_res) drop them.
'''

import pymol
from pymol import cmd
import argparse
import csv
import itertools
import math
import os
import sys

STANDARD_AA = ('ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL')

# Metal ions in the Rosetta fa_standard database (set up via -in:auto_setup_metals).
METAL_IONS = ('CA', 'MG', 'ZN', 'MN', 'FE', 'FE2', 'CU', 'CO', 'NI', 'NA', 'K', 'LI')

# Force-field residue names (AMBER, CHARMM, GROMACS) -> standard PDB names.
RESN_ALIASES = {
    'HIE': 'HIS', 'HID': 'HIS', 'HIP': 'HIS',                  # AMBER His tautomers
    'HSE': 'HIS', 'HSD': 'HIS', 'HSP': 'HIS',                  # CHARMM
    'HISE': 'HIS', 'HISD': 'HIS', 'HISH': 'HIS',               # GROMACS
    'CYX': 'CYS', 'CYM': 'CYS',                                # disulfide / deprotonated Cys
    'ASH': 'ASP', 'GLH': 'GLU', 'LYN': 'LYS',                  # protonation variants
}

PROTEIN_SEL = f'resn {"+".join(STANDARD_AA)}'
MAX_PEPTIDE_CN = 2.0  # Angstrom; a longer C(i)-N(i+1) gap is a chain break

def _resolve_altlocs_and_occupancy(obj):
    cmd.remove(f'{obj} and not alt ""+A')
    cmd.alter(obj, "alt=''")
    cmd.alter(obj, 'q=1.0')

def _renumber_if_colliding(obj):
    '''
    Collapsing every receptor chain onto chain A collides residue numbers when
    chains share a numbering (e.g. four CRD monomers each numbered 255-380).
    Renumber sequentially in that case and return the old->new mapping;
    return None if no renumbering was needed.
    '''
    resns = {}
    cmd.iterate(obj, '_ = resns.setdefault((segi, chain, resi), resn)', space={'resns': resns})

    resis = [key[2] for key in resns]
    if len(set(resis)) == len(resis):
        return None

    remap = {key: i for i, key in enumerate(resns, start=1)}  # file order
    cmd.alter(obj, 'resv = remap[(segi, chain, resi)]', space={'remap': remap})
    return [(new, *key, resns[key]) for key, new in remap.items()]

def prep_receptor(receptor_pdb, keep_hydrogens=False, name=None):
    '''
    Load and prepare input receptor for docking. Returns the PyMOL object
    name and a dict summarizing the clean-up.
    '''
    rec_name = name or os.path.basename(receptor_pdb.replace('.pdb', ''))
    if receptor_pdb is not None:
        cmd.load(receptor_pdb, rec_name)

    report = {'solvent_removed': cmd.count_atoms(f'{rec_name} and solvent')}
    cmd.remove(f'{rec_name} and solvent')
    _resolve_altlocs_and_occupancy(rec_name)

    renamed = {}
    for alias, std in RESN_ALIASES.items():
        n = cmd.count_atoms(f'{rec_name} and resn {alias} and name CA')
        if n:
            cmd.alter(f'{rec_name} and resn {alias}', f"resn='{std}'")
            renamed[alias] = n
    report['renamed'] = renamed

    if not keep_hydrogens:
        report['hydrogens_removed'] = cmd.count_atoms(f'{rec_name} and hydro')
        cmd.remove(f'{rec_name} and hydro')

    report['resmap'] = _renumber_if_colliding(rec_name)
    cmd.alter(rec_name, "chain='A'")
    cmd.alter(rec_name, "segi=''")

    resns = set()
    cmd.iterate(rec_name, 'resns.add(resn)', space={'resns': resns})
    report['unrecognized'] = sorted(resns - set(STANDARD_AA) - set(METAL_IONS))

    return rec_name, report

def prep_glycoligand(gly_pdb, name=None):
    '''
    Load and prepare the glycoligand structure to be appended
    to each receptor.
    '''
    gly_name = name or os.path.basename(gly_pdb.replace('.pdb', ''))
    if gly_pdb is not None:
        cmd.load(gly_pdb, gly_name)
    cmd.remove(f'{gly_name} and solvent')
    _resolve_altlocs_and_occupancy(gly_name)
    cmd.alter(gly_name, "chain='X'")
    cmd.alter(gly_name, "segi=''")

    return gly_name

def cat_gly_to_rec(rec_name, gly_name, complex_name=None):
    '''
    Append a loaded glycoligand structure to the end of
    a loaded receptor structure.
    '''
    complex_name = complex_name or f'{rec_name}_{gly_name}'
    cmd.create(complex_name, rec_name, zoom=0)
    cmd.copy_to(complex_name, gly_name, zoom=0)
    cmd.alter(complex_name, "segi=''")  # copy_to can (re)assign segi

    return complex_name

def _xyz(residue_lines, atom_name):
    for ln in residue_lines:
        if ln[12:16].strip() == atom_name:
            return float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
    return None

def _res_key(line):
    return line[17:27] if line.startswith(('ATOM', 'HETATM')) else None

def add_chain_break_ters(pdbstr):
    '''
    Insert a TER between consecutive protein residues that are not peptide
    bonded (C-N > MAX_PEPTIDE_CN, or either atom missing). Returns the new
    PDB string and the number of TERs inserted.
    '''
    out, last_protein, n_inserted = [], None, 0
    for key, group in itertools.groupby(pdbstr.splitlines(), key=_res_key):
        group = list(group)
        if key is None:
            if any(ln.startswith('TER') for ln in group):
                last_protein = None
            out.extend(group)
            continue

        if key[:3].strip() in STANDARD_AA:
            if last_protein is not None:
                c, n = _xyz(last_protein, 'C'), _xyz(group, 'N')
                if c is None or n is None or math.dist(c, n) > MAX_PEPTIDE_CN:
                    if not out[-1].startswith('TER'):
                        out.append('TER')
                        n_inserted += 1
            last_protein = group
        out.extend(group)

    return '\n'.join(out) + '\n', n_inserted

def write_complex(complex_name, out_pdb):
    '''
    Write protein, then receptor hetero residues, then the chain-X glycoligand,
    with a TER at every protein chain break. Returns the number of TERs added.
    '''
    retain_order = cmd.get('retain_order')
    cmd.set('retain_order', 1)
    try:
        cmd.alter(f'{complex_name} and chain A and not ({PROTEIN_SEL})', 'rank = rank + 1000000')
        cmd.alter(f'{complex_name} and chain X', 'rank = rank + 2000000')
        cmd.sort(complex_name)
        pdbstr = cmd.get_pdbstr(complex_name)
    finally:
        cmd.set('retain_order', retain_order)

    pdbstr, n_ters = add_chain_break_ters(pdbstr)
    with open(out_pdb, 'w') as f:
        f.write(pdbstr)

    return n_ters

def write_resmap(resmap, out_csv):
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['resi', 'orig_segi', 'orig_chain', 'orig_resi', 'resn'])
        writer.writerows(resmap)

def print_report(label, report, n_ters, out_pdb):
    print(f'{label} -> {out_pdb}')
    if report['solvent_removed']:
        print(f'  removed {report["solvent_removed"]} solvent atoms')
    for alias, n in report['renamed'].items():
        print(f'  renamed {n} {alias} -> {RESN_ALIASES[alias]}')
    if report.get('hydrogens_removed'):
        print(f'  removed {report["hydrogens_removed"]} receptor hydrogens')
    if report['resmap']:
        print(f'  renumbered {len(report["resmap"])} receptor residues sequentially '
              '(chains shared residue numbers)')
    if n_ters:
        print(f'  inserted {n_ters} TER records at protein chain breaks')
    if report['unrecognized']:
        print(f'  WARNING: receptor residue names Rosetta may not recognize: '
              f'{", ".join(report["unrecognized"])}', file=sys.stderr)

def out_path(name, outprefix):
    return f'{outprefix}_{name}.pdb' if outprefix else f'{name}.pdb'

def finish(complex_name, label, report, outprefix):
    out_pdb = out_path(complex_name, outprefix)
    n_ters = write_complex(complex_name, out_pdb)
    if report['resmap']:
        write_resmap(report['resmap'], out_pdb.replace('.pdb', '_resmap.csv'))
    print_report(label, report, n_ters, out_pdb)

def main():

    parser = argparse.ArgumentParser(
        description='Create receptor-glycoligand complexes suited for glycan docking via the '
                    'Rosetta GlycanDockProtocol mover.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''examples:
  # probe mode: every receptor x every probe
  prepare_complexes.py -rec siglec7.pdb -lig probes/*.pdb -o probe

  # complex mode: native binder already posed (as chain X) in the receptor
  prepare_complexes.py -c tetra-native1k9iB.pdb tetra-native1sl5B.pdb -o prepared

  # complex mode, ligand picked out of a crystal structure
  prepare_complexes.py -c 1k9i.pdb --lig-sel "chain E and resn MAN+NAG"
''')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('-rec', '--receptor-structures', type=str, nargs='+',
                        help='Receptor structures to process in PDB format (probe mode).')
    inputs.add_argument('-c', '--complexes', type=str, nargs='+',
                        help='Pre-assembled receptor-glycoligand complexes in PDB format '
                             '(complex mode).')
    parser.add_argument('-lig', '--glycoligand-structures', type=str, nargs='+',
                        help='Glycoligand structures to append to each provided receptor in PDB '
                             'format (probe mode).')
    parser.add_argument('--lig-sel', type=str, default='chain X',
                        help='PyMOL selection identifying the glycoligand in each complex '
                             '(complex mode; default: "chain X").')
    parser.add_argument('-o', '--outprefix', type=str, default=None,
                        help='Optional prefix to append to each output complex.')
    parser.add_argument('--keep-hydrogens', action='store_true',
                        help='Keep receptor hydrogens instead of letting Rosetta rebuild them.')

    args = parser.parse_args()
    if args.receptor_structures and not args.glycoligand_structures:
        parser.error('-rec requires -lig')
    if args.complexes and args.glycoligand_structures:
        parser.error('-lig is only used with -rec; in complex mode use --lig-sel')

    pymol.finish_launching(['pymol', '-c'])

    if args.complexes:
        for complex_pdb in args.complexes:
            name = os.path.basename(complex_pdb.replace('.pdb', ''))
            cmd.load(complex_pdb, f'{name}_in')
            lig_sel = f'{name}_in and ({args.lig_sel})'
            if not cmd.count_atoms(lig_sel):
                raise SystemExit(f'{complex_pdb}: no atoms match --lig-sel "{args.lig_sel}"')
            cmd.create(f'{name}_rec', f'{name}_in and not ({args.lig_sel})', zoom=0)
            cmd.create(f'{name}_lig', lig_sel, zoom=0)

            rec_name, report = prep_receptor(None, args.keep_hydrogens, name=f'{name}_rec')
            gly_name = prep_glycoligand(None, name=f'{name}_lig')
            complex_name = cat_gly_to_rec(rec_name, gly_name, complex_name=name)
            finish(complex_name, complex_pdb, report, args.outprefix)
            cmd.delete(f'{name}*')
        return 0

    lig_names = {lig_pdb: prep_glycoligand(lig_pdb) for lig_pdb in args.glycoligand_structures}

    for rec_pdb in args.receptor_structures:
        rec_name, report = prep_receptor(rec_pdb, args.keep_hydrogens)

        for lig_pdb, lig_name in lig_names.items():
            complex_name = cat_gly_to_rec(rec_name=rec_name, gly_name=lig_name)
            finish(complex_name, f'{rec_pdb} + {lig_pdb}', report, args.outprefix)
            cmd.delete(complex_name)

    return 0

if __name__ == '__main__':
    exit(main())
else:
    cmd.extend('prep_receptor', prep_receptor)
    cmd.extend('prep_glycoligand', prep_glycoligand)
    cmd.extend('cat_gly_to_rec', cat_gly_to_rec)
