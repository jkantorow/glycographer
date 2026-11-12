#!/usr/bin/env python3

'''
Model energetically minimized free-glycan probes from Rosetta simplified-IUPAC
sequences and write each one as <outdir>/<probe_name>.pdb.

This is a thin CLI over glycographer.model (modeling) and glycographer.naming
(probe names). Input is either one sequence or a file of sequences, one per
line, each optionally followed by an explicit name:

    model_glycan_probes.py 'a-Neup5Ac-(2->3)-b-D-Galp-(1->3)-b-D-GalpNAc'
        -> ./sia23gal13nga.pdb
    model_glycan_probes.py probes.iupac -o probes/

Names are built from the residues' PDB 3-letter codes and linkage locants,
with branches in {} (see glycographer/naming.py and docs/probe_modeling.md),
so they slot straight into a run_id as <receptor>-<probe>_<grid_id>.

Use --dry-run to check the names and output paths without PyRosetta.
'''

import argparse
import os
import sys

# The HPC env is not guaranteed to have the editable install, so put the repo
# root (one directory up from <repo>/scripts/) on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from glycographer.model import (
    MODELING_OPTIONS, init_modeling, probe_specs, read_sequence_file,
    write_probe_models,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Model minimized free-glycan probes from simplified-IUPAC '
                    'sequences and write them to PDB.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''examples:
  model_glycan_probes.py 'b-D-Galp-(1->4)-[a-L-Fucp-(1->3)]-b-D-GlcpNAc'
  model_glycan_probes.py 'a-Neup5Ac-(2->3)-b-D-Galp' -n 3sl -o probes/
  model_glycan_probes.py probes.iupac -o probes/ --manifest probes/probes.csv
  model_glycan_probes.py probes.iupac --dry-run
''')
    parser.add_argument('sequences', type=str,
                        help='A simplified-IUPAC sequence, or a file of them '
                             '(one per line, optionally followed by a name; '
                             '# comments allowed). Quote sequences on the '
                             'command line: they contain ( ) [ ] >.')
    parser.add_argument('-o', '--outdir', type=str, default='.',
                        help='Directory for the <name>.pdb models (default: .).')
    parser.add_argument('-n', '--name', type=str, default=None,
                        help='Explicit probe name for a single input sequence '
                             '(default: built from the sequence). For a file, '
                             'put names in a second column instead.')
    parser.add_argument('--options', type=str, default=MODELING_OPTIONS,
                        help='Rosetta flags file (default: config/glycan_modeling.init).')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed. GlycanSampler starts from a randomized '
                             'conformation, so set this for reproducible models.')
    parser.add_argument('--no-minimize', action='store_true',
                        help='Write the idealized built pose without GlycanSampler.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Remodel probes whose PDB already exists in --outdir '
                             '(default: skip them).')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Write a CSV of name, sequence, pdb, status and score '
                             'for every probe.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print each probe\'s name and output path, then exit '
                             'without initializing Rosetta.')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if os.path.isfile(args.sequences):
        if args.name:
            parser.error('--name applies to a single sequence; for a file, put '
                         'names in a second column.')
        entries = read_sequence_file(args.sequences)
    elif args.sequences.endswith(('.iupac', '.txt')):
        parser.error(f'sequence file not found: {args.sequences}')
    else:
        entries = [(args.sequences, args.name)]

    try:
        specs = probe_specs(entries)
    except ValueError as e:
        raise SystemExit(f'error: {e}')

    if args.dry_run:
        width = max(len(s.name) for s in specs)
        for s in specs:
            exists = '  (exists)' if os.path.exists(s.pdb_path(args.outdir)) else ''
            print(f'{s.name:<{width}}  {s.pdb_path(args.outdir)}  {s.sequence}{exists}')
        print(f'{len(specs)} probe(s); dry run, nothing modeled.')
        return 0

    init_modeling(args.options, seed=args.seed)
    records = write_probe_models(specs, args.outdir,
                                 minimize=not args.no_minimize,
                                 overwrite=args.overwrite,
                                 manifest=args.manifest)

    n_failed = sum(r['status'] == 'failed' for r in records)
    n_mismatch = sum(r['code_check'].startswith('mismatch') for r in records)
    print(f'{len(records)} probe(s): '
          f'{sum(r["status"] == "ok" for r in records)} modeled, '
          f'{sum(r["status"] == "exists" for r in records)} existing, '
          f'{n_failed} failed'
          + (f', {n_mismatch} with residue-code mismatches' if n_mismatch else ''))
    return 1 if n_failed else 0


if __name__ == '__main__':
    exit(main())
