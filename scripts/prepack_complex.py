#!/usr/bin/env python3

'''
Run GlycanDock Stage 0 pre-packing on a receptor-glycoligand complex.

Stage 0 separates the docking partners, repacks each in its unbound state, and
re-forms the complex -- so the result barely depends on where the glycoligand
started, and one prepacked structure can serve an entire docking run.

This is a separate entry point rather than a step inside run_glycandock.py for
a reason that cannot be worked around: the published GlycanDock protocol
prepacks with fine rotamer sampling (-ex1 -ex2 -ex3 -ex4 -ex1aro -ex2aro) and
refines with only -ex1 -ex2. Rosetta takes its options once per process, so the
two stages cannot share a session. Fine sampling is affordable once per complex
and ruinous once per decoy.

Feed the output to run_glycandock.py with --prepack-mode once:

    prepack_complex.py complex.pdb -o complex_prepacked.pdb
    run_glycandock.py complex_prepacked.pdb -n 1000 --grid grid.pdb \
        --prepack-mode once

NOTE: every glycographer ensemble generated so far used the legacy in-loop
prepack ('per_decoy', still the default), under the coarser -ex1 -ex2 flags.
Decoys produced this way are not guaranteed comparable with those -- validate
before mixing them in a single analysis.
'''

import argparse
import os

from glycographer.sample import GlycanDockConfig, STAGE_0_OPTIONS, prepack


def build_parser():
    parser = argparse.ArgumentParser(
        description='Pre-pack a receptor-glycoligand complex (GlycanDock Stage 0).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''examples:
  prepack_complex.py complex.pdb
  prepack_complex.py complex.pdb -o prepacked/1o7v_prepacked.pdb --native xtal.pdb
''')

    parser.add_argument('complex', type=str,
                        help='Receptor-glycoligand complex in PDB format.')
    parser.add_argument('-o', '--out', type=str, default=None,
                        help='Output PDB path (default: <complex>_prepacked.pdb '
                             'beside the input).')
    parser.add_argument('--native', type=str, default=None,
                        help='Native crystal complex, passed as -in:file:native.')
    parser.add_argument('--options', type=str, default=STAGE_0_OPTIONS,
                        help='Rosetta flags file (default: config/glycandock_stage_0.init).')
    parser.add_argument('--partners', type=str, default='A_X',
                        help='Docking partner chain ids (default: A_X).')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed. Pre-packing is stochastic, so set this if you need '
                             'the prepacked structure to be reproducible.')
    parser.add_argument('--config', type=str, default=None,
                        help='Load protocol parameters from a JSON config. Only `partners` and '
                             '`slide` affect Stage 0.')
    return parser


def main():
    args = build_parser().parse_args()

    if not os.path.exists(args.complex):
        raise SystemExit(f'Input complex not found: {args.complex}')

    if args.config:
        config = GlycanDockConfig.from_json(args.config)
    else:
        config = GlycanDockConfig(partners=args.partners)

    out = args.out
    if not out:
        base = args.complex[:-4] if args.complex.endswith('.pdb') else args.complex
        out = f'{base}_prepacked.pdb'

    prepack(args.complex, out,
            config=config,
            options_file=args.options,
            native=args.native,
            seed=args.seed)
    return 0


if __name__ == '__main__':
    exit(main())
