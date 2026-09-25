#!/usr/bin/env python3

'''
Generate GlycanDock decoys for a receptor-glycoligand complex.

This is a thin CLI over glycographer.sample; all of the protocol logic lives in
the module. Two sampling modes, selected by --preset (or inferred from --grid):

- probe       Glycographer grid-probe sampling. A sampling grid is supplied and
              each decoy places the glycoligand fragment at one of the grid's
              coordinates before docking, building up an ensemble across a
              putative binding site.
- refinement  Standard GlycanDock refinement of a known or designed complex.
              Rosetta's GlycanDock has an effective docking range of ~7 A, so
              the input pose is assumed to be near the true binding pose.

Parallelism
-----------
A run of N decoys is split into blocks with --n-blocks; each process runs one
block chosen by --block-index. Under SLURM both default from the array
environment, so a 1000-decoy run across 42 tasks is just:

    sbatch --array=0-41 scripts/slurm/run_glycandock.sh complex.pdb -n 1000 \
        --grid grid.pdb --n-blocks 42

Blocks partition the index range exactly, so tasks never collide on an output
filename and no decoy is dropped. This replaces the manual --start-count-from
submissions.

Use --dry-run to print the plan (block slices, seeds, output names, resolved
config) without initializing Rosetta -- it works on machines without PyRosetta.
'''

import argparse
import os
import sys

from glycographer.sample import (
    GlycanDockConfig, STAGE_1_2_OPTIONS, STAGE_2_OPTIONS,
    block_seed, block_slice, check_setters, decoy_name, default_outprefix,
    run_block,
)


def _env_int(name, default):
    '''Read an integer from the environment, falling back if unset/garbage.'''
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        print(f'Warning: ignoring non-integer {name}={raw!r}', file=sys.stderr)
        return default


_NO_PYROSETTA = '''\
PyRosetta is not available in this environment.

Docking requires it, but the rest of this CLI does not -- use --dry-run to
check a run plan (block slices, seeds, output names, resolved config) on a
machine without PyRosetta, such as a Windows workstation.

PyRosetta is not on PyPI; it is a licensed Graylab build. On the cluster,
activate the glycographer conda env where it is installed.'''


def _require_pyrosetta():
    '''Turn the lazy import failure into an explanation rather than a traceback.'''
    try:
        import pyrosetta  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit(_NO_PYROSETTA)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Output predicted docking poses between a receptor and '
                    'glycoligand, sampled and scored by the Rosetta '
                    'GlycanDockProtocol mover under REF15.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''examples:
  # single process, 100 probe decoys over a sampling grid
  run_glycandock.py complex.pdb -n 100 --grid grid.pdb

  # same run split across 42 SLURM array tasks
  sbatch --array=0-41 scripts/slurm/run_glycandock.sh complex.pdb \\
      -n 1000 --grid grid.pdb --n-blocks 42

  # refine a known complex against a native reference
  run_glycandock.py complex.pdb -n 50 --preset refinement --native xtal.pdb

  # sweep a protocol parameter without a dedicated flag
  run_glycandock.py complex.pdb -n 20 --set mc_kt=0.8 --set rb_rounds=12

  # check the plan without running anything (no PyRosetta needed)
  run_glycandock.py complex.pdb -n 1000 --n-blocks 42 --block-index 7 --dry-run
''')

    parser.add_argument('complex', type=str, nargs='?',
                        help='Receptor-glycoligand complex in PDB format. Receptor and glycoligand '
                             'chain ids should be A and X respectively (see --set partners=...).')
    parser.add_argument('-n', '--nstruct', type=int, default=1,
                        help='Total number of sampled and scored poses to output across the '
                             'whole run, i.e. across all blocks (default: 1).')
    parser.add_argument('-grid', '--grid', type=str, default=None,
                        help='Sampling grid PDB for energy landscape mapping. Not required when '
                             'refining a predetermined input pose.')
    parser.add_argument('-o', '--outprefix', type=str, default=None,
                        help='Prefix for output pose files (default: derived from the complex filename).')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Directory for output poses, timings and manifest (default: cwd).')
    parser.add_argument('--native', type=str, default=None,
                        help='Native crystal complex to score decoys against. Without it, Fnat, '
                             'heavy_Lrmsd, ring_Lrmsd and the other native-relative scores are NA.')

    proto = parser.add_argument_group('protocol')
    proto.add_argument('--preset', choices=('probe', 'refinement', 'publication'),
                       default=None,
                       help="Protocol parameter preset. Default: 'probe' when --grid is given, "
                            "otherwise 'refinement'. 'publication' uses the Nance et al. defaults.")
    proto.add_argument('--config', type=str, default=None,
                       help='Load protocol parameters from a JSON config (as written into a run '
                            'manifest). Overrides --preset; individual flags still apply on top.')
    proto.add_argument('--set', dest='overrides', action='append', default=[],
                       metavar='FIELD=VALUE',
                       help='Override any GlycanDockConfig field, e.g. --set mc_kt=0.8. '
                            'Repeatable, and applied last so it wins over other flags.')
    proto.add_argument('--n-cycles', type=int, default=None,
                       help='Stage 2 outer ramping cycles (preset default: 1 for probe, 10 for refinement).')
    proto.add_argument('--refine-only', action='store_true',
                       help='Skip Stage 1 and run only Stage 2 on the input structure.')
    proto.add_argument('--no-random-start', action='store_true',
                       help='Do not randomize glycoligand orientation before docking.')
    proto.add_argument('--prepack-mode', choices=('per_decoy', 'once', 'none'),
                       default=None,
                       help="Stage 0 handling. 'per_decoy' (default) prepacks inside the loop, "
                            "matching every existing glycographer ensemble. 'once' expects the "
                            "input PDB to be already prepacked by prepack_complex.py.")
    proto.add_argument('--cst-file', type=str, default=None,
                       help='Rosetta constraint file (-cst_fa_file) for restrained refinement. '
                            'Not used for probe sampling.')
    proto.add_argument('--options', type=str, default=None,
                       help='Rosetta flags file. Default: config/glycandock_stage_2.init when '
                            '--refine-only, else config/glycandock_stage_1_stage_2.init.')

    par = parser.add_argument_group('parallelism')
    par.add_argument('--n-blocks', type=int, default=None,
                     help='Split the run into this many blocks (default: $SLURM_ARRAY_TASK_COUNT, else 1).')
    par.add_argument('--block-index', type=int, default=None,
                     help='Which block this process runs, 0-based (default: $SLURM_ARRAY_TASK_ID, else 0).')
    par.add_argument('--start', '--start-count-from', dest='start', type=int, default=1,
                     help='Index of the first decoy of the whole run (default: 1).')
    par.add_argument('--run-id', type=str, default=None,
                     help='Identifier for the whole run; also seeds the RNG. Must be the same '
                          'across all blocks of one run (default: the output prefix).')
    par.add_argument('--seed', type=int, default=None,
                     help='Explicit RNG seed for this block, overriding the derived one. '
                          'Pass -1 to disable seeding and use Rosetta default behavior.')
    par.add_argument('--no-reuse-template', action='store_true',
                     help='Re-read the input PDB for every decoy instead of copying a template '
                          'pose. Slower; restores the exact legacy read-per-decoy behavior.')

    misc = parser.add_argument_group('diagnostics')
    misc.add_argument('--dry-run', action='store_true',
                      help='Print the resolved plan and exit without initializing Rosetta.')
    misc.add_argument('--check-setters', action='store_true',
                      help='Verify GlycanDockProtocol setter names in this PyRosetta build, then exit.')
    misc.add_argument('-q', '--quiet', action='store_true',
                      help='Suppress per-decoy progress output.')

    return parser


def resolve_config(args):
    '''Build the GlycanDockConfig from --config/--preset plus flag overrides.'''
    if args.config:
        config = GlycanDockConfig.from_json(args.config)
    else:
        preset = args.preset or ('probe' if args.grid else 'refinement')
        config = getattr(GlycanDockConfig, preset)()

    changes = {}
    if args.n_cycles is not None:
        changes['n_cycles'] = args.n_cycles
    if args.refine_only:
        changes['refine_only'] = True
    if args.no_random_start:
        changes['random_start'] = False
    if args.prepack_mode is not None:
        changes['prepack_mode'] = args.prepack_mode
    if changes:
        config = config.replace(**changes)

    # --set is applied last so it always wins.
    if args.overrides:
        config = config.with_overrides(args.overrides)

    return config


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.check_setters:
        _require_pyrosetta()
        results = check_setters(verbose=True)
        return 0 if all(results.values()) else 1

    if not args.complex:
        parser.error('the complex PDB argument is required '
                     '(omit it only with --check-setters)')
    if args.nstruct < 1:
        parser.error('number of output structures must be at least 1')
    if args.start < 1:
        parser.error('output pose count must begin at least from 1')

    n_blocks = args.n_blocks if args.n_blocks is not None else \
        _env_int('SLURM_ARRAY_TASK_COUNT', 1)
    block_index = args.block_index if args.block_index is not None else \
        _env_int('SLURM_ARRAY_TASK_ID', 0)

    try:
        config = resolve_config(args)
    except (ValueError, OSError) as e:
        parser.error(str(e))

    outprefix = args.outprefix or default_outprefix(args.complex)
    run_id = args.run_id or outprefix
    options = args.options or (
        STAGE_2_OPTIONS if config.refine_only else STAGE_1_2_OPTIONS)

    try:
        indices = block_slice(args.nstruct, n_blocks, block_index, start=args.start)
    except ValueError as e:
        parser.error(str(e))

    seed = args.seed if args.seed is not None else block_seed(run_id, block_index)
    if seed is not None and seed < 0:
        seed = None

    if not indices:
        print(f'Block {block_index} of {n_blocks} has no decoys assigned '
              f'({args.nstruct} total). Nothing to do.')
        return 0

    if args.dry_run:
        print(f'run_id       : {run_id}')
        print(f'complex      : {args.complex}')
        print(f'grid         : {args.grid}')
        print(f'native       : {args.native}')
        print(f'options      : {options}')
        print(f'outdir       : {os.path.abspath(args.outdir)}')
        print(f'block        : {block_index} of {n_blocks}')
        print(f'decoys       : {len(indices)} of {args.nstruct} total '
              f'({indices[0]}..{indices[-1]})')
        print(f'seed         : {seed}')
        print(f'first output : {decoy_name(outprefix, indices[0])}')
        print(f'last output  : {decoy_name(outprefix, indices[-1])}')
        print('\nresolved config vs publication defaults:')
        diff = GlycanDockConfig.publication().diff(config)
        if diff:
            for field, (pub, mine) in sorted(diff.items()):
                print(f'  {field:18s} {pub!r:>8} -> {mine!r}')
        else:
            print('  (identical to publication defaults)')
        return 0

    _require_pyrosetta()
    manifest = run_block(
        config, args.complex,
        indices=indices,
        grid_pdb=args.grid,
        native=args.native,
        cst_file=args.cst_file,
        options_file=options,
        outprefix=outprefix,
        outdir=args.outdir,
        run_id=run_id,
        block_index=block_index,
        seed=seed,
        reuse_template=not args.no_reuse_template,
        verbose=not args.quiet,
    )

    print(f'\nWrote {len(manifest["outputs"])} decoys to '
          f'{os.path.abspath(args.outdir)}')
    return 0


if __name__ == '__main__':
    exit(main())
