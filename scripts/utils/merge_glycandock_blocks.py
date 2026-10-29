#!/usr/bin/env python

'''
Merge the per-block archives of a finished GlycanDock run into one tar.xz.

run_glycandock.py writes one <run_id>_block<NNN>.tar.xz per SLURM array task.
This folds them into a single <run_id>.tar.xz -- decoys at the top level, every
block's manifest and timings under meta/, and (with --logs) the SLURM logs
under meta/logs/ -- then deletes the block archives and logs, so a finished
run costs one inode.

Merging is optional: process_glycandock_output.py and map_ensembles.py read a
directory of block archives just as well as the merged archive.

Nothing is deleted unless every block 0..n_blocks-1 is present and finished,
and the merged archive has been re-read and holds every decoy. Simply joining
the block archives with `cat` is NOT equivalent: plain `tar -xJf` stops at the
end of the first block and silently drops the rest.

Run it after the docking array, e.g. as a dependent SLURM job:

    jid=$(sbatch --parsable --array=0-41 scripts/slurm/run_glycandock.sh \\
        complex.pdb -n 1000 --grid grid.pdb --outdir runs/1k9i)
    sbatch --dependency=afterok:$jid scripts/slurm/merge_glycandock_blocks.sh \\
        runs/1k9i --logs "gdock_${jid}_*.out"

If any array task fails, afterok never fires: rerun the failed blocks with
--resume, then submit the merge by hand.
'''

import os
import sys
import glob
import argparse

# The glycographer package may not be pip-installed on the cluster, so put the
# repo root (three directories up from <repo>/scripts/utils/) on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from glycographer.archive import find_block_archives, merge_block_archives


def main():
    parser = argparse.ArgumentParser(
        description='Merge the per-block archives of a GlycanDock run into one tar.xz.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('\n\n', 1)[1])
    parser.add_argument('outdir',
                        help='Directory holding the <run_id>_block<NNN>.tar.xz archives '
                             '(the --outdir of run_glycandock.py)')
    parser.add_argument('--run-id', default=None,
                        help='Run to merge, if outdir holds block archives from more than one')
    parser.add_argument('-o', '--output', default=None,
                        help='Merged archive path (default: <outdir>/<run_id>.tar.xz)')
    parser.add_argument('--logs', nargs='+', default=[], metavar='GLOB',
                        help='SLURM log files to pack under meta/logs/ (and delete), e.g. '
                             '"gdock_12345_*.out". Quote globs so they expand here.')
    parser.add_argument('--keep-inputs', action='store_true',
                        help='Keep the block archives and logs after merging')
    parser.add_argument('--threads', type=int,
                        default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)),
                        help='xz compression threads (default: $SLURM_CPUS_PER_TASK, else 1). '
                             'Uses the xz binary when > 1, otherwise Python lzma.')
    parser.add_argument('--preset', type=int, default=6, choices=range(10),
                        metavar='0-9', help='xz compression level (default: 6)')
    args = parser.parse_args()

    runs = find_block_archives(args.outdir, args.run_id)
    if not runs:
        which = f' for run {args.run_id!r}' if args.run_id else ''
        raise SystemExit(f'No block archives{which} in {args.outdir}.')
    if len(runs) > 1:
        raise SystemExit(f'{args.outdir} holds block archives from several runs '
                         f'({", ".join(sorted(runs))}); pick one with --run-id.')
    run_id, blocks = runs.popitem()

    # Never pack (and delete) this merge job's own log, which a broad glob
    # like "*.out" would otherwise match while it is still being written.
    own_job = os.environ.get('SLURM_JOB_ID')
    logs = sorted({p for pattern in args.logs for p in glob.glob(pattern)
                   if not (own_job and own_job in os.path.basename(p))})
    if args.logs and not logs:
        print(f'Warning: no log files matched {args.logs}', file=sys.stderr)

    out_path = args.output or os.path.join(args.outdir, f'{run_id}.tar.xz')
    if os.path.exists(out_path):
        raise SystemExit(f'{out_path} already exists; remove it or pass -o.')

    print(f'[{run_id}] merging {len(blocks)} block archives'
          + (f' and {len(logs)} logs' if logs else ''))
    try:
        result = merge_block_archives(
            blocks, out_path, extra_files=logs, threads=args.threads,
            preset=args.preset, delete_inputs=not args.keep_inputs)
    except (ValueError, RuntimeError) as e:
        raise SystemExit(f'Error: {e}')

    if result['deleted']:
        print(f'[{run_id}] removed {len(result["deleted"])} input files')
    return 0


if __name__ == '__main__':
    exit(main())
