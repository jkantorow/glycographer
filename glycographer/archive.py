'''
Reading and writing GlycanDock decoys stored in tar archives.

A docking run writes one decoy PDB per sample, which on a quota'd cluster
filesystem costs one inode per decoy. Instead, each SLURM array task packs its
block of decoys (plus its manifest and timings) into a single
<run_id>_block<NNN>.tar.xz, and merge_block_archives() can later fold a whole
run into one <run_id>.tar.xz.

xz rather than gzip: the receptor is near-identical in every decoy, but
DEFLATE only looks back 32 KB -- less than one decoy -- so gzip compresses each
decoy on its own. xz's 8 MB window spans dozens of decoys. Measured on 500
1k9i decoys (95 MB): tar.gz 22.3 MB, tar.xz 2.4 MB.

Pose paths
----------
Decoys inside an archive are addressed as '<archive>::<member>', e.g.
'runs/1k9i_block007.tar.xz::1k9i_0042.pdb'. Everywhere a pose path is accepted
(build_pose_list, GlycanDockEnsemble.from_poses, ...) a plain .pdb path and an
archive member path are interchangeable, and archive members are read in
memory -- nothing is extracted to disk.

Any tar archive works (.tar, .tar.gz/.tgz, .tar.xz/.txz), including the
tar.gz archives made by hand with `tar -I pigz -cf`, and archives that were
concatenated with `cat`.
'''

import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

MEMBER_SEP = '::'
ARCHIVE_EXTS = ('.tar.gz', '.tgz', '.tar.xz', '.txz', '.tar')
META_DIR = 'meta'

# Errors a truncated or corrupt archive can raise while being read.
_READ_ERRORS = (tarfile.TarError, EOFError, OSError, ValueError)


# ---- pose paths ------------------------------------------------------------

def is_archive(path) -> bool:
    '''True if `path` names a tar archive (by extension).'''
    return str(path).lower().endswith(ARCHIVE_EXTS)


def strip_archive_ext(name: str) -> str:
    ''''runs/1k9i.tar.xz' -> 'runs/1k9i'; other names are returned unchanged.'''
    for ext in ARCHIVE_EXTS:
        if name.lower().endswith(ext):
            return name[:-len(ext)]
    return name


def member_path(archive: str, member: str) -> str:
    return f'{archive}{MEMBER_SEP}{member}'


def split_pose_path(path: str) -> Tuple[Optional[str], str]:
    '''
    Split a pose path into (archive, member). A plain file path returns
    (None, path). Splits on the last '::' so Windows drive letters are safe.
    '''
    archive, sep, member = str(path).rpartition(MEMBER_SEP)
    if sep and is_archive(archive):
        return archive, member
    return None, str(path)


def pose_name(path: str) -> str:
    '''Basename of the decoy itself, whether a plain path or an archive member.'''
    return os.path.basename(split_pose_path(path)[1])


def abs_pose_path(path: str) -> str:
    '''Absolute form of a pose path (only the archive part is made absolute).'''
    archive, member = split_pose_path(path)
    if archive is None:
        return os.path.abspath(member)
    return member_path(os.path.abspath(archive), member)


# ---- reading ---------------------------------------------------------------

def _open_archive(archive: str):
    # ignore_zeros reads past the end-of-archive marker, so archives that were
    # joined with `cat` are read in full rather than only their first part.
    return tarfile.open(archive, 'r:*', ignore_zeros=True)


def list_archive_poses(archive: str) -> List[str]:
    '''Member paths ('<archive>::<member>') of every .pdb in `archive`.'''
    with _open_archive(archive) as tf:
        return [member_path(archive, m.name) for m in tf
                if m.isfile() and m.name.endswith('.pdb')]


def iter_pose_texts(pose_paths: Iterable[str]) -> Iterator[Tuple[str, str]]:
    '''
    Yield (pose_path, text) for each pose path, reading each archive in a
    single sequential pass.

    Plain files are yielded in the order given; archive members are yielded
    in the order they are stored, which need not match the order given (a tar
    made from a directory is in filesystem order). Callers that care about
    order should key on the returned pose_path.
    '''
    by_archive: Dict[str, set] = {}
    for p in pose_paths:
        archive, member = split_pose_path(p)
        if archive is None:
            with open(member) as f:
                yield p, f.read()
        else:
            by_archive.setdefault(archive, set()).add(member)

    for archive, wanted in by_archive.items():
        found = set()
        with _open_archive(archive) as tf:
            for m in tf:
                if m.name in wanted and m.name not in found:
                    found.add(m.name)
                    text = tf.extractfile(m).read().decode()
                    yield member_path(archive, m.name), text
        missing = wanted - found
        if missing:
            raise FileNotFoundError(
                f'{len(missing)} pose(s) not found in {archive}, e.g. '
                f'{sorted(missing)[0]}')


def read_pose_text(pose_path: str) -> str:
    '''Text of a single pose. For many poses use iter_pose_texts instead.'''
    for _, text in iter_pose_texts([pose_path]):
        return text


# ---- block archives (written by sample.run_block) ---------------------------

def block_archive_name(run_id: str, block_index: int) -> str:
    return f'{run_id}_block{block_index:03d}.tar.xz'


def add_bytes(tf: tarfile.TarFile, arcname: str, data: bytes) -> None:
    '''Add an in-memory file to an open tar archive.'''
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = time.time()
    tf.addfile(info, io.BytesIO(data))


def read_block_manifest(archive: str) -> Optional[dict]:
    '''
    The block manifest packed into a block archive, or None if the archive is
    unreadable or has none (e.g. a block killed mid-write). The manifest is
    the last member written, so a readable one means the block finished.
    '''
    try:
        with _open_archive(archive) as tf:
            for m in tf:
                if m.isfile() and m.name.endswith('_manifest.json'):
                    return json.load(tf.extractfile(m))
    except _READ_ERRORS:
        return None
    return None


def find_block_archives(outdir: str, run_id: Optional[str] = None) -> Dict[str, List[str]]:
    '''
    {run_id: [block archive paths, sorted]} for the block archives in
    `outdir`, optionally only those of one run_id.
    '''
    runs: Dict[str, List[str]] = {}
    for name in sorted(os.listdir(outdir)):
        if not name.endswith('.tar.xz'):
            continue
        stem, sep, block = name[:-len('.tar.xz')].rpartition('_block')
        if not sep or not block.isdigit():
            continue
        if run_id is None or stem == run_id:
            runs.setdefault(stem, []).append(os.path.join(outdir, name))
    return runs


# ---- merging ---------------------------------------------------------------

def _open_xz_writer(path: str, threads: int, preset: int):
    '''
    Open `path` for streamed tar.xz writing. Uses the `xz` binary when it is
    on PATH, since it can compress on several cores; Python's lzma is
    single-threaded. Returns (tarfile, closer).
    '''
    xz = shutil.which('xz')
    if xz is None or threads <= 1:
        tf = tarfile.open(path, 'w:xz', preset=preset)
        return tf, tf.close

    out = open(path, 'wb')
    proc = subprocess.Popen([xz, f'-T{threads}', f'-{preset}', '-c'],
                            stdin=subprocess.PIPE, stdout=out)
    tf = tarfile.open(fileobj=proc.stdin, mode='w|')

    def close():
        tf.close()
        proc.stdin.close()
        rc = proc.wait()
        out.close()
        if rc != 0:
            raise RuntimeError(f'xz exited with status {rc} writing {path}')

    return tf, close


def merge_block_archives(block_archives: Sequence[str], out_path: str, *,
                         extra_files: Sequence[str] = (),
                         threads: int = 1, preset: int = 6,
                         delete_inputs: bool = False,
                         verbose: bool = True) -> dict:
    '''
    Merge the block archives of one run into a single tar.xz.

    Every block must be complete (have a manifest), blocks 0..n_blocks-1 must
    all be present, and no decoy may appear twice; otherwise nothing is
    written. The merged archive holds the decoys at its top level and every
    block's manifest and timings under meta/, alongside any `extra_files`
    (e.g. SLURM logs) under meta/logs/.

    The output is written to <out_path>.partial and re-read to check that
    every decoy made it before being renamed into place. Only then, and only
    with delete_inputs, are the block archives and extra files removed.

    Returns
    -------
    dict : summary with run_id, n_blocks, n_decoys, out_path, deleted.
    '''
    if not block_archives:
        raise ValueError('no block archives to merge')

    # Pass 1: validate every block before writing anything.
    manifests = []
    for path in block_archives:
        m = read_block_manifest(path)
        if m is None:
            raise ValueError(f'{path} has no readable manifest -- the block did '
                             f'not finish. Rerun it with --resume first.')
        manifests.append((m, path))
    manifests.sort(key=lambda mp: mp[0]['block_index'])

    run_ids = {m['run_id'] for m, _ in manifests}
    if len(run_ids) > 1:
        raise ValueError(f'block archives come from several runs: {sorted(run_ids)}')
    run_id = run_ids.pop()

    block_ids = [m['block_index'] for m, _ in manifests]
    if len(set(block_ids)) != len(block_ids):
        raise ValueError(f'duplicate block indices among {block_archives}')
    n_blocks = {m.get('n_blocks') for m, _ in manifests}
    if len(n_blocks) > 1:
        raise ValueError(f'blocks disagree on n_blocks: {sorted(n_blocks, key=str)}')
    n_blocks = n_blocks.pop()
    if n_blocks is None:
        if verbose:
            print('Warning: manifests do not record n_blocks, so a missing '
                  'final block cannot be detected.')
        expected = range(max(block_ids) + 1)
    else:
        expected = range(n_blocks)
    missing = sorted(set(expected) - set(block_ids))
    if missing:
        raise ValueError(f'{run_id}: missing block(s) {missing}')

    decoys = [d for m, _ in manifests for d in m['outputs']]
    if len(set(decoys)) != len(decoys):
        raise ValueError(f'{run_id}: some decoys appear in more than one block')

    # Pass 2: stream every member into the merged archive.
    partial = out_path + '.partial'
    t0 = time.perf_counter()
    try:
        tf, close = _open_xz_writer(partial, threads, preset)
        try:
            meta = []
            for m, path in manifests:
                with _open_archive(path) as src:
                    for member in src:
                        if not member.isfile():
                            continue
                        data = src.extractfile(member).read()
                        if member.name.startswith(META_DIR + '/'):
                            meta.append((member.name, data))
                        else:
                            add_bytes(tf, member.name, data)
                if verbose:
                    print(f'  block {m["block_index"]:3d}: {len(m["outputs"])} decoys')
            for name, data in meta:
                add_bytes(tf, name, data)
            for path in extra_files:
                with open(path, 'rb') as f:
                    add_bytes(tf, f'{META_DIR}/logs/{os.path.basename(path)}', f.read())
        finally:
            close()

        # Re-read before replacing anything.
        with _open_archive(partial) as tf:
            written = {os.path.basename(m.name) for m in tf
                       if m.isfile() and m.name.endswith('.pdb')}
        if written != set(decoys):
            raise RuntimeError(f'{partial} holds {len(written)} decoys but '
                               f'{len(decoys)} were expected; inputs left in place.')
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    os.replace(partial, out_path)

    deleted = []
    if delete_inputs:
        for path in [p for _, p in manifests] + list(extra_files):
            os.remove(path)
            deleted.append(path)

    if verbose:
        size = os.path.getsize(out_path) / 2**20
        print(f'Merged {len(decoys)} decoys from {len(manifests)} blocks into '
              f'{out_path} ({size:.1f} MB, {time.perf_counter() - t0:.0f}s)')

    return {'run_id': run_id, 'n_blocks': len(manifests),
            'n_decoys': len(decoys), 'out_path': os.path.abspath(out_path),
            'deleted': deleted}
