'''
Dataclasses and tools for glycan conformation sampling during docking.

This module holds the PyRosetta-facing half of the glycographer pipeline: the
GlycanDock protocol parameters (`GlycanDockConfig`), Rosetta session setup, and
the decoy-generation loop that `scripts/run_glycandock.py` drives.

PyRosetta is imported lazily inside the functions that need it, so the module
stays importable on machines without a PyRosetta build. Config objects can be
constructed, serialized, compared and inspected anywhere -- which is what makes
a run's provenance readable from the analysis side (Windows, notebooks) rather
than only on the HPC partition.

Parallelism
-----------
A decoy is an independent sample, so the unit of parallel work is a *block* of
decoy indices run by one process, i.e. one SLURM array task. Nothing is shared
between blocks except the read-only input PDBs, and decoy indices are assigned
up front so blocks can never collide on an output filename. (The SLURM array
driver that calls run_block() once per task is not written yet.)

Staging
-------
`GlycanDockConfig.prepack_mode` selects how Stage 0 pre-packing is handled:

    'per_decoy'  legacy glycographer behavior -- prepack inside the decoy loop,
                 in the same Rosetta session as Stages 1/2. This is what every
                 existing glycographer ensemble was generated with, so it is
                 the default; results stay comparable.
    'once'       the input PDB is expected to be ALREADY prepacked by a
                 separate `prepack()` job, run as its own process.
                 Stage 0 is skipped here. This matches the published GlycanDock
                 workflow, where prepacking runs under different Rosetta flags
                 (-ex3 -ex4 -ex1aro -ex2aro) than refinement (-ex1 -ex2) and
                 therefore cannot share a process.
    'none'       no pre-packing at all.
'''

from dataclasses import dataclass, asdict, replace, fields
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence
import json
import os
import time
import zlib

# --- option file locations -------------------------------------------------
# config/ currently lives at the repo root, beside the package. Under the
# editable install this resolves to the live source tree; see the note in
# pyproject.toml about eventually moving it inside the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(_REPO_ROOT, 'config')

STAGE_0_OPTIONS = os.path.join(CONFIG_DIR, 'glycandock_stage_0.init')
STAGE_1_2_OPTIONS = os.path.join(CONFIG_DIR, 'glycandock_stage_1_stage_2.init')
STAGE_2_OPTIONS = os.path.join(CONFIG_DIR, 'glycandock_stage_2.init')


@dataclass(frozen=True)
class GlycanDockConfig:
    '''
    Protocol parameters for a GlycanDock run.

    Field defaults are the GlycanDock publication defaults (Nance et al. 2021);
    descriptions are taken from that paper's supplemental information and the
    PyRosetta glycan_docking API docs. Use the `probe()` / `refinement()`
    presets rather than the bare constructor unless you specifically want the
    publication baseline -- see the class docstring table below.

    This object holds *protocol* parameters only. Input paths (complex, grid,
    native, constraint file) are arguments to the run functions instead, so a
    config stays portable between machines and serializes cleanly for
    provenance.
    '''

    # Full protocol parameters:
    partners: str = 'A_X' # Set the chain IDs identifying the protein-glycoligand complex protein is chain A, glycoligand is chain X –> partners = A_X
    refine_only: bool = False # Skip the Stage 1 initial perturbation and employ only the uniform ± 15° glycosidic torsion angle perturbations using the SmallBBSampler during Stage 2 docking and refinement
    prepack_only: bool = False # Perform only Stage 0 pre-packing
    intf_pack_dist: float = 16.0 # (Angstroms) Cutoff distance to define protein-glycan interface residues to be considered during packing
    slide: bool = True # Allow the glycoligand’s center-of-mass to be periodically translated toward the protein receptor’s center-of-mass during rigid-body sampling
    rand_glyc_jump: bool = False # Set whether to use a random, non-branch-point residue of the glycoligand to set as the Jump residue for the docking FoldTree

    # Stage 1-specific parameters:
    stage1_rand_rot: bool = False # During Stage 1 initial perturbation, randomly rotate the glycoligand about its center-of-mass in 360˚ space
    stage1_trans_mag: float = 0.5 # (Angstroms) The translational magnitude of the Gaussian perturbation on the glycoligand’s center-of-mass performed during Stage 1
    stage1_rot_mag: float = 7.5 # (Degrees) The rotational magnitude of the Gaussian perturbation on the glycoligand’s center-of-mass performed during Stage 1
    stage1_tor_mag: float = 12.5 # (Degrees) The magnitude of the uniform perturbation on each glycosidic torsion angle

    # Stage 2-specific parameters:
    n_repeats: int = 3 # Number of times to repeat Stage 2 of GlycanDock if the final model does not have a negative interface score
    n_cycles: int = 10 # Number of stage 2 outer cycles to perform
    rb_rounds: int = 8 # The number of inner cycles of rigid-body perturbations to perform
    tor_rounds: int = 8 # The number of inner cycles of glycosidic torsion perturbations to perform
    stage2_trans_mag: float = 0.5 # (Angstroms) During Stage 2 docking and refinement, this is the magnitude for performing a translational Gaussian perturbation on the glycoligand's center-of-mass
    stage2_rot_mag: float = 7.5 # (Degrees) During Stage 2 docking and refinement, this is the magnitude for performing a rotational Gaussian perturbation on the glycoligand's center-of-mass
    full_pack_freq: int = 8 # When packing after each sampling perturbation in Stage 2, at what frequency should the PackRotamersMover be called? Set to 0 to use only the EnergyCutRotamerTrialsMover after every perturbation. Set to the value of n_rigid_body_rounds and n_torsion_rounds to use the PackRotamersMover only after the last perturbation
    mc_kt: float = 0.6 # (a.u.) During Stage 2 docking and refinement, the value of kT used to accept or reject moves based on the Metropolis criterion
    ramp_sf: bool = True # Set whether to ramp the fa_atr and fa_rep score terms of the ScoreFunction used during Stage 2 sampling and optimization

    # Glycographer staging parameters (not GlycanDock mover settings):
    prepack_mode: str = 'per_decoy' # How Stage 0 is handled: 'per_decoy' (legacy), 'once' (input is pre-prepacked), or 'none'. See module docstring.
    random_start: bool = True # Randomize glycoligand orientation with RigidBodyRandomizeMover before docking

    _PREPACK_MODES = ('per_decoy', 'once', 'none')

    def __post_init__(self):
        if self.prepack_mode not in self._PREPACK_MODES:
            raise ValueError(
                f'prepack_mode must be one of {self._PREPACK_MODES}, '
                f'got {self.prepack_mode!r}')
        if self.n_cycles < 1:
            raise ValueError(f'n_cycles must be >= 1, got {self.n_cycles}')
        if self.n_repeats < 1:
            raise ValueError(f'n_repeats must be >= 1, got {self.n_repeats}')

    # ---- presets ---------------------------------------------------------
    #
    # `probe` and `refinement` reproduce scripts/run_glycandock.py as it stood
    # before this module existed, so ensembles generated through sample.py stay
    # comparable with everything already in the archive. Where they differ from
    # the publication defaults:
    #
    #   field              publication   probe    refinement
    #   n_cycles                    10       1            10
    #   stage1_rand_rot          False    True          True
    #   stage1_rot_mag             7.5   180.0           7.5
    #   stage2_trans_mag           0.5     0.2           0.5
    #   stage2_rot_mag             7.5    45.0           7.5
    #   rb_rounds                    8      20             8
    #   tor_rounds                   8      20            20
    #
    # Note `tor_rounds`: the old script set n_torsion_rounds = 20
    # unconditionally, so past *refinement* runs also used 20 rather than the
    # publication's 8. That is preserved here deliberately -- changing it is a
    # protocol decision, not a side effect of the refactor.

    @classmethod
    def probe(cls, **overrides) -> 'GlycanDockConfig':
        '''
        Glycographer grid-probe sampling preset.

        Probes are small (one or a few residues) compared with a full glycan
        chain, so Stage 1 places them with a full 360 degree randomization,
        Stage 2 runs a single ramping cycle, and sampling is weighted toward
        rigid-body moves with tighter translational steps.
        '''
        base = dict(
            n_cycles=1,
            stage1_rand_rot=True,
            stage1_rot_mag=180.0,
            stage2_trans_mag=0.2,
            stage2_rot_mag=45.0,
            rb_rounds=20,
            tor_rounds=20,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def refinement(cls, **overrides) -> 'GlycanDockConfig':
        '''
        Preset for refining a known or designed protein-glycan complex, i.e. a
        standard GlycanDock run rather than grid probing.

        Pass `refine_only=True` to bypass Stage 1 entirely; the stage1_* fields
        are then unused.
        '''
        base = dict(
            n_cycles=10,
            stage1_rand_rot=True,
            stage1_rot_mag=7.5,
            stage2_trans_mag=0.5,
            stage2_rot_mag=7.5,
            rb_rounds=8,
            tor_rounds=20,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def publication(cls, **overrides) -> 'GlycanDockConfig':
        '''GlycanDock defaults exactly as published (Nance et al. 2021).'''
        return cls(**overrides)

    # ---- serialization ---------------------------------------------------
    def as_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, outname: str) -> str:
        with open(outname, 'w') as f:
            json.dump(self.as_dict(), f, indent=2, sort_keys=True)
        return outname

    @classmethod
    def from_json(cls, path: str) -> 'GlycanDockConfig':
        with open(path) as f:
            return cls(**json.load(f))

    def replace(self, **changes) -> 'GlycanDockConfig':
        '''Return a copy with `changes` applied (the config is frozen).'''
        return replace(self, **changes)

    def diff(self, other: 'GlycanDockConfig') -> Dict:
        '''
        Fields that differ between two configs, as {field: (self, other)}.
        Useful for stating in a methods section exactly how two ensembles were
        sampled differently.
        '''
        return {f.name: (getattr(self, f.name), getattr(other, f.name))
                for f in fields(self)
                if getattr(self, f.name) != getattr(other, f.name)}


# --- config field -> GlycanDockProtocol setter ------------------------------
# Kept as an explicit table rather than getattr-by-convention so that a rename
# on either side fails loudly and in one place.
_SETTERS = {
    'partners': 'set_partners',
    'refine_only': 'set_refine_only',
    'intf_pack_dist': 'set_interface_packing_distance',
    'slide': 'set_slide_glycan_into_contact',
    'rand_glyc_jump': 'set_rand_glycan_jump_res',
    'stage1_rand_rot': 'set_stage1_rotate_glycan_about_com',
    'stage1_trans_mag': 'set_stage1_perturb_glycan_com_trans_mag',
    'stage1_rot_mag': 'set_stage1_perturb_glycan_com_rot_mag',
    'stage1_tor_mag': 'set_stage1_torsion_uniform_pert_mag',
    'n_repeats': 'set_n_repeats',
    'rb_rounds': 'set_n_rigid_body_rounds',
    'tor_rounds': 'set_n_torsion_rounds',
    'stage2_trans_mag': 'set_stage2_trans_mag',
    'stage2_rot_mag': 'set_stage2_rot_mag',
    'full_pack_freq': 'set_full_packing_frequency',
    'mc_kt': 'set_mc_kt',
    'ramp_sf': 'set_ramp_scorefxn',
}

# Fields with no corresponding mover setter:
#   n_cycles      GlycanDockProtocol exposes no set_n_cycles; the publication's
#                 own CLI examples pass -n_cycles as a Rosetta flag, and the
#                 mover reads it from the option system when constructed. It is
#                 therefore applied at init_rosetta() time, which means
#                 n_cycles is FIXED FOR THE LIFETIME OF A PROCESS and a block
#                 cannot mix cycle counts. Run probe and refinement separately.
#   prepack_only  set explicitly by _build_protocol, not driven from config.
#   prepack_mode, random_start
#                 glycographer staging, not mover settings.
_NO_SETTER_FIELDS = ('n_cycles', 'prepack_only', 'prepack_mode', 'random_start')

# Every config field must be accounted for: either it maps to a mover setter or
# it is explicitly listed as handled elsewhere. Checked at import (pure Python,
# no PyRosetta needed) so that adding a field to GlycanDockConfig without
# wiring it up fails immediately, rather than silently never reaching Rosetta.
_unmapped = ({f.name for f in fields(GlycanDockConfig)}
             - set(_SETTERS) - set(_NO_SETTER_FIELDS))
if _unmapped:
    raise RuntimeError(
        f'GlycanDockConfig field(s) {sorted(_unmapped)} are neither in '
        f'_SETTERS nor _NO_SETTER_FIELDS, so they would never be applied to '
        f'the GlycanDockProtocol mover. Add them to one or the other.')
del _unmapped


def check_setters(verbose: bool = True) -> Dict[str, bool]:
    '''
    Verify that every setter name in `_SETTERS` actually exists on
    GlycanDockProtocol in the installed PyRosetta build.

    Run this once on the HPC partition after any PyRosetta upgrade. It is much
    cheaper than discovering a renamed setter partway through an array job, and
    it is the intended way to resolve the two names that could not be verified
    off-cluster (`set_rand_glycan_jump_res` in particular).

    Returns
    -------
    dict : {config_field: setter_exists}
    '''
    from pyrosetta.rosetta.protocols.glycan_docking import GlycanDockProtocol

    available = set(dir(GlycanDockProtocol))
    result = {field: (setter in available)
              for field, setter in _SETTERS.items()}

    if verbose:
        missing = {f: _SETTERS[f] for f, ok in result.items() if not ok}
        if missing:
            print('MISSING setters on GlycanDockProtocol:')
            for field, setter in missing.items():
                print(f'  {field:20s} -> {setter}')
            print('\nCandidate setter names in this build:')
            for name in sorted(n for n in available if n.startswith('set_')):
                print(f'  {name}')
        else:
            print(f'All {len(result)} setters resolve on GlycanDockProtocol.')

    return result


def _build_protocol(config: GlycanDockConfig, prepack_only: bool = False):
    '''
    Construct a GlycanDockProtocol mover from `config`.

    With prepack_only=True only the three settings that Stage 0 actually
    consults are applied (partners, prepack_only, slide) -- matching both the
    legacy script and the publication's prepack invocation. Sampling parameters
    are meaningless for a pre-pack and are deliberately left at their defaults.
    '''
    from pyrosetta.rosetta.protocols.glycan_docking import GlycanDockProtocol

    mover = GlycanDockProtocol()

    if prepack_only:
        mover.set_partners(config.partners)
        mover.set_prepack_only(True)
        mover.set_slide_glycan_into_contact(config.slide)
        return mover

    mover.set_prepack_only(False)
    for field, setter_name in _SETTERS.items():
        setter = getattr(mover, setter_name, None)
        if setter is None:
            raise AttributeError(
                f'GlycanDockProtocol has no {setter_name!r} (for config field '
                f'{field!r}) in this PyRosetta build. Run '
                f'glycographer.sample.check_setters() to list the setters this '
                f'build does expose, then update sample._SETTERS.')
        setter(getattr(config, field))

    return mover


# --- timing -----------------------------------------------------------------
@contextmanager
def _timed(record: Dict[str, float], key: str):
    '''
    Accumulate wall time for a named step into `record`.

    perf_counter is the right clock here: each mover's .apply() is a blocking
    C++ call on this thread, so on a dedicated core wall time is CPU time. The
    overhead is nanoseconds against minute-scale movers, so timing stays on
    permanently -- profiling data comes from real production runs rather than a
    separate benchmark.
    '''
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record[key] = record.get(key, 0.0) + (time.perf_counter() - t0)


# --- Rosetta session --------------------------------------------------------
def block_seed(run_id: str, block_index: int) -> int:
    '''
    Deterministic RNG seed for one block of decoys.

    Seeding is per *block*, not per decoy: a block shares one Rosetta session,
    so all its decoys draw from a single RNG stream that advances naturally
    across the loop. Re-seeding per decoy would require re-initializing Rosetta,
    which is not supported.

    crc32 of "<run_id>:<block_index>" spreads adjacent block indices across the
    seed space, so array tasks launched in the same instant cannot collide the
    way time-derived seeds can, and a given (run_id, block) is reproducible.
    '''
    key = f'{run_id}:{block_index}'.encode()
    return (zlib.crc32(key) & 0x7fffffff) or 1


def _seed_flags(seed: Optional[int]) -> str:
    '''
    Rosetta flags pinning the RNG seed.

    `-run:jran` only takes effect together with `-run:constant_seed`; without
    it Rosetta derives a seed from the system clock / urandom and ignores jran.

    VERIFY ONCE ON THE CLUSTER: run the same block twice with the same seed and
    confirm the decoys are identical, then run two blocks with different seeds
    and confirm they differ. Silent seed collisions across array tasks would
    duplicate decoys without raising anything, and would not be obvious in the
    output -- an energy map built from duplicated samples just looks sharper
    than it should.

    Pass seed=None to fall back to Rosetta's default non-deterministic seeding.
    '''
    if seed is None:
        return ''
    return f'-run:constant_seed -run:jran {int(seed)}'


def init_rosetta(options_file: str, complex_pdb: Optional[str] = None, *,
                 native: Optional[str] = None,
                 n_cycles: Optional[int] = None,
                 cst_file: Optional[str] = None,
                 seed: Optional[int] = None,
                 extra_flags: str = '',
                 mute: bool = False) -> str:
    '''
    Start a Rosetta session for one stage of the protocol.

    A process gets exactly one of these. Because prepacking and refinement use
    different rotamer-sampling flags (-ex3 -ex4 -ex1aro -ex2aro vs -ex1 -ex2),
    they cannot share a session -- which is why `prepack_mode='once'` requires a
    separate job rather than a separate function call.

    Parameters
    ----------
    options_file : str
        Rosetta flags file -- one of STAGE_0_OPTIONS, STAGE_1_2_OPTIONS,
        STAGE_2_OPTIONS, or a caller-supplied path.
    complex_pdb : str, optional
        Passed as -in:file:s. Redundant with an explicit pose_from_pdb() call,
        but some option-system defaults key off it, and it records the input in
        the Rosetta log.
    native : str, optional
        Reference complex for RMSD/Fnat scoring (-in:file:native). Without it
        Fnat, heavy_Lrmsd, ring_Lrmsd and the other native-relative scores come
        out as NA.
    n_cycles : int, optional
        Stage 2 outer ramping cycles. Applied here rather than on the mover
        because GlycanDockProtocol exposes no setter for it; see _SETTERS.
    cst_file : str, optional
        Constraint file (-cst_fa_file), e.g. AtomPair restraints holding a
        known glycan in its site during refinement. Not used for probe
        sampling, where restraining the probe would defeat the scan.
    seed : int, optional
        RNG seed; see _seed_flags. Use block_seed() to derive one.

    Returns
    -------
    str : the full flag string passed to pyrosetta.init(), for provenance.
    '''
    from pyrosetta import init

    flags = []
    if complex_pdb:
        flags.append(f'-in:file:s {complex_pdb}')
    if native:
        flags.append(f'-in:file:native {native}')
    if n_cycles is not None:
        flags.append(f'-n_cycles {int(n_cycles)}')
    if cst_file:
        flags.append(f'-cst_fa_file {cst_file}')
    # -nstruct does NOT control the decoy count here -- the loop below does.
    # It is supplied because Rosetta's job machinery expects it to be set.
    flags.append('-nstruct 1')
    seed_str = _seed_flags(seed)
    if seed_str:
        flags.append(seed_str)
    if mute:
        flags.append('-mute all')
    if extra_flags:
        flags.append(extra_flags)
    flags.append(f'@{options_file}')

    flag_string = ' '.join(flags)
    init(flag_string)
    return flag_string


# --- Stage 0 ----------------------------------------------------------------
def prepack(complex_pdb: str, out_pdb: str, *,
            config: Optional[GlycanDockConfig] = None,
            options_file: str = STAGE_0_OPTIONS,
            native: Optional[str] = None,
            seed: Optional[int] = None,
            initialize: bool = True) -> str:
    '''
    Run GlycanDock Stage 0 pre-packing once and write the prepacked complex.

    Stage 0 separates the docking partners, repacks each in its unbound state,
    and re-forms the complex, so the result is essentially independent of where
    the glycoligand started. That is what makes it safe to run once per complex
    rather than once per decoy -- and why the legacy per-decoy call repeats the
    same work for every sample.

    This is intended to run as its own job, under STAGE_0_OPTIONS, whose finer
    rotamer sampling (-ex3 -ex4 -ex1aro -ex2aro) is affordable once and not
    thousands of times. Feed `out_pdb` to run_block() with
    prepack_mode='once'.

    Parameters
    ----------
    initialize : bool
        Call init_rosetta() first. Pass False only if this process already
        initialized Rosetta with Stage 0 flags.

    Returns
    -------
    str : path to the written prepacked PDB.
    '''
    from pyrosetta import pose_from_pdb

    if initialize:
        init_rosetta(options_file, complex_pdb, native=native, seed=seed)

    config = config or GlycanDockConfig()
    timings: Dict[str, float] = {}

    with _timed(timings, 'pose_from_pdb'):
        pose = pose_from_pdb(complex_pdb)
    with _timed(timings, 'prepack'):
        _build_protocol(config, prepack_only=True).apply(pose)

    out_dir = os.path.dirname(os.path.abspath(out_pdb))
    os.makedirs(out_dir, exist_ok=True)
    pose.dump_pdb(out_pdb)

    print(f'Prepacked {complex_pdb} -> {out_pdb} '
          f'({timings["prepack"]:.1f}s pack, '
          f'{timings["pose_from_pdb"]:.1f}s read)')
    return out_pdb


# --- Stages 1/2 -------------------------------------------------------------
class GlycanDockRunner:
    '''
    Holds the per-process state for generating decoys: the template pose, the
    movers built from a config, and the grid StartFrom mover.

    Everything expensive is built once here and reused across the block. The
    only things rebuilt per decoy are the pose copy and RigidBodyRandomizeMover
    (which needs the pose at construction).

    Rosetta must already be initialized -- call init_rosetta() first, or use the
    module-level run_block(), which does both.
    '''

    def __init__(self, config: GlycanDockConfig, complex_pdb: str, *,
                 grid_pdb: Optional[str] = None,
                 reuse_template: bool = True):
        from pyrosetta import pose_from_pdb
        from pyrosetta.rosetta.protocols.ligand_docking import StartFrom

        self.config = config
        self.complex_pdb = complex_pdb
        self.grid_pdb = grid_pdb
        self.reuse_template = reuse_template

        self._lig_chain = config.partners.split('_')[-1]

        # Parsing a glycan PDB under -include_sugars
        # -auto_detect_glycan_connections is not cheap, so read it once and
        # deep-copy per decoy. NOTE: the legacy script re-read the file every
        # iteration. Copying yields an identical starting pose, but if PDB
        # reading consumes RNG draws (rebuilding missing density, idealizing
        # geometry) the downstream RNG stream will differ from a legacy run --
        # so legacy comparison is statistical, not bitwise. Set
        # reuse_template=False to restore the old read-per-decoy behavior.
        self._template = pose_from_pdb(complex_pdb) if reuse_template else None

        self.start_from = None
        if grid_pdb:
            self.start_from = StartFrom()
            self.start_from.chain(self._lig_chain)
            self.start_from.parse_pdb_file(grid_pdb)

        self.gdock_full = _build_protocol(config, prepack_only=False)
        self.gdock_prepack = (
            _build_protocol(config, prepack_only=True)
            if config.prepack_mode == 'per_decoy' else None)

    def _fresh_pose(self, timings: Dict[str, float]):
        from pyrosetta import Pose, pose_from_pdb

        with _timed(timings, 'pose_load'):
            if self._template is not None:
                return Pose(self._template)
            return pose_from_pdb(self.complex_pdb)

    def run_decoy(self, index: int, outname: str) -> Dict[str, float]:
        '''
        Generate one decoy and write it to `outname`.

        Returns the step timing record for this decoy. The step order matches
        the legacy script exactly: read -> grid placement -> randomize ->
        prepack -> GlycanDock.
        '''
        from pyrosetta import Vector1
        from pyrosetta.rosetta.protocols.rigid import (
            RigidBodyRandomizeMover, partner_downstream)
        from pyrosetta.rosetta.protocols.docking import setup_foldtree

        cfg = self.config
        timings: Dict[str, float] = {}

        pose = self._fresh_pose(timings)

        if self.start_from is not None:
            # StartFrom picks one of the grid's pseudo-atom coordinates and
            # moves the glycoligand there, so decoys are distributed across
            # sampling points rather than fixed to one.
            with _timed(timings, 'start_from'):
                self.start_from.apply(pose)

        if cfg.random_start:
            with _timed(timings, 'randomize'):
                # Jump 1 is the glycoligand jump. RigidBodyRandomizeMover
                # rotates the downstream partner about its own center of mass,
                # so a StartFrom placement survives the randomization.
                setup_foldtree(pose, cfg.partners, Vector1([1]))
                RigidBodyRandomizeMover(pose, 1, partner_downstream).apply(pose)

        if self.gdock_prepack is not None:
            with _timed(timings, 'prepack'):
                self.gdock_prepack.apply(pose)

        with _timed(timings, 'glycandock'):
            self.gdock_full.apply(pose)

        with _timed(timings, 'dump'):
            pose.dump_pdb(outname)

        timings['total'] = sum(v for k, v in timings.items() if k != 'total')
        timings['index'] = index
        return timings


def decoy_name(outprefix: str, index: int) -> str:
    '''
    Output filename for one decoy.

    Zero-padded to 4 digits, matching the legacy naming that
    utils.build_pose_list / POSE_NUM_RE already parse. Indices above 9999 widen
    gracefully and still parse.
    '''
    return f'{outprefix}_{index:04d}.pdb'


def default_outprefix(complex_pdb: str) -> str:
    return os.path.basename(complex_pdb).replace('.pdb', '')


def run_block(config: GlycanDockConfig, complex_pdb: str, *,
              start: int = 1, count: int = 1,
              indices: Optional[Sequence[int]] = None,
              grid_pdb: Optional[str] = None,
              native: Optional[str] = None,
              cst_file: Optional[str] = None,
              options_file: Optional[str] = None,
              outprefix: Optional[str] = None,
              outdir: str = '.',
              run_id: Optional[str] = None,
              block_index: int = 0,
              seed: Optional[int] = None,
              reuse_template: bool = True,
              write_manifest: bool = True,
              initialize: bool = True,
              verbose: bool = True) -> Dict:
    '''
    Generate one block of decoys in this process. This is the unit of parallel
    work: one call per SLURM array task.

    Decoy indices are assigned explicitly (either `indices`, or `count` of them
    starting at `start`), so blocks can never collide on an output filename and
    the old --start-count-from bookkeeping goes away.

    Parameters
    ----------
    config : GlycanDockConfig
        Protocol parameters. If config.prepack_mode == 'once', `complex_pdb`
        must ALREADY be a prepacked structure from prepack().
    start, count : int
        Decoy index range, used when `indices` is not given.
    indices : sequence of int, optional
        Explicit decoy indices, overriding start/count.
    options_file : str, optional
        Rosetta flags file. Defaults to STAGE_2_OPTIONS when
        config.refine_only is set, otherwise STAGE_1_2_OPTIONS.
    run_id : str, optional
        Identifier for the whole multi-block run; also seeds block_seed().
        Defaults to the output prefix.
    block_index : int
        Which block this is (e.g. $SLURM_ARRAY_TASK_ID). Only used for seeding
        and provenance.
    seed : int, optional
        Explicit RNG seed, overriding block_seed(run_id, block_index). Pass
        seed=-1 to disable seeding and use Rosetta's default.
    initialize : bool
        Call init_rosetta() first. Pass False if the caller already did.

    Returns
    -------
    dict : run manifest (config, inputs, seed, outputs, timing summary). Also
        written to <outdir>/<run_id>_block<NNN>_manifest.json unless
        write_manifest is False.
    '''
    if indices is None:
        if count < 1:
            raise ValueError(f'count must be >= 1, got {count}')
        if start < 1:
            raise ValueError(f'decoy indices must start at 1 or above, got {start}')
        indices = list(range(start, start + count))
    else:
        indices = list(indices)
        if not indices:
            raise ValueError('indices is empty; nothing to run')

    if config.prepack_mode == 'once' and verbose:
        print(f"prepack_mode='once': assuming {complex_pdb} is already "
              f"prepacked (Stage 0 will NOT run in this process).")

    outprefix = outprefix or default_outprefix(complex_pdb)
    run_id = run_id or outprefix
    if options_file is None:
        options_file = STAGE_2_OPTIONS if config.refine_only else STAGE_1_2_OPTIONS

    if seed is None:
        seed = block_seed(run_id, block_index)
    elif seed < 0:
        seed = None

    os.makedirs(outdir, exist_ok=True)

    flag_string = None
    if initialize:
        flag_string = init_rosetta(
            options_file, complex_pdb, native=native,
            n_cycles=config.n_cycles, cst_file=cst_file, seed=seed)

    setup_timings: Dict[str, float] = {}
    with _timed(setup_timings, 'runner_setup'):
        runner = GlycanDockRunner(config, complex_pdb, grid_pdb=grid_pdb,
                                  reuse_template=reuse_template)

    timings_path = os.path.join(outdir, f'{run_id}_block{block_index:03d}_timings.jsonl')
    outputs: List[str] = []
    per_decoy: List[Dict[str, float]] = []

    with open(timings_path, 'w') as tf:
        for i in indices:
            outname = os.path.join(outdir, decoy_name(outprefix, i))
            t = runner.run_decoy(i, outname)
            outputs.append(outname)
            per_decoy.append(t)
            tf.write(json.dumps(t) + '\n')
            tf.flush()  # survive a walltime kill with partial timing data
            if verbose:
                print(f'[{run_id} block {block_index}] decoy {i} '
                      f'-> {os.path.basename(outname)} ({t["total"]:.1f}s)')

    summary = _summarize_timings(per_decoy)
    manifest = {
        'run_id': run_id,
        'block_index': block_index,
        'indices': indices,
        'seed': seed,
        'config': config.as_dict(),
        'inputs': {
            'complex_pdb': os.path.abspath(complex_pdb),
            'grid_pdb': os.path.abspath(grid_pdb) if grid_pdb else None,
            'native': os.path.abspath(native) if native else None,
            'cst_file': os.path.abspath(cst_file) if cst_file else None,
            'options_file': os.path.abspath(options_file),
        },
        'rosetta_flags': flag_string,
        'outputs': [os.path.abspath(p) for p in outputs],
        'timings_jsonl': os.path.abspath(timings_path),
        'timing_summary': summary,
        'setup_seconds': setup_timings.get('runner_setup'),
    }

    if write_manifest:
        manifest_path = os.path.join(
            outdir, f'{run_id}_block{block_index:03d}_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        manifest['manifest_path'] = os.path.abspath(manifest_path)

    if verbose:
        _print_timing_summary(summary, len(indices))

    return manifest


def _summarize_timings(per_decoy: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    '''Mean/total per step across a block's decoys.'''
    if not per_decoy:
        return {}
    keys = [k for k in per_decoy[0] if k != 'index']
    summary = {}
    for k in keys:
        vals = [d[k] for d in per_decoy if k in d]
        summary[k] = {
            'total': sum(vals),
            'mean': sum(vals) / len(vals),
            'n': len(vals),
        }
    return summary


def _print_timing_summary(summary: Dict, n_decoys: int) -> None:
    if not summary:
        return
    total = summary.get('total', {}).get('total', 0.0)
    print(f'\n--- timing over {n_decoys} decoys ({total / 60:.1f} min) ---')
    for step, s in sorted(summary.items(),
                          key=lambda kv: -kv[1]['total']):
        if step == 'total':
            continue
        share = (100 * s['total'] / total) if total else 0.0
        print(f'  {step:14s} {s["mean"]:8.2f}s/decoy  {share:5.1f}%')
    prepack = summary.get('prepack', {}).get('total', 0.0)
    gdock = summary.get('glycandock', {}).get('total', 0.0)
    if prepack and gdock:
        print(f'\n  prepack / (prepack + glycandock) = '
              f'{100 * prepack / (prepack + gdock):.1f}%')
        print('  This is the number that decides whether hoisting Stage 0 out '
              'of the loop is worth a protocol change.')
