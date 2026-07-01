'''
Classes to manage volumetric mapping from a scored ensemble of docked
receptor-glycoligand poses.

The mapping stack is built around three ideas:

1. A shared GridSpec (origin + shape + voxel_size) so that every probe run
   scored against the same receptor is voxelized on an identical grid. Voxel
   index i then means the same physical location in every map, which is what
   makes cross-probe consensus mapping possible.

2. A template-method Mapper: the base class owns the whole voxelization loop
   (atom binning, per-voxel deduplication, reduction) and subclasses only
   supply (a) where the per-contribution energies come from and (b) how the
   energies occupying a voxel are reduced to a single value.

3. Atoms are binned into cubic cells by integer division -- floor((xyz -
   origin) / voxel_size) -- so each atom lands in exactly one voxel. This is
   both a correctness fix (the old cKDTree sphere query overlapped neighboring
   voxels and double-counted atoms) and ~100x faster than looping over grid
   points.
'''

import os
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple, Dict, List, Optional

import MDAnalysis as mda
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from glycographer.dock import GlycanDockEnsemble


# ---------------------------------------------------------------------------
# Shared grid definition
# ---------------------------------------------------------------------------
@dataclass
class GridSpec:
    '''
    An axis-aligned cubic voxel grid: a lower corner (origin), an integer
    shape (nx, ny, nz), and an isotropic voxel_size. A grid point / cell i,j,k
    has its lower corner at origin + (i, j, k) * voxel_size.

    Build one GridSpec and hand it to every Mapper that shares a receptor so
    all resulting VolMaps are index-aligned and can be combined into a
    ConsensusMap.
    '''
    origin: np.ndarray
    shape: Tuple[int, int, int]
    voxel_size: float = 1.0

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=float)
        self.shape = tuple(int(n) for n in self.shape)

    @classmethod
    def from_bounds(cls, min_corner, max_corner, voxel_size=1.0, padding=0.0):
        '''Build a grid spanning [min_corner - padding, max_corner + padding].'''
        min_c = np.asarray(min_corner, dtype=float) - padding
        max_c = np.asarray(max_corner, dtype=float) + padding
        shape = tuple(int(np.ceil((max_c[d] - min_c[d]) / voxel_size)) + 1
                      for d in range(3))
        return cls(origin=min_c, shape=shape, voxel_size=voxel_size)

    @classmethod
    def from_coords(cls, coords, voxel_size=1.0, padding=5.0):
        '''Build a grid enclosing a single (N, 3) coordinate array.'''
        coords = np.asarray(coords, dtype=float)
        return cls.from_bounds(coords.min(axis=0), coords.max(axis=0),
                               voxel_size=voxel_size, padding=padding)

    @classmethod
    def from_pdb(cls, pdb_file, voxel_size=1.0, padding=5.0,
                 selection='not name *H*'):
        '''
        Build a grid from a structure file (receptor PDB, or the docking
        gridbox pseudoatom PDB). For the docking gridbox use selection='all'
        and a generous padding to cover the glycan reach beyond the seed shell.
        '''
        u = mda.Universe(pdb_file)
        coords = u.select_atoms(selection).positions
        return cls.from_coords(coords, voxel_size=voxel_size, padding=padding)

    @classmethod
    def from_ensembles(cls, ensembles, voxel_size=1.0, padding=5.0,
                       include_receptor_pdb=None):
        '''
        Build a single grid enclosing the ligand atoms of every ensemble in
        `ensembles`. This is the recommended constructor for a consensus set:
        it guarantees every pose of every probe falls inside the grid, so all
        maps share the same voxel indexing.

        Parameters
        ----------
        ensembles : iterable of GlycanDockEnsemble
            Ensembles must already have their ensemble .pdb written
            (call ensemble.to_pdb(...) first).
        include_receptor_pdb : str, optional
            Also enclose the receptor's heavy atoms (useful if you want the
            grid pinned to the full receptor frame rather than just the
            sampled ligand volume).
        '''
        mins, maxs = [], []
        for ens in ensembles:
            ens_file = ens.get_ensemble_file()
            if not ens_file:
                raise ValueError(
                    f'Ensemble {getattr(ens, "run_id", ens)} has no ensemble '
                    f'file; call to_pdb() before building a shared grid.')
            lo, hi = cls._coord_bounds(ens_file)
            mins.append(lo)
            maxs.append(hi)
        if include_receptor_pdb is not None:
            u = mda.Universe(include_receptor_pdb)
            rec = u.select_atoms('not name *H*').positions
            mins.append(rec.min(axis=0))
            maxs.append(rec.max(axis=0))
        min_c = np.min(np.vstack(mins), axis=0)
        max_c = np.max(np.vstack(maxs), axis=0)
        return cls.from_bounds(min_c, max_c, voxel_size=voxel_size,
                               padding=padding)

    @staticmethod
    def _coord_bounds(pdb_file, selection='not name *H*'):
        '''Min/max heavy-atom coordinates over all frames of a (multi-model) file.'''
        u = mda.Universe(pdb_file)
        atoms = u.select_atoms(selection)
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for _ in u.trajectory:
            pos = atoms.positions
            lo = np.minimum(lo, pos.min(axis=0))
            hi = np.maximum(hi, pos.max(axis=0))
        return lo, hi

    @property
    def n_voxels(self):
        return int(np.prod(self.shape))

    @property
    def spacing(self):
        '''DX-style delta vectors.'''
        v = self.voxel_size
        return [[v, 0, 0], [0, v, 0], [0, 0, v]]

    def cell_indices(self, coords):
        '''Integer (i, j, k) cell index for each coordinate in (N, 3) coords.'''
        return np.floor((np.asarray(coords, dtype=float) - self.origin)
                        / self.voxel_size).astype(np.int64)

    def inside_mask(self, ijk):
        '''Boolean mask of which (N, 3) integer indices fall inside the grid.'''
        shape = np.asarray(self.shape)
        return np.all((ijk >= 0) & (ijk < shape), axis=1)

    def flat_index(self, ijk):
        '''C-order flat index for in-bounds integer indices (N, 3).'''
        return np.ravel_multi_index(ijk.T, self.shape)


# ---------------------------------------------------------------------------
# Mapper base class (template method) and concrete mappers
# ---------------------------------------------------------------------------
@dataclass
class Mapper:
    '''
    Base class for building voxel-wise mappings from a GlycanDockEnsemble.

    Subclasses customize two hooks:
      _atom_keys_and_energies() -> (keys, energies) aligned with the stacked
          heavy-atom coordinates. `keys` identifies the independent
          contribution an atom belongs to (a pose, or a pose+residue) so that
          each contribution is only counted once per voxel; `energies` is the
          per-contribution energy attached to that atom.
      _reduce(energies) -> float
          Collapse the (deduplicated) energies occupying one voxel into a
          single voxel value.

    Optionally override _prepare_scores() to derive score columns before mapping.
    '''
    ensemble: 'GlycanDockEnsemble'
    voxel_size: float = 1.0
    padding: float = 5.0
    grid: Optional[GridSpec] = None

    _map_type: str = field(default=None, init=False)
    _score_column: str = field(default='interaction_energy', init=False)
    _universe: mda.Universe = field(default=None, init=False)
    _volmap_shape: Tuple = field(default=None, init=False)
    _volmap_origin: np.ndarray = field(default=None, init=False)
    _volmap_values: np.ndarray = field(default=None, init=False)
    _atom_mappings: Dict = field(default=None, init=False)

    # ---- ensemble / atom bookkeeping ------------------------------------
    def _load_ensemble(self):
        '''Load the ensemble into an MDAnalysis universe.'''
        ensemble_file = self.ensemble.get_ensemble_file()
        if not ensemble_file:
            raise ValueError(
                'No ensemble file available; call ensemble.to_pdb() first.')
        self._universe = mda.Universe(ensemble_file)
        print(f'Loaded {ensemble_file} with '
              f'{len(self._universe.trajectory)} frames')

    def _build_atom_mappings(self):
        '''
        Stack heavy-atom coordinates over all frames and record, per atom,
        which model and which residue it came from.
        '''
        if self._universe is None:
            self._load_ensemble()

        heavy = self._universe.select_atoms('not name *H*')
        n_heavy = len(heavy)

        # Atom identity is constant across frames; capture once.
        self._universe.trajectory[0]
        heavy_names = heavy.names
        heavy_resids = heavy.resids

        all_coords = []
        model_map = []
        resid_map = []
        for ts in self._universe.trajectory:
            all_coords.append(heavy.positions)
            model_num = ts.frame + 1
            model_map.append(np.full(n_heavy, model_num, dtype=np.int64))
            resid_map.append(heavy_resids.astype(np.int64))

        self._atom_mappings = {
            'coords': np.vstack(all_coords),
            'model_map': np.concatenate(model_map),
            'resid_map': np.concatenate(resid_map),
            'atom_names': np.tile(heavy_names, len(all_coords)),
            'n_heavy_atoms_per_model': n_heavy,
            'unique_atom_names': heavy_names,
            'unique_resids': np.unique(heavy_resids),
        }

        print(f'Built atom mappings: {len(self._atom_mappings["coords"])} '
              f'total heavy atoms ({n_heavy} per model)')

    def _setup_grid(self):
        '''
        Ensure self.grid is set. If no shared GridSpec was provided, build one
        enclosing this ensemble's own ligand atoms (single-ensemble fallback).
        '''
        if self.grid is not None:
            # A shared grid is authoritative for spacing.
            self.voxel_size = self.grid.voxel_size
            return
        if self._atom_mappings is None:
            self._build_atom_mappings()
        self.grid = GridSpec.from_coords(
            self._atom_mappings['coords'],
            voxel_size=self.voxel_size, padding=self.padding)
        print(f'Grid dimensions: {self.grid.shape[0]} x '
              f'{self.grid.shape[1]} x {self.grid.shape[2]} '
              f'(no shared grid supplied; built from this ensemble)')

    # ---- hooks for subclasses -------------------------------------------
    def _prepare_scores(self):
        '''Ensure any derived score columns exist. Override as needed.'''
        pass

    def _per_pose_keys_and_energies(self, score_column):
        '''
        Per-pose contributions: every atom carries its pose's whole-ligand
        score. The dedup key is the model number, so a pose that puts several
        atoms in one voxel is counted once.
        '''
        models = self._atom_mappings['model_map']
        scores = pd.to_numeric(self.ensemble.scoredata[score_column],
                               errors='coerce')
        energies = scores.reindex(models).to_numpy()
        return models, energies

    def _atom_keys_and_energies(self):
        '''Default: per-pose contributions from self._score_column.'''
        return self._per_pose_keys_and_energies(self._score_column)

    def _reduce(self, energies):
        '''Collapse per-voxel energies to a single value.'''
        raise NotImplementedError

    # ---- template method: the shared voxelization loop ------------------
    def map(self, debug=False):
        '''
        Voxelize the ensemble: bin every heavy atom into a cubic cell, keep one
        contribution per (voxel, key), and reduce the energies in each voxel to
        a single value via the subclass _reduce hook.
        '''
        self._prepare_scores()
        if self._atom_mappings is None:
            self._build_atom_mappings()
        self._setup_grid()
        grid = self.grid

        coords = self._atom_mappings['coords']
        ijk = grid.cell_indices(coords)
        inside = grid.inside_mask(ijk)
        n_out = int(np.count_nonzero(~inside))
        if n_out:
            print(f'Warning: {n_out} atoms fell outside the grid and were '
                  f'ignored. Widen the grid padding or build it from the '
                  f'union of all ensembles (GridSpec.from_ensembles).')

        flat = grid.flat_index(ijk[inside])
        keys, energies = self._atom_keys_and_energies()
        keys = keys[inside]
        energies = np.asarray(energies, dtype=float)[inside]

        df = pd.DataFrame({'voxel': flat, 'key': keys, 'energy': energies})
        df = df.dropna(subset=['energy'])
        # One contribution per (voxel, key): all atoms of a contribution share
        # the same energy, so drop_duplicates leaves one representative each.
        df = df.drop_duplicates(subset=['voxel', 'key'])

        if debug:
            counts = df.groupby('voxel')['key'].nunique()
            busy = counts[counts > 3]
            print(f'{len(counts)} occupied voxels; '
                  f'{len(busy)} with >3 contributions.')

        reduced = df.groupby('voxel')['energy'].agg(self._reduce)

        voxel_values = np.zeros(grid.n_voxels, dtype=float)
        if len(reduced):
            voxel_values[reduced.index.to_numpy()] = reduced.to_numpy()

        self._volmap_values = voxel_values
        self._volmap_origin = grid.origin
        self._volmap_shape = grid.shape
        self.voxel_size = grid.voxel_size

        return VolMap.from_mapper(self)


@dataclass
class IntEngMinMapper(Mapper):
    '''
    Voxel value = minimum favorable (negative) whole-pose interaction energy of
    any pose occupying the voxel. Outlier-sensitive; kept for compatibility and
    as the beta -> infinity limit of the Boltzmann mapper.
    '''
    def __post_init__(self):
        self._map_type = 'intengmin'
        self._score_column = 'interaction_energy'

    def _reduce(self, energies):
        favorable = np.asarray(energies)[np.asarray(energies) < 0]
        return float(favorable.min()) if favorable.size else 0.0


@dataclass
class IntEngAvgMapper(Mapper):
    '''
    Voxel value = mean of the min-max-scaled interaction energies of the poses
    occupying the voxel. Kept for compatibility; note this uses the [0, 1]
    "higher is better" convention and so is NOT directly comparable to the
    REU-scale maps in a consensus.
    '''
    def __post_init__(self):
        self._map_type = 'intengavg'
        self._score_column = 'scaled_interaction_energy'

    def _prepare_scores(self):
        if 'scaled_interaction_energy' not in self.ensemble.scoredata:
            self.ensemble.scale_inteng()

    def _reduce(self, energies):
        return float(np.mean(energies))


@dataclass
class BoltzmannMapper(Mapper):
    '''
    Voxel value = Boltzmann-weighted soft-min of the whole-pose interaction
    energies occupying the voxel:

        G_v = -(1/beta) * ln( sum_i exp(-beta * E_i) )

    This interpolates between the min mapper (beta -> infinity) and an
    occupancy-weighted average (beta -> 0). It rewards voxels visited by many
    favorable poses without letting a single outlier dominate. Values are on
    the REU scale (favorable = negative), so BoltzmannMappers over different
    probes are directly comparable in a ConsensusMap.

    Treat `beta` as an empirical focusing knob (NOT 1/kT): tune it against
    known complexes so the map neither collapses onto one voxel (too high) nor
    flattens out (too low). It also serves as your hotspot cutoff determinant.
    '''
    beta: float = 0.5

    def __post_init__(self):
        self._map_type = 'boltzmann'
        self._score_column = 'interaction_energy'

    def _reduce(self, energies):
        e = np.asarray(energies, dtype=float)
        return float(-np.logaddexp.reduce(-self.beta * e) / self.beta)


@dataclass
class ResidueBoltzmannMapper(BoltzmannMapper):
    '''
    Per-residue Boltzmann soft-min. Instead of stamping each atom with its
    whole-pose score, each atom carries the REF15 energy of the *glycan residue
    it belongs to* (summed over energy terms from ensemble.residue_energies).
    Contributions are deduplicated per (model, residue), so this resolves which
    fragment actually sits in a voxel -- the SILCS-style fragment quantity --
    and sharpens anchors while deflating hotspots inherited from floppy
    residues elsewhere in the same pose.

    NOTE on residue numbering: this matches the ensemble PDB's residue ids
    (resid) against residue_energies['residue_num']. Those come from the
    Rosetta POSE_ENERGIES_TABLE labels; if your PDB resSeq differs from the
    Rosetta pose numbering the match will fail and you'll get an empty map --
    a warning is printed listing the mismatch so you can supply a mapping.
    '''
    def __post_init__(self):
        super().__post_init__()
        self._map_type = 'residue_boltzmann'

    def _prepare_scores(self):
        if self.ensemble.residue_energies is None:
            raise ValueError(
                'ResidueBoltzmannMapper needs per-residue energies; run '
                'read_poses/from_poses with parse_energies=True first.')

    def _atom_keys_and_energies(self):
        models = self._atom_mappings['model_map']
        resids = self._atom_mappings['resid_map']

        re = self.ensemble.residue_energies
        gly = re[re.is_glycan]
        # Total (summed-over-terms) energy per (model, residue):
        res_energy = gly.groupby(['model_num', 'residue_num'])['weighted'].sum()

        # Warn early if the PDB residue ids don't line up with the energy table.
        pdb_resids = set(np.unique(resids).tolist())
        table_resids = set(np.unique(gly['residue_num']).tolist())
        if pdb_resids.isdisjoint(table_resids):
            print('Warning: ensemble PDB residue ids '
                  f'{sorted(pdb_resids)} do not intersect residue_energies '
                  f'residue_num {sorted(table_resids)}. The per-residue map '
                  'will be empty -- PDB resSeq likely differs from Rosetta '
                  'pose numbering; supply a resid->pose-num mapping.')

        idx = pd.MultiIndex.from_arrays([models, resids],
                                        names=['model_num', 'residue_num'])
        energies = res_energy.reindex(idx).to_numpy()

        # Dedup key encodes (model, residue) into a single integer.
        keys = models * 1_000_000 + resids
        return keys, energies


# ---------------------------------------------------------------------------
# VolMap: read / store / write voxel grids
# ---------------------------------------------------------------------------
@dataclass
class VolMap:
    '''
    Object for reading, storing, and writing voxel grid data.
    '''
    map_id: str = None
    map_type: str = None
    coords: np.ndarray = None
    spacing: List = None
    shape: List = None
    origin: Tuple = None
    values: np.ndarray = None
    ensemble: 'GlycanDockEnsemble' = None

    _values_3d: np.ndarray = field(default=None, init=False)
    _dx_file: str = field(default=None, init=False)
    _dx_path: str = field(default=None, init=False)

    @classmethod
    def from_dx(cls, dx_file: str):
        '''
        Construct the VolMap data object from an input dx file.
        '''
        # Cache the filepath:
        _dx_path = os.path.abspath(dx_file)

        # Define map name from dx file:
        map_id = os.path.basename(dx_file.replace('.dx', ''))

        with open(dx_file, 'r') as f:
            lines = f.readlines()

        values = []
        spacing = []
        data_started = False
        map_type = None
        shape = None
        origin = None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                fields = line.split()
                if len(fields) >= 3 and fields[1] == 'map_type':
                    map_type = fields[2]
            if line.startswith('object 1 class gridpositions counts'):
                fields = line.split()
                shape = [int(fields[5]), int(fields[6]), int(fields[7])]
            elif line.startswith('origin'):
                fields = line.split()
                origin = [float(fields[1]), float(fields[2]), float(fields[3])]
            elif line.startswith('delta'):
                fields = line.split()
                spacing.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif 'data follows' in line and 'object 3 class array' in line:
                data_started = True
                continue
            elif data_started and not line.startswith(('attribute', 'object', 'component')):
                fields = line.split()
                for val in fields:
                    try:
                        values.append(float(val))
                    except ValueError:
                        continue

        if shape is None:
            raise ValueError(f'Could not read grid dimensions from map file {dx_file}')
        if origin is None:
            raise ValueError(f'Could not read origin from map file {dx_file}')
        if not values:
            raise ValueError(f'Warning: No numeric data found in map file {dx_file}')
        expected_length = int(np.prod(shape))
        if len(values) != expected_length:
            print(f'Warning: Expected {expected_length} total values; got {len(values)} instead.')
            if len(values) < expected_length:
                values = np.pad(values, (0, expected_length - len(values)), 'constant')
            else:
                values = values[:expected_length]

        values = np.asarray(values, dtype=float)
        _values_3d = values.reshape(shape)

        volmap = cls.__new__(cls)
        volmap.map_id = map_id
        volmap.map_type = map_type
        volmap.coords = None
        volmap.spacing = spacing
        volmap.shape = shape
        volmap.origin = np.asarray(origin, dtype=float)
        volmap.values = values
        volmap.ensemble = None
        volmap._values_3d = _values_3d
        volmap._dx_file = dx_file
        volmap._dx_path = _dx_path

        return volmap

    @classmethod
    def from_mapper(cls, mapper: Mapper):
        '''
        Construct a VolMap object from data calculated by and stored within
        a Mapper object.
        '''
        return cls.from_values(
            values=mapper._volmap_values,
            shape=mapper._volmap_shape,
            origin=mapper._volmap_origin,
            voxel_size=mapper.voxel_size,
            map_id=f'{mapper.ensemble.run_id}_{mapper._map_type}',
            map_type=mapper._map_type,
            ensemble=mapper.ensemble,
        )

    @classmethod
    def from_values(cls, values, shape, origin, voxel_size, map_id,
                    map_type, ensemble=None):
        '''Construct a VolMap directly from a flat value array + grid geometry.'''
        spacing = [[voxel_size, 0, 0],
                   [0, voxel_size, 0],
                   [0, 0, voxel_size]]

        values = np.asarray(values, dtype=float)
        volmap = cls.__new__(cls)
        volmap.map_id = map_id
        volmap.map_type = map_type
        volmap.coords = None
        volmap.spacing = spacing
        volmap.shape = list(shape)
        volmap.origin = np.asarray(origin, dtype=float)
        volmap.values = values
        volmap.ensemble = ensemble
        volmap._values_3d = values.reshape(tuple(shape))
        volmap._dx_file = None
        volmap._dx_path = None
        return volmap

    def to_dx(self, filename=None, n_cols=3):
        '''
        Write the VolMap data to a DX format file.

        Parameters:
        -----------
        filename : str, optional
            Output filename. If None, uses map name

        n_cols : int
            Number of values per line in output file

        Returns:
        --------
        str : Path to written file
        '''
        if not filename:
            filename = f'{self.map_id}.dx'

        if self._values_3d is None:
            raise ValueError('No 3D data available to write')

        nx, ny, nz = self.shape

        # Extract spacing from spacing vectors
        if isinstance(self.spacing[0], (list, np.ndarray)):
            dx = self.spacing[0][0]
            dy = self.spacing[1][1]
            dz = self.spacing[2][2]
        else:
            # Assume uniform spacing
            dx = dy = dz = self.spacing[0] if self.spacing else 1.0

        with open(filename, 'w') as f:
            # Header with metadata (need to add more detail)
            f.write(f'# map_type {self.map_type}\n')
            f.write(f'object 1 class gridpositions counts {nx} {ny} {nz}\n')
            f.write(f'origin {self.origin[0]:.6f} {self.origin[1]:.6f} {self.origin[2]:.6f}\n')
            f.write(f'delta {dx:.6f} 0.000000 0.000000\n')
            f.write(f'delta 0.000000 {dy:.6f} 0.000000\n')
            f.write(f'delta 0.000000 0.000000 {dz:.6f}\n')
            f.write(f'object 2 class gridconnections counts {nx} {ny} {nz}\n')
            f.write(f'object 3 class array type double rank 0 items {nx*ny*nz} data follows\n')

            # Data - DX format expects Z to vary fastest, then Y, then X
            value_count = 0
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if value_count % n_cols == 0 and value_count > 0:
                            f.write('\n')
                        f.write(f'{self._values_3d[i,j,k]:.6f} ')
                        value_count += 1

            f.write('\n')
            f.write('attribute "dep" string "positions"\n')
            f.write('object "density" class field\n')

        # Cache the file data:
        self._dx_file = filename
        self._dx_path = os.path.abspath(filename)

        print(f'Wrote {self.map_type} type map in OpenDX format to {self._dx_path}')

        return filename

    def to_json(self, filename=None):
        '''Export map metadata to JSON.'''
        if filename is None:
            filename = f'{self.map_id}_metadata.json'

        metadata = {
            'map_id': self.map_id,
            'map_type': self.map_type,
            'origin': self.origin.tolist() if self.origin is not None else None,
            'shape': self.shape,
            'spacing': self.spacing,
            'dx_file': self._dx_file,
            'dx_path': self._dx_path
        }

        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f'Wrote JSON metadata to {os.path.abspath(filename)}')

        return filename


# ---------------------------------------------------------------------------
# Consensus mapping across probes on a shared grid
# ---------------------------------------------------------------------------
@dataclass
class ConsensusMap:
    '''
    Combine several index-aligned VolMaps (different probes, same receptor,
    same GridSpec) into consensus / selectivity fields.

    All input maps MUST share origin, shape, and spacing -- build them with a
    single shared GridSpec (e.g. GridSpec.from_ensembles). Because empty voxels
    carry a sentinel value of 0.0, reductions treat exactly-zero voxels as
    "this probe did not visit here" (NaN) so they don't dilute the consensus.

    Intended for REU-scale maps where favorable = negative (BoltzmannMapper /
    ResidueBoltzmannMapper / IntEngMinMapper). Mixing in the [0, 1]
    IntEngAvgMapper convention will give meaningless results.
    '''
    maps: List[VolMap]
    probe_labels: List[str] = None

    _stack: np.ndarray = field(default=None, init=False)   # (n_probes, n_vox)
    _masked: np.ndarray = field(default=None, init=False)  # zeros -> NaN

    def __post_init__(self):
        if len(self.maps) < 2:
            raise ValueError('ConsensusMap needs at least two maps.')
        ref = self.maps[0]
        for m in self.maps[1:]:
            if (list(m.shape) != list(ref.shape)
                    or not np.allclose(m.origin, ref.origin)
                    or not np.allclose(np.ravel(m.spacing), np.ravel(ref.spacing))):
                raise ValueError(
                    'All maps must share grid geometry (origin/shape/spacing). '
                    'Build them from one shared GridSpec.')
        types = {m.map_type for m in self.maps}
        if len(types) > 1:
            print(f'Warning: combining maps of differing types {types}; '
                  'consensus is only meaningful across a common convention.')

        if self.probe_labels is None:
            self.probe_labels = [m.map_id for m in self.maps]

        self._stack = np.vstack([m.values for m in self.maps])
        self._masked = np.where(self._stack == 0.0, np.nan, self._stack)

    def _as_map(self, values, tag):
        ref = self.maps[0]
        v = ref.spacing[0][0] if isinstance(ref.spacing[0], (list, np.ndarray)) \
            else ref.spacing
        return VolMap.from_values(
            values=np.nan_to_num(values, nan=0.0),
            shape=ref.shape, origin=ref.origin, voxel_size=v,
            map_id=f'consensus_{tag}', map_type=f'consensus_{tag}')

    def consensus_min(self):
        '''Best-case favorability at each voxel (most favorable across probes).'''
        return self._as_map(np.nanmin(self._masked, axis=0), 'min')

    def consensus_mean(self):
        '''
        Average favorability across probes that visit each voxel -- generalist
        anchor sites (favorable to many fragment types) score most negative.
        '''
        with np.errstate(invalid='ignore'):
            return self._as_map(np.nanmean(self._masked, axis=0), 'mean')

    def support_count(self, threshold=0.0):
        '''
        Number of probes with a favorable value (< threshold) at each voxel.
        A reproducibility/generalist filter: high count == many probes bind.
        '''
        counts = np.nansum(self._masked < threshold, axis=0)
        return self._as_map(counts.astype(float), 'support')

    def best_probe(self):
        '''
        Identity map: 1-based index of the probe with the most favorable value
        at each voxel (0 where no probe visits). Feeds anchor selection --
        which fragment binds best where.
        '''
        all_nan = np.all(np.isnan(self._masked), axis=0)
        filled = np.where(np.isnan(self._masked), np.inf, self._masked)
        best = np.argmin(filled, axis=0) + 1
        best[all_nan] = 0
        return self._as_map(best.astype(float), 'best_probe')

    def selectivity_entropy(self, beta=0.5):
        '''
        Shannon entropy (nats) of the Boltzmann weights across probes at each
        voxel. Low entropy == the site strongly prefers one probe (a
        specificity determinant); high entropy == promiscuous. Empty voxels -> 0.
        '''
        w = np.exp(-beta * np.where(np.isnan(self._masked), np.inf, self._masked))
        z = w.sum(axis=0)
        occupied = z > 0
        p = np.divide(w, z, out=np.zeros_like(w), where=occupied)
        with np.errstate(divide='ignore', invalid='ignore'):
            ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=0)
        ent[~occupied] = 0.0
        return self._as_map(ent, 'selectivity')
