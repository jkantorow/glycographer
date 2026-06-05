'''
Classes to manage volumetric mapping from a scored ensemble of docked
receptor-glycoligand poses
'''

import os
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple, Dict, List

import MDAnalysis as mda
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from glycographer.dock import GlycanDockEnsemble

@dataclass
class Mapper:
    '''
    Base class for building voxel-wise atom mappings from a GlycanDockEnsemble
    object
    '''
    ensemble: 'GlycanDockEnsemble'
    voxel_size: float = 1.0
    padding: float = 5.0

    _map_type: str = field(default=None, init=False)
    _universe: mda.Universe = field(default=None, init=False)
    _volmap_coords: np.ndarray = field(default=None, init=False)
    _volmap_shape: Tuple = field(default=None, init=False)
    _volmap_origin: np.ndarray = field(default=None, init=False)
    _volmap_values: np.ndarray = field(default=None, init=False)
    _atom_mappings: Dict = field(default=None, init=False)

    def _load_ensemble(self):
        '''Load the ensemble into MDAnalysis universe.'''
        ensemble_file = self.ensemble._ensemble_file
        self._universe = mda.Universe(ensemble_file)
        print(f'Loaded {ensemble_file} with {len(self._universe.trajectory)} frames')        

    def _build_atom_mappings(self):
        '''Build comprehensive atom-to-model mapping data structures.'''
        if self._universe is None:
            self._load_ensemble()
        
        # Get heavy atoms (non-hydrogen) selection
        heavy_atoms = self._universe.select_atoms('not name *H*')
        n_heavy_atoms_per_model = len(heavy_atoms)
        
        all_atom_coords = []
        atom_to_model_map = []
        atom_to_local_index_map = []
        atom_names = []
        atom_indices = []
        model_atom_counts = {}
        
        # Store atom names and indices from first frame (same for all frames)
        self._universe.trajectory[0]
        heavy_atom_names = heavy_atoms.names
        heavy_atom_indices = heavy_atoms.indices
        
        for ts in self._universe.trajectory:
            frame_coords = heavy_atoms.positions  # Only heavy atoms
            model_num = ts.frame + 1
            n_atoms_this_model = len(frame_coords)
            
            all_atom_coords.append(frame_coords)
            model_atom_counts[model_num] = n_atoms_this_model
            
            # Create mappings for each heavy atom in this model
            for local_atom_idx in range(n_atoms_this_model):
                atom_to_model_map.append(model_num)
                atom_to_local_index_map.append(local_atom_idx)
                atom_names.append(heavy_atom_names[local_atom_idx])
                atom_indices.append(heavy_atom_indices[local_atom_idx])
        
        self._atom_mappings = {
            'coords': np.vstack(all_atom_coords),
            'model_map': np.array(atom_to_model_map),
            'local_index_map': np.array(atom_to_local_index_map),
            'atom_names': np.array(atom_names),
            'atom_indices': np.array(atom_indices),
            'model_counts': model_atom_counts,
            'n_heavy_atoms_per_model': n_heavy_atoms_per_model,
            'unique_atom_names': heavy_atom_names,
            'unique_atom_indices': heavy_atom_indices
        }
        
        print(f'Built atom mappings: {len(self._atom_mappings["coords"])} total heavy atoms')
        print(f'Heavy atoms per model: {n_heavy_atoms_per_model}')
        print(f'Unique heavy atom names: {list(heavy_atom_names)}')
        
    def _setup_grid(self):
        '''Setup the voxel grid based on atom coordinates.'''
        if self._atom_mappings is None:
            self._build_atom_mappings()
            
        coords = self._atom_mappings['coords']
        
        # Get grid boundaries
        min_coords = np.min(coords, axis=0) - self.padding
        max_coords = np.max(coords, axis=0) + self.padding
        
        # Calculate grid dimensions
        nx = int((max_coords[0] - min_coords[0]) / self.voxel_size) + 1
        ny = int((max_coords[1] - min_coords[1]) / self.voxel_size) + 1
        nz = int((max_coords[2] - min_coords[2]) / self.voxel_size) + 1
        
        # Create coordinate arrays
        x = np.linspace(min_coords[0], max_coords[0], nx)
        y = np.linspace(min_coords[1], max_coords[1], ny)
        z = np.linspace(min_coords[2], max_coords[2], nz)
        
        self._volmap_coords = np.meshgrid(x, y, z, indexing='ij')
        self._volmap_shape = (nx, ny, nz)
        self._volmap_origin = min_coords
        
        print(f'Grid dimensions: {nx} x {ny} x {nz}')

@dataclass
class IntEngMinMapper(Mapper):
    '''
    Interface for mapping voxel-wise minimum interaction energy values from a
    GlycanDockEnsemble.
    '''
    def __post_init__(self):
        self._map_type = 'intengmin'

    def map(self, debug=False):
        '''
        Use the input docking run data to calculate
        voxel-wise interaction energy scaled by pose
        count:
        '''
        if self._atom_mappings is None:
            self._build_atom_mappings()
        if self._volmap_coords is None:
            self._setup_grid()

        grid_points = np.column_stack([coords.ravel() for coords in self._volmap_coords])
        tree = cKDTree(self._atom_mappings['coords'])
        voxel_radius = self.voxel_size * np.sqrt(3) / 2

        # Initialize all voxel values as 0:
        voxel_values = np.zeros(grid_points.shape[0])

        # Process energy calculation for each atom in each voxel:
        for i, point in enumerate(grid_points):
            if i % 10000 == 0:
                print(f'Mapping progress: {(i/len(grid_points))*100}%')

            # Query voxel for atom coordinates within radius:
            atom_indices_in_voxel = tree.query_ball_point(point, voxel_radius)
            
            # If at least one atom is within this voxel:
            if atom_indices_in_voxel:
                # Get the model id of each atom:
                models_in_voxel = self._atom_mappings['model_map'][atom_indices_in_voxel]
                unique_models_in_voxel = np.unique(models_in_voxel)
                if debug and len(unique_models_in_voxel) > 3:
                    print(f'Model IDs found within voxel {i} @ coords {point}: {models_in_voxel} (unique: {unique_models_in_voxel})')

                # Get interaction_energy scores for models within this voxel:
                model_scores = []
                for model_num in unique_models_in_voxel:
                    score_val = self.ensemble.scoredata.loc[model_num, 'interaction_energy']
                    if pd.notna(score_val):
                        model_scores.append(score_val)

                # Calculate voxel value:
                if model_scores:
                    quality_scores = [s for s in model_scores if s < 0]
                    voxel_values[i] = np.min(quality_scores) if quality_scores else 0.0
                    if debug and len(quality_scores) > 3:
                        print(f'Model scores found for {unique_models_in_voxel} in voxel {i}: {model_scores}')
                        print(f'Quality scores (score < 0) found in voxel {i}: {quality_scores}')
                        print(f'Voxel value assigned: {voxel_values[i]}')

        self._volmap_values = voxel_values

        return VolMap.from_mapper(self)

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
        spacing = [[mapper.voxel_size, 0, 0],
                   [0, mapper.voxel_size, 0],
                   [0, 0, mapper.voxel_size]]

        volmap = cls.__new__(cls)
        volmap.map_id = f'{mapper.ensemble.run_id}_{mapper._map_type}'
        volmap.map_type = mapper._map_type
        volmap.coords = mapper._volmap_coords
        volmap.spacing = spacing
        volmap.shape = mapper._volmap_shape
        volmap.origin = mapper._volmap_origin
        volmap.values = mapper._volmap_values
        volmap.ensemble = mapper.ensemble
        volmap._values_3d = mapper._volmap_values.reshape(mapper._volmap_shape)
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