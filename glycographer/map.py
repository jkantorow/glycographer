'''
Classes to manage volumetric mapping from a scored ensemble of docked
receptor-glycoligand poses
'''

import MDAnalysis as mda
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
import json
import os

from glycographer.dock import GlycanDockEnsemble

@dataclass
class VolMap(GlycanDockEnsemble):
    '''
    Object for reading, storing, and writing voxel grid data
    along with relevant statistical analyses.
    '''
    name: str = None
    maptype: str = None
    origin: Tuple = None
    spacing: List = None
    dims: List = None
    data: np.ndarray = None
    data3d: np.ndarray = None
    mapper: object = None
    resnames: List[str] = field(default_factory=list)
    
    # Derived attributes computed from data
    nzvals: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        '''Initialize derived attributes after dataclass initialization.'''
        self._update_derived_data()

    def _update_derived_data(self):
        '''Update derived data arrays when primary data changes.'''
        if self.data is not None:
            self.nzvals = np.nonzero(self.data)[0]
        elif self.data3d is not None:
            self.data = np.ravel(self.data3d)
            self.nzvals = np.flatnonzero(self.data3d)

    def get_min_voxel_val(self):
        '''Get minimum occupied voxel value in the map.'''
        occupied = self.data[self.data != 0]
        return np.min(occupied) if occupied.any() else None

    def get_max_voxel_val(self):
        '''Get maximum occupied voxel value in the map.'''
        occupied = self.data[self.data != 0]
        return np.max(occupied) if occupied.any() else None

    def get_avg_voxel_val(self):
        '''Get mean value of all data points.'''
        occupied = self.data[self.data != 0]
        return np.mean(occupied) if occupied.any() else None

    @classmethod
    def from_mapper(cls, name, maptype, recname, ligname,
                    lig_iupac, grid_coords, grid_values, spacing, runtag=None):
        '''
        Create a VolMap from calculated grid data contained in a
        Mapper class instance.
            
        Returns:
        --------
        VolMap : New VolMap instance
        '''
        dims = list(grid_values.shape)
        origin = [grid_coords[0][0,0,0], grid_coords[1][0,0,0], grid_coords[2][0,0,0]]
        spacing_vec = [[spacing, 0, 0], [0, spacing, 0], [0, 0, spacing]]
        
        volmap = cls(
            name=name,
            maptype=maptype,
            runtag=runtag,
            recname=recname,
            ligname=ligname,
            lig_iupac=lig_iupac,
            origin=origin,
            spacing=spacing_vec,
            dims=dims,
            data3d=grid_values
        )
        # Update derived data after initialization
        volmap._update_derived_data()
        return volmap
    
    def read_dx(self, dx_file: str):
        '''
        Construct the VolMap data object from an input dx file.
        '''

        # Cache the filepath:
        self._dx_path = os.path.abspath(dx_file)

        # Define map name from dx file if one is not already specified:
        if self.name == '':
            self.name = os.path.basename(dx_file.replace('.dx', ''))
        if self.data or self.data3d:
            print(f'Warning: Map data already exists for map object {self}')
            print(f'Preexisting map data will be overwritten by provided dx data.')
            self.data = []
            self.data3d = []

        with open(dx_file, 'r') as f:
            lines = f.readlines()

        data = []
        spacing = []
        data_started = False

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                # I would like to change this so that
                # dx file constructed from the core
                # logic outputs maptype, receptor,
                # and resname(s) data after the #
                # header to be parsed into the class.
                continue

            if line.startswith('object 1 class gridpositions counts'):
                fields = line.split()
                self.dims = [int(fields[5]), int(fields[6]), int(fields[7])]
            elif line.startswith('origin'):
                fields = line.split()
                self.origin = [float(fields[1]), float(fields[2]), float(fields[3])]
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
                        data.append(float(val))
                    except ValueError:
                        continue
            
        if not data:
            raise ValueError(f'Warning: No numeric data found in map file {dx_file}')
        
        data = np.array(data)
        self.data = data
        self.spacing = spacing

        expected_length = np.prod(self.dims)
        if len(data) != expected_length:
            print(f'Warning: Expected {expected_length} total values; got {len(data)} instead.')
            if len(data) < expected_length:
                data = np.pad(data, (0, expected_length - len(data)), 'constant')
            else:
                data = data[:expected_length]

        self.data3d = data.reshape(self.dims)
        self._update_derived_data()

        return self

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
        if filename is None:
            filename = f'{self.name}.dx'
            
        if self.data3d is None:
            raise ValueError('No 3D data available to write')
        
        nx, ny, nz = self.dims
        
        # Extract spacing from spacing vectors
        if isinstance(self.spacing[0], (list, np.ndarray)):
            dx = self.spacing[0][0]
            dy = self.spacing[1][1] 
            dz = self.spacing[2][2]
        else:
            # Assume uniform spacing
            dx = dy = dz = self.spacing[0] if self.spacing else 1.0

        with open(filename, 'w') as f:
            # Header with metadata
            f.write(f'# {self.maptype} map for {self.recname}\n')
            if self.resnames:
                f.write(f'# Residues: {", ".join(self.resnames)}\n')
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
                        f.write(f'{self.data3d[i,j,k]:.6f} ')
                        value_count += 1
            
            # Ensure file ends with newline; always write a newline so the
            # following attribute lines start on their own line. This prevents
            # the attribute string from being appended to the last data line
            # when the data count is an exact multiple of n_cols.
            f.write('\n')
            f.write('attribute "dep" string "positions"\n')
            f.write('object "density" class field\n')

        # Cache the filepath:
        self._dx_path = os.path.abspath(filename)

        return filename
    
    def to_json(self, filename=None):
        '''Export map metadata and statistics to JSON.'''
        if filename is None:
            filename = f'{self.name}_metadata.json'
            
        metadata = {
            'name': self.name,
            'maptype': self.maptype,
            'recname': self.recname,
            'resnames': self.resnames,
            'origin': self.origin.tolist() if self.origin is not None else None,
            'dims': self.dims,
            'spacing': self.spacing,
            'statistics': {
                'min_val': self.get_min_voxel_val(),
                'max_val': self.get_max_voxel_val(),
                'mean': self.get_avg_voxel_val()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Cache file path:
        self._json_path = os.path.abspath(filename)
            
        return filename

@dataclass
class EnergyMapper(GlycanDockEnsemble):
    '''
    Object for extracting data from a GlycanDockEnsemble instance
    and calculating the volumetric binding energy landscape.
    '''
    glycan_dock_ensemble: GlycanDockEnsemble = None
    voxel_size: float = 1.0
    padding: float = 5.0
    universe: mda.Universe = field(default=None, init=False)
    grid_coords: np.ndarray = field(default=None, init=False)
    grid_shape: Tuple = field(default=None, init=False)
    origin: np.ndarray = field(default=None, init=False)
    _atom_mappings: Dict = field(default=None, init=False)

    def __post_init__(self):
        '''Initialize metadata from the ensemble.'''
        if self.glycan_dock_ensemble:
            # Inherit metadata from the ensemble
            self.runtag = self.glycan_dock_ensemble.runtag
            self.recname = self.glycan_dock_ensemble.recname
            self.ligname = self.glycan_dock_ensemble.ligname
            self.lig_iupac = self.glycan_dock_ensemble.lig_iupac

    def _load_ensemble(self):
        '''Load the ensemble into MDAnalysis universe.'''
        ensemble_file = self.glycan_dock_ensemble.get_ensemble_file()
        self.universe = mda.Universe(ensemble_file)
        print(f'Loaded {ensemble_file} with {len(self.universe.trajectory)} frames')        

    def _build_atom_mappings(self):
        '''Build comprehensive atom-to-model mapping data structures.'''
        if self.universe is None:
            self._load_ensemble()
        
        # Get heavy atoms (non-hydrogen) selection
        heavy_atoms = self.universe.select_atoms('not name *H*')
        n_heavy_atoms_per_model = len(heavy_atoms)
        
        all_atom_coords = []
        atom_to_model_map = []
        atom_to_local_index_map = []
        atom_names = []
        atom_indices = []
        model_atom_counts = {}
        
        # Store atom names and indices from first frame (same for all frames)
        self.universe.trajectory[0]
        heavy_atom_names = heavy_atoms.names
        heavy_atom_indices = heavy_atoms.indices
        
        for ts in self.universe.trajectory:
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
        
        self.grid_coords = np.meshgrid(x, y, z, indexing='ij')
        self.grid_shape = (nx, ny, nz)
        self.origin = min_coords
        
        print(f'Grid dimensions: {nx} x {ny} x {nz}')

    def _map_per_atom_counts(self, debug=False):
        '''
        Generate per-atom count maps for each unique heavy atom in the ligand.
        Returns a dictionary of VolMap objects, one for each unique atom.
        '''
        if not self._atom_mappings:
            self._build_atom_mappings()
        if not self.grid_coords:
            self._setup_grid()

        grid_points = np.column_stack([coords.ravel() for coords in self.grid_coords])
        tree = cKDTree(self._atom_mappings['coords'])
        voxel_radius = self.voxel_size * np.sqrt(3) / 2
        
        # Get unique atom indices and names for creating maps
        unique_atom_indices = self._atom_mappings['unique_atom_indices']
        unique_atom_names = self._atom_mappings['unique_atom_names']
        n_atoms_per_model = self._atom_mappings['n_heavy_atoms_per_model']
        
        # Initialize count maps for each unique atom
        atom_count_maps = {}
        for atom_idx, atom_name in zip(unique_atom_indices, unique_atom_names):
            atom_count_maps[f'{atom_name}_{atom_idx}'] = np.zeros(grid_points.shape[0])
            
        print(f'Mapping counts for {len(unique_atom_indices)} unique heavy atoms')
        
        # Process each voxel
        for i, point in enumerate(grid_points):
            if i % 10000 == 0:
                print(f'Per-atom mapping progress: {(i/len(grid_points))*100:.1f}%')

            # Query voxel for atom coordinates within radius
            atom_indices_in_voxel = tree.query_ball_point(point, voxel_radius)
            
            if atom_indices_in_voxel:
                # Get model numbers and local atom indices for atoms in this voxel
                models_in_voxel = self._atom_mappings['model_map'][atom_indices_in_voxel]
                local_atom_indices = self._atom_mappings['local_index_map'][atom_indices_in_voxel]
                
                # Count occurrences of each atom type
                for atom_idx, atom_name in zip(unique_atom_indices, unique_atom_names):
                    # Convert atom_idx to local index (0-based within heavy atoms)
                    local_heavy_atom_idx = np.where(unique_atom_indices == atom_idx)[0][0]
                    
                    # Count how many times this specific atom appears in the voxel
                    count = np.sum(local_atom_indices == local_heavy_atom_idx)
                    atom_count_maps[f'{atom_name}_{atom_idx}'][i] = count
                    
                    if debug and count > 0:
                        print(f'Voxel {i}: Found {count} instances of atom {atom_name}_{atom_idx}')

        # Convert to VolMap objects
        volmaps = {}
        for atom_key, count_data in atom_count_maps.items():
            count_data_3d = count_data.reshape(self.grid_shape)
            volmap = VolMap.from_mapper(
                name=f'{self.runtag}_{atom_key}_count',
                maptype=f'{atom_key}_count',
                runtag=self.runtag,
                recname=self.recname,
                ligname=self.ligname,
                lig_iupac=self.lig_iupac,
                grid_coords=self.grid_coords,
                grid_values=count_data_3d,
                spacing=self.voxel_size,
            )
            volmaps[f'{atom_key}_count'] = volmap
            
        return volmaps

    def map(self, debug=False, include_atom_counts=False):
        '''
        Use the input docking run data to calculate
        voxel-wise interaction energy scaled by pose
        count:
        '''
        if not self._atom_mappings:
            self._build_atom_mappings()
        if not self.grid_coords:
            self._setup_grid()

        grid_points = np.column_stack([coords.ravel() for coords in self.grid_coords])
        tree = cKDTree(self._atom_mappings['coords'])
        voxel_radius = self.voxel_size * np.sqrt(3) / 2

        # Initialize all voxel values as 0:
        energy_map = np.zeros(grid_points.shape[0])

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
                    score_val = self.glycan_dock_ensemble.scoredata.loc[model_num, 'interaction_energy']
                    if pd.notna(score_val):
                        model_scores.append(score_val)

                # Calculate voxel value:
                if model_scores:
                    quality_scores = [s for s in model_scores if s < 0]
                    # Here's where we have a decision to make:

                    # sum all interaction energy scores and scale 
                    # by the fraction of voxel models vs ensemble models:
                    energy_map[i] = np.min(quality_scores) if quality_scores else 0.0
                    if debug and len(quality_scores) > 3:
                        print(f'Model scores found for {unique_models_in_voxel} in voxel {i}: {model_scores}')
                        print(f'Quality scores (score < 0) found in voxel {i}: {quality_scores}')
                        print(f'Voxel value assigned: {np.min(quality_scores)}')

        maps = {}
        maps['energy_min'] = energy_map
        self.maps = maps

        # Return map data as VolMap object instances:
        volmaps = {}
        for map_id, map_data in maps.items():
            map_data_3d = map_data.reshape(self.grid_shape)
            volmap = VolMap.from_mapper(
                name=f'{self.runtag}_{map_id}',
                maptype=map_id,
                runtag=self.runtag,
                recname=self.recname,
                ligname=self.ligname,
                lig_iupac=self.lig_iupac,
                grid_coords=self.grid_coords,
                grid_values=map_data_3d,
                spacing=self.voxel_size,
            )
            volmaps[map_id] = volmap

        # Optionally include per-atom count maps
        if include_atom_counts:
            print('Generating per-atom count maps...')
            atom_count_maps = self._map_per_atom_counts(debug=debug)
            volmaps.update(atom_count_maps)

        return volmaps