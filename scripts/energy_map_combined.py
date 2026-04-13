#!/usr/bin/env python3

'''
Dataclass objects used for processing glycan interaction map calculations.
'''

import MDAnalysis as mda
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
import json
import os

@dataclass
class GlycanDockMetadata:
    '''
    Base class for storing common metadata across glycan docking classes.
    '''
    runtag: str = None
    recname: str = None
    ligname: str = None
    lig_iupac: str = None

@dataclass
class GlycanDockEnsemble(GlycanDockMetadata):
    '''
    Object for storing and accessing data describing
    the output of a GlycanDock sampling simulation.
    '''
    complex_file: str = None
    pose_files: List[str] = None
    grid_box_file: str = None
    scoredata: pd.DataFrame = None

    # Cached attributes (use field with init=False for derived attributes):
    _rec_chain_id: str = field(default='A', init=False)
    _lig_chain_id: str = field(default='X', init=False)
    _n_poses: int = field(default=None, init=False)
    _complex_path: os.PathLike = field(default=None, init=False)
    _grid_box_path: os.PathLike = field(default=None, init=False)
    _pose_dir: os.PathLike = field(default=None, init=False)
    _ensemble_path: os.PathLike = field(default=None, init=False)
    _ensemble_file: str = field(default=None, init=False)
    clusters: Dict = field(default=None, init=False)

    def __post_init__(self):
        '''Initialize derived attributes after dataclass initialization.'''
        if self.complex_file:
            self._complex_path = os.path.abspath(self.complex_file)
        if self.grid_box_file:
            self._grid_box_path = os.path.abspath(self.grid_box_file)

    # Class-level constant for standard GlycanDock score names
    STANDARD_SCORE_NAMES = [
        'Fnat', 'Fnat_intf_residues', 'glycan_Jump_res',
        'heavy_Lrmsd', 'heavy_Srmsd', 'interaction_energy',
        'mc_acceptance', 'n_intf_res_contacts', 'n_intf_residues',
        'n_nat_intf_res_contacts', 'n_nat_intf_residues',
        'n_rb_cycles', 'n_rb_moves_accepted', 'n_rb_moves_made',
        'n_tor_cycles', 'n_tor_moves_accepted', 'n_tor_moves_made',
        'ring_Lrmsd', 'ring_Srmsd'
    ]

    def read_poses(self, pose_list, score_names=None):
        '''
        Import object data from a collection of Rosetta output
        poses.
        '''
        # Create a list of each pose file:
        if not self.pose_files:
            pose_files = sorted(pose_list)
            self.pose_files = pose_files
            self._pose_dir = os.path.dirname(pose_files[0])
                    
        if not self._n_poses:
            self._n_poses = len(self.pose_files)

        # Use provided score names or default to standard ones:
        if score_names is None:
            score_names = self.STANDARD_SCORE_NAMES.copy()
        
        # Create a pandas DataFrame for storing score data for each pose:
        # Use model numbers as index for easier lookup
        model_indices = list(range(1, self._n_poses + 1))
        scoredata = pd.DataFrame(index=model_indices, columns=score_names)
        scoredata.index.name = 'model_num'
        
        # Add a column to track the original pose file for each model
        scoredata['pose_file'] = ''

        # Parse each pose file to extract data:
        for i, file in enumerate(self.pose_files):
            model_num = i + 1
            scoredata.loc[model_num, 'pose_file'] = file
            
            with open(file, 'r') as f:
                lines = f.readlines()

            # Extract scores and metadata from pose file:
            for line in lines:
                if not self.lig_iupac and line.startswith('->'):
                    # Extract IUPAC name and clean it
                    iupac_raw = line.split()[0]
                    # Remove Rosetta-specific suffixes
                    self.lig_iupac = iupac_raw.split(':')[0]
                elif line.startswith(tuple(score_names)):
                    for score in score_names:
                        if line.startswith(score):
                            try:
                                scoredata.loc[model_num, score] = float(line.split()[1])
                            except (ValueError, IndexError) as e:
                                print(f'Warning: Could not parse {score} from {file}: {e}')

        self.scoredata = scoredata
        return self

    def to_ensemble(self, outname=None):
        '''
        Write GlycanDock run pose files to a multimodel pdb
        ensemble.
        
        Parameters:
        -----------
        outname : str, optional
            Output filename. If None, creates one from runtag
            
        Returns:
        --------
        str : Path to created ensemble file
        '''
        if not self.pose_files:
            raise ValueError(f'No pose data found in object {self}')
        
        if outname is None:
            outname = f'{self.runtag}_ensemble.pdb' if self.runtag else 'ensemble.pdb'
        
        ensemble_lines = []

        for i, file in enumerate(self.pose_files):
            model_num = i + 1
            with open(file, 'r') as f:
                lines = f.readlines()

                # Write Model header for each pose:
                ensemble_lines.append(f'MODEL     {model_num:4d}\n')

                # Extract only glycoligand data:
                for line in lines:
                    if line.startswith(('ATOM', 'HETATM')) and line[21] == self._lig_chain_id:
                        ensemble_lines.append(line)

                # Designate the end of the current model:
                ensemble_lines.append('ENDMDL\n')

        with open(outname, 'w') as f:
            f.writelines(ensemble_lines)

        self._ensemble_file = os.path.abspath(outname)  # Cache the ensemble file path
        return outname
    
    def scores_to_csv(self, outname=None):
        '''
        Dump score dataframe to a csv file.
        
        Parameters:
        -----------
        outname : str, optional
            Output filename. If None, creates one from runtag
        '''
        if self.scoredata is None:
            raise ValueError(f'No score data found in glycan dock run {self}')
        
        if outname is None:
            outname = f'{self.runtag}_scores.csv' if self.runtag else 'scores.csv'
            
        self.scoredata.to_csv(outname)
        return outname
    
    def cluster_poses(self, rmsd_cutoff=2.0, min_cluster_size=4):
        '''
        Perform a leader-follower clustering algorithm via
        pairwise rmsd and add the cluster id as a column in
        the scoredata dataframe. Only clusters with at least
        min_cluster_size members are considered relevant.
        '''
        import pymol
        from pymol import cmd

        if not self._ensemble_file:
            self.to_ensemble()
        ens_name = os.path.basename(os.path.splitext(self._ensemble_file)[0])
        rmsd_sel = f'{ens_name} and not elem H'
        self.scoredata['cluster_id'] = None

        pymol.finish_launching(['pymol', '-qc'])
        cmd.load(self._ensemble_file, ens_name)
        
        clusters = {}
        leaders = []

        # Leader-follower clustering
        for model_id in self.scoredata.index:
            assigned = False
            for j, leader_id in enumerate(leaders):
                cluster_id = f'cluster_{j+1}'
                rmsd = cmd.rms_cur(rmsd_sel, rmsd_sel,
                                   mobile_state=model_id,
                                   target_state=leader_id)
                if rmsd <= rmsd_cutoff:
                    clusters.setdefault(cluster_id, []).append(model_id)
                    assigned = True
                    break
            if not assigned:
                leaders.append(model_id)
                cluster_id = f'cluster_{len(leaders)}'
                clusters[cluster_id] = [model_id]

        cmd.quit()

        # Filter clusters by min_cluster_size and update scoredata
        relevant_clusters = {}
        cluster_num = 1
        for cluster_id, members in clusters.items():
            if len(members) >= min_cluster_size:
                new_cluster_id = f'cluster_{cluster_num}'
                relevant_clusters[new_cluster_id] = members
                for model_id in members:
                    self.scoredata.loc[model_id, 'cluster_id'] = cluster_num
                cluster_num += 1
            else:
                # For poses in small clusters, leave cluster_id as None
                for model_id in members:
                    self.scoredata.loc[model_id, 'cluster_id'] = None

        self.clusters = relevant_clusters

        return relevant_clusters

    def get_score_values(self, score_name='interaction_energy'):
        '''
        Get all values for a specific score type.
        
        Parameters:
        -----------
        score_name : str
            Name of the score to extract
            
        Returns:
        --------
        pd.Series : Score values indexed by model number
        '''
        if self.scoredata is None:
            raise ValueError('No score data available')
        
        if score_name not in self.scoredata.columns:
            raise ValueError(f'Score {score_name} not found in data')
            
        return pd.to_numeric(self.scoredata[score_name].dropna(), errors='coerce')
    
    def filter_by_score(self, score_name='interaction_energy', threshold=None,
                        top_n=None, negative_is_better=True):
        '''
        Filter models based on score criteria.

        Parameters:
        -----------
        score_name : str
            Name of the score to filter by
        threshold : float, optional
            Only keep models with score <= threshold (or >= if negative_is_better=False)
        top_n : int, optional
            Only keep top N models (lowest or highest scores depending on negative_is_better)
        negative_is_better : bool, optional
            If True (default), lower scores are better; if False, higher scores are better

        Returns:
        --------
        pd.DataFrame : Filtered scoredata
        '''
        if self.scoredata is None:
            raise ValueError('No score data available')

        # Ensure the column is float for correct sorting/comparison
        score_col = pd.to_numeric(self.scoredata[score_name], errors='coerce')
        score_data = score_col.dropna()

        if threshold is not None:
            if negative_is_better:
                mask = score_data <= threshold
            else:
                mask = score_data >= threshold
            filtered_indices = score_data[mask].index
        elif top_n is not None:
            if negative_is_better:
                filtered_indices = score_data.nsmallest(top_n).index
            else:
                filtered_indices = score_data.nlargest(top_n).index
        else:
            filtered_indices = score_data.index

        return self.scoredata.loc[filtered_indices]
    
    def get_ensemble_file(self):
        '''Get the path to the ensemble file, creating it if necessary.'''
        if self._ensemble_file is None or not os.path.exists(self._ensemble_file):
            self._ensemble_file = self.to_ensemble()
        return self._ensemble_file

@dataclass
class VolMap(GlycanDockMetadata):
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
class EnergyMapper(GlycanDockMetadata):
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

'''
Calculate and output a volumetric map of interaction energy
between a receptor and glycoligand fragment from the output
of a Rosetta GlycanDock ensemble.
'''

def main():

    import argparse
    import glob

    parser = argparse.ArgumentParser(
        description='Output a volumetric map of interaction energy from a Rosetta GlycanDocking ensemble.'
    )

    parser.add_argument("--pattern", "-p", required=True, 
                       help="Glob pattern for Rosetta pose PDB files (e.g., '*_complex_*.pdb')")
    parser.add_argument("--runtag", default=None,
                       help="Tag to identify this run (default: fragmap_run)")
    parser.add_argument("-rec", "--recname", default=None,
                       help="Receptor name for metadata")
    parser.add_argument("-lig", "--ligname", default=None,
                       help="Ligand name for metadata")
    parser.add_argument("--voxel-size", type=float, default=1.0,
                       help="Grid spacing in Angstroms (default: 1.0)")
    parser.add_argument("--padding", type=float, default=5.0,
                       help="Buffer distance around ligand ensemble in Angstroms (default: 5.0)")
    parser.add_argument("--outdir", "-o", default=".",
                       help="Output directory (default: current directory)")
    parser.add_argument("--posedir", default=".",
                        help="Directory where the GlycanDock poses are stored to be processed.")
    parser.add_argument("--dx-format", type=int, choices=[1, 3], default=3,
                       help="Values per line in DX files (1 or 3, default: 3)")
    parser.add_argument("--cluster-cutoff", type=float, default=2.0,
                        help="RMSD cutoff for assigning poses to a cluster in Angstroms (defalut: 2.0)")
    parser.add_argument("--min-cluster-size", type=int, default=4,
                        help="Minimum number of clustered poses to be considered significant")
    parser.add_argument("--write-scoredata", action="store_true", default=True,
                       help="Save scores to CSV file")
    parser.add_argument("--write-json", action="store_true",
                       help="Save map metadata to JSON files")
    parser.add_argument("--atom-counts", action="store_true",
                       help="Generate per-atom count maps for heavy atoms")
    
    args = parser.parse_args()

    # Ensure outdir exists:
    os.makedirs(args.outdir, exist_ok=True)

    # Instantiate a GlycanDockEnsemble object:
    gdock_output = GlycanDockEnsemble(
        runtag=args.runtag,
        recname=args.recname,
        ligname=args.ligname
    )

    pattern = os.path.join(os.path.abspath(args.posedir), args.pattern)
    pose_files = sorted(glob.glob(pattern))

    gdock_output.read_poses(pose_files)
    gdock_output.cluster_poses(rmsd_cutoff=args.cluster_cutoff,
                               min_cluster_size=args.min_cluster_size)

    # Generate ensemble file from GlycanDock output:
    ens_file = os.path.join(args.outdir, f'{args.runtag}_ligand_ensemble.pdb')
    gdock_output.to_ensemble(ens_file)

    if args.write_scoredata:
        score_file = os.path.join(args.outdir, f'{args.runtag}_scoredata.csv')
        gdock_output.scores_to_csv(score_file)
    
    # Build energy map from EnergyMapper instance:
    mapper = EnergyMapper(
        glycan_dock_ensemble=gdock_output,
        voxel_size=args.voxel_size,
        padding=args.padding
    )

    volmaps = mapper.map(include_atom_counts=args.atom_counts)

    # Write DX files for each generated volmap:
    for map_name, volmap in volmaps.items():
        dx_file = os.path.join(args.outdir, f'{args.runtag}_{map_name}.dx')
        volmap.to_dx(dx_file, n_cols=args.dx_format)
        print(f'Wrote map file to dx format: {dx_file}')

        # Write json containing metadata if specified:
        if args.write_json:
            json_file = os.path.join(args.outdir, f'{args.runtag}_{map_name}_data.json')
            volmap.to_json(json_file)
            print(f'Wrote json data for map: {json_file}')

if __name__ == '__main__':
    exit(main())