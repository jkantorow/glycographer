'''
Dataclass for managing a GlycanDocking run between a receptor and glycoligand
'''

from dataclasses import dataclass, field
from typing import Tuple, Dict, List
import json
import os

from scipy.spatial import cKDTree
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import pandas as pd

import pymol
from pymol import cmd

#from pyrosetta import init, pose_from_pdb, Vector1
#from pyrosetta.rosetta.protocols.rigid import RigidBodyRandomizeMover, partner_downstream
#from pyrosetta.rosetta.protocols.docking import setup_foldtree
#from pyrosetta.rosetta.protocols.ligand_docking import StartFrom
#from pyrosetta.rosetta.protocols.glycan_docking import GlycanDockProtocol

@dataclass
class GlycanDockEnsemble:
    '''
    Object for storing and accessing data describing
    the output of a GlycanDock sampling simulation.
    '''
    run_type: str = field(default='probe')
    run_id: str = field(default=None)
    in_complex_pdb: str = None
    pose_files: List[str] = None
    grid_pdb: str = None
    lig_iupac: str = None
    scoredata: pd.DataFrame = None

    # Internal attributes:
    _rec_chain_id: str = field(default='A', init=False)
    _lig_chain_id: str = field(default='X', init=False)
    _n_poses: int = field(default=None, init=False)
    _complex_path: os.PathLike = field(default=None, init=False)
    _grid_path: os.PathLike = field(default=None, init=False)
    _pose_dir: os.PathLike = field(default=None, init=False)
    _ensemble_path: os.PathLike = field(default=None, init=False)
    _ensemble_file: str = field(default=None, init=False)
    _clusters: Dict = field(default=None, init=False)

    def __post_init__(self):
        '''Initialize derived attributes after dataclass initialization.'''
        if self.in_complex_pdb:
            self._complex_path = os.path.abspath(self.in_complex_pdb)
        if self.grid_pdb:
            self._grid_path = os.path.abspath(self.grid_pdb)

    # Class-level constant for standard GlycanDock score names
    _SCORE_FEATURES = [
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

        if not self.run_id:
            self.run_id = f"{self.in_complex_pdb.replace('.pdb', '')}_{self.grid_pdb.replace('.pdb', '')}_{self._n_poses}p"

        # Use provided score names or default to standard ones:
        if score_names is None:
            score_names = self._SCORE_FEATURES.copy()
        
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
            scoredata.loc[model_num, 'pose_file'] = os.path.abspath(file)
            
            with open(file, 'r') as f:
                lines = f.readlines()

            parsed_sugar_residue_data = []

            # Extract scores and metadata from pose file:
            for line in lines:
                if line.startswith('->'):
                    parsed_sugar_residue_data.append(line.split()[0])
                elif line.startswith(tuple(score_names)):
                    for score in score_names:
                        if line.startswith(score):
                            try:
                                scoredata.loc[model_num, score] = float(line.split()[1])
                            except (ValueError, IndexError) as e:
                                print(f'Warning: Could not parse {score} from {file}: {e}')

            # I need a better way to scrape the iupac and sugar residue data:
            self._lig_residues = parsed_sugar_residue_data
            if not self.lig_iupac:
                self.lig_iupac = ','.join(parsed_sugar_residue_data)

        self.scoredata = scoredata
        return self

    def to_pdb(self, outname=None, include_receptor=False):
        '''
        Write GlycanDock run pose files to a multimodel pdb
        ensemble.
        
        Parameters:
        -----------
        outname : str, optional
            Output filename. If None, creates one from runtag

        include_receptor : bool, optional
            Output receptor coordinates along with ligand
            coordinates to ensemble pdb
            
        Returns:
        --------
        str : Path to created ensemble file
        '''
        if not self.pose_files:
            raise ValueError(f'No pose data found in object {self}')
        
        if outname is None:
            outname = f'{self.run_id}_ensemble.pdb'
        
        ensemble_lines = []

        for i, file in enumerate(self.pose_files):
            model_num = i + 1
            with open(file, 'r') as f:
                lines = f.readlines()

                # Write Model header for each pose:
                ensemble_lines.append(f'MODEL     {model_num:4d}\n')

                # Extract only glycoligand data:
                for line in lines:
                    if include_receptor:
                        if line.startswith(('ATOM', 'HETATM')) and line[21] == self._rec_chain_id:
                            ensemble_lines.append(line)
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
            outname = f'{self.run_id}_scores.csv'
            
        self.scoredata.to_csv(outname)
        return outname
    
    def cluster_poses(self, rmsd_cutoff=2.0, min_cluster_size=4):
        '''
        Perform a leader-follower clustering algorithm via
        pairwise rmsd and add the cluster id as a column in
        the scoredata dataframe. Only clusters with at least
        min_cluster_size members are considered relevant.
        '''
        if not self._ensemble_file:
            raise ValueError('Cannot cluster poses until ensemble file is created.')
        ens_name = os.path.basename(self._ensemble_file.replace('.pdb', ''))
        rmsd_sel = f'{ens_name} and not elem H'
        self.scoredata['cluster_id'] = None

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

        self._clusters = relevant_clusters

        return self

    def rmsd_from_reference(self, reference_pdb: str, use_mda=False):
        '''
        Calculate the in-place RMSD of each pose to a known reference
        pose.
        '''
        if use_mda:
            u = mda.Universe(self._ensemble_file)
            ref = mda.Universe(reference_pdb)

            R = rms.RMSD(u, ref, select='not name *H*')
            R.run()

            self._rmsd_results = R.results.rmsd
            self.scoredata['ref_rmsd'] = self._rmsd_results[2]

        else:
            cmd.load(self._ensemble_file, 'ensemble')
            cmd.load(reference_pdb, 'reference')

            cmd.select('ensemble_heavy', 'ensemble and not elem H')
            cmd.select('reference_heavy', 'reference and not elem H')

            rmsd_vals = []
            for i in range(cmd.count_states('ensemble')):
                rmsd = cmd.rms_cur('ensemble_heavy', 'reference_heavy',
                                   mobile_state=i+1,
                                   target_state=0)
                rmsd_vals.append(rmsd)

            self._rmsd_results = rmsd_vals
            self.scoredata['ref_rmsd'] = self._rmsd_results

        return self

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
            print('No ensemble file has been generated.')
        return self._ensemble_file