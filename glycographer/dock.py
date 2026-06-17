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
    residue_energies: pd.DataFrame = None

    # Internal attributes:
    _rec_chain_id: str = field(default='A', init=False)
    _lig_chain_id: str = field(default='X', init=False)
    weights: pd.Series = field(default=None, init=False)
    energy_terms: List[str] = field(default=None, init=False)
    _lig_residues: List[str] = field(default=None, init=False)
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

    @classmethod
    def from_poses(cls, pose_list, in_complex_pdb=None, grid_pdb=None,
                   run_type='probe', run_id=None, score_names=None,
                   parse_energies=True, glycan_only=True):
        '''
        Instantiate a GlycanDockEnsemble directly from a collection of
        GlycanDock output pose files.

        This is the preferred entry point: it constructs the object and
        runs read_poses() in one step.

        Parameters
        ----------
        pose_list : list of str
            Paths to GlycanDock output PDB files.
        in_complex_pdb, grid_pdb : str, optional
            Paths to the input complex / grid PDBs, used to build run_id.
        run_type, run_id : str, optional
            Run metadata. run_id is auto-generated if not provided.
        score_names : list of str, optional
            Per-pose GlycanDock score labels to extract.
        parse_energies : bool, optional
            Parse the per-residue REF15 energy table into residue_energies.
        glycan_only : bool, optional
            Only store glycan ligand residue energies (default True).

        Returns
        -------
        GlycanDockEnsemble
        '''
        ensemble = cls(
            run_type=run_type, run_id=run_id,
            in_complex_pdb=in_complex_pdb, grid_pdb=grid_pdb,
        )
        ensemble.read_poses(
            pose_list, score_names=score_names,
            parse_energies=parse_energies, glycan_only=glycan_only,
        )
        return ensemble

    @staticmethod
    def _to_float(token):
        '''Convert a score-table token to float, mapping 'NA' -> NaN.'''
        try:
            return float(token)
        except (ValueError, TypeError):
            return np.nan

    @staticmethod
    def _split_residue_label(token):
        '''
        Split a pose-energies residue label into (label, residue_number).

        Residue labels can themselves contain underscores and colons
        (e.g. '->8)-alpha-Neup:reducing_end:5-Ac_128'), so the residue
        number is split off the *last* underscore only.
        '''
        label, _, num = token.rpartition('_')
        if num.isdigit():
            return label, int(num)
        return token, None

    def _register_weights(self, terms, weight_tokens, source_file=''):
        '''
        Store the REF15 term weights from the first pose seen, and verify
        that every subsequent pose carries the same weights.
        '''
        if terms is None or len(terms) != len(weight_tokens):
            return
        w = pd.Series([self._to_float(x) for x in weight_tokens],
                      index=terms, dtype='float64', name='weight')
        if self.weights is None:
            self.weights = w
            self.energy_terms = list(terms)
        elif (not self.weights.index.equals(w.index)
              or not np.allclose(self.weights.values, w.values,
                                 equal_nan=True)):
            print(f'Warning: REF15 weights in {source_file} differ from the '
                  f'rest of the ensemble; keeping the first set seen. This '
                  f'usually means poses were scored with different score '
                  f'functions.')

    def _parse_energy_table(self, lines, model_num, glycan_only,
                            cols, source_file=''):
        '''
        Parse the #...POSE_ENERGIES_TABLE block of a single pose and append
        per-residue, per-term rows into the columnar accumulator `cols`.
        '''
        begin = end = None
        for idx, line in enumerate(lines):
            if line.startswith('#BEGIN_POSE_ENERGIES_TABLE'):
                begin = idx
            elif line.startswith('#END_POSE_ENERGIES_TABLE'):
                end = idx
                break
        if begin is None or end is None:
            print(f'Warning: no pose energies table found in {source_file}')
            return

        terms = None
        for line in lines[begin + 1:end]:
            toks = line.split()
            if not toks:
                continue
            head = toks[0]
            if head == 'label':
                terms = toks[1:]
            elif head == 'weights':
                self._register_weights(terms, toks[1:], source_file)
            elif head == 'pose':
                continue  # whole-complex totals, not a residue
            else:
                if terms is None:
                    continue
                label, resnum = self._split_residue_label(head)
                is_glycan = head.startswith('->')
                if glycan_only and not is_glycan:
                    continue
                vals = toks[1:]
                if len(vals) != len(terms):
                    print(f'Warning: term/value count mismatch for {head} '
                          f'in {source_file}; skipping residue.')
                    continue
                for term, v in zip(terms, vals):
                    cols['model_num'].append(model_num)
                    cols['residue_num'].append(resnum if resnum is not None else -1)
                    cols['residue_label'].append(label)
                    cols['is_glycan'].append(is_glycan)
                    cols['term'].append(term)
                    cols['weighted'].append(self._to_float(v))

    def read_poses(self, pose_list, score_names=None,
                   parse_energies=True, glycan_only=True):
        '''
        Import object data from a collection of Rosetta output poses.

        Populates the per-pose `scoredata` DataFrame and, when
        parse_energies is True, the long-format `residue_energies`
        DataFrame (one row per pose / residue / energy term).

        Parameters
        ----------
        pose_list : list of str
            Paths to GlycanDock output PDB files.
        score_names : list of str, optional
            Per-pose GlycanDock score labels to extract. Defaults to
            _SCORE_FEATURES.
        parse_energies : bool, optional
            Parse the per-residue REF15 energy table at the bottom of each
            pose (default True). The table is already in memory from reading
            the file, so this adds negligible time.
        glycan_only : bool, optional
            If True (default), only store energy rows for glycan ligand
            residues. Set False to also store protein residues (much larger).
        '''
        # Create a list of each pose file:
        if not self.pose_files:
            pose_files = sorted(pose_list)
            self.pose_files = pose_files
            self._pose_dir = os.path.dirname(pose_files[0])

        if not self._n_poses:
            self._n_poses = len(self.pose_files)

        if not self.run_id:
            if self.in_complex_pdb and self.grid_pdb:
                self.run_id = (f"{self.in_complex_pdb.replace('.pdb', '')}_"
                               f"{self.grid_pdb.replace('.pdb', '')}_"
                               f"{self._n_poses}p")
            else:
                self.run_id = f'glycandock_{self._n_poses}p'

        # Use provided score names or default to standard ones:
        if score_names is None:
            score_names = self._SCORE_FEATURES.copy()
        score_set = set(score_names)

        # Create a pandas DataFrame for storing score data for each pose:
        # Use model numbers as index for easier lookup
        model_indices = list(range(1, self._n_poses + 1))
        scoredata = pd.DataFrame(index=model_indices, columns=score_names)
        scoredata.index.name = 'model_num'

        # Add a column to track the original pose file for each model
        scoredata['pose_file'] = ''

        # Columnar accumulators for the long-format residue energy table:
        cols = {k: [] for k in ('model_num', 'residue_num', 'residue_label',
                                 'is_glycan', 'term', 'weighted')}

        # Parse each pose file to extract data:
        for i, file in enumerate(self.pose_files):
            model_num = i + 1
            scoredata.loc[model_num, 'pose_file'] = os.path.abspath(file)

            with open(file, 'r') as f:
                lines = f.readlines()

            glycan_labels = []

            # Extract per-pose GlycanDock scores and glycan residue labels:
            for line in lines:
                if line.startswith('->'):
                    if model_num == 1:
                        label, _ = self._split_residue_label(line.split()[0])
                        glycan_labels.append(label)
                else:
                    toks = line.split()
                    # Match the score label exactly to avoid prefix
                    # collisions (e.g. 'Fnat' vs 'Fnat_intf_residues').
                    if toks and toks[0] in score_set:
                        try:
                            scoredata.loc[model_num, toks[0]] = float(toks[1])
                        except (ValueError, IndexError) as e:
                            print(f'Warning: Could not parse {toks[0]} from '
                                  f'{file}: {e}')

            # Ligand identity is constant across the run; capture it once.
            if model_num == 1 and glycan_labels:
                self._lig_residues = glycan_labels
                if not self.lig_iupac:
                    self.lig_iupac = ','.join(glycan_labels)

            # Parse the per-residue REF15 energy table:
            if parse_energies:
                self._parse_energy_table(lines, model_num, glycan_only,
                                         cols, source_file=file)

        self.scoredata = scoredata

        # Build the long-format residue energy table with compact dtypes:
        if parse_energies and cols['model_num']:
            self.residue_energies = pd.DataFrame({
                'model_num': np.asarray(cols['model_num'], dtype=np.int32),
                'residue_num': np.asarray(cols['residue_num'], dtype=np.int32),
                'residue_label': pd.Categorical(cols['residue_label']),
                'is_glycan': np.asarray(cols['is_glycan'], dtype=bool),
                'term': pd.Categorical(cols['term']),
                'weighted': np.asarray(cols['weighted'], dtype=np.float32),
            })

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

    def energies_to_parquet(self, outname=None):
        '''
        Dump the long-format per-residue REF15 energy table to a Parquet
        file. Parquet is columnar and compresses this repetitive long table
        far better than CSV, and round-trips the categorical/float32 dtypes.

        Requires a Parquet engine (pyarrow or fastparquet) to be installed.

        Parameters
        ----------
        outname : str, optional
            Output filename. Defaults to '<run_id>_residue_energies.parquet'.
        '''
        if self.residue_energies is None:
            raise ValueError(
                f'No residue energy data found in {self}; run read_poses '
                f'(or from_poses) with parse_energies=True first.')

        if outname is None:
            outname = f'{self.run_id}_residue_energies.parquet'

        self.residue_energies.to_parquet(outname, index=False)
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