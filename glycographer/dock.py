'''
Dataclass for managing a GlycanDocking run between a receptor and glycoligand
'''

from dataclasses import dataclass, field
from typing import Tuple, Dict, List
import json
import os

from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import cKDTree
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from MDAnalysis.coordinates.memory import MemoryReader
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
    interface_energies: pd.DataFrame = None

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

    # In-memory structural data captured during read_poses. The coordinate
    # arrays are the source of truth for the ensemble structure; the multimodel
    # PDB (_ensemble_file) and the MDAnalysis universe are both derived from
    # them on demand, so pose files are only ever parsed once.
    _lig_coords: np.ndarray = field(default=None, init=False)   # (n_models, n_lig, 3)
    _rec_coords: np.ndarray = field(default=None, init=False)   # (n_models, n_rec, 3)
    _lig_template: List[str] = field(default=None, init=False)  # model-1 ATOM/HETATM lines
    _rec_template: List[str] = field(default=None, init=False)
    _lig_names: np.ndarray = field(default=None, init=False)
    _lig_resids_atom: np.ndarray = field(default=None, init=False)   # per-atom resSeq
    _lig_resnames_atom: np.ndarray = field(default=None, init=False)
    _lig_elements: np.ndarray = field(default=None, init=False)
    _universe: mda.Universe = field(default=None, init=False)

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
                   parse_energies=True, glycan_energies_only=True,
                   store_receptor=False):
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
        glycan_energies_only : bool, optional
            Scope of the parsed REF15 energy table: if True (default), only
            store energy rows for glycan ligand residues; set False to also
            store protein residues (much larger table).
        store_receptor : bool, optional
            Also capture receptor coordinates for every pose so that
            to_pdb(include_receptor=True) can serialize them without
            re-reading pose files (default False; ligand coords are always
            captured). Costs n_models x n_receptor_atoms x 3 floats of RAM.

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
            parse_energies=parse_energies,
            glycan_energies_only=glycan_energies_only,
            store_receptor=store_receptor,
        )
        return ensemble

    @classmethod
    def from_files(cls, run_id=None, ensemble_pdb=None, scores_csv=None,
                   residue_energies_parquet=None,
                   interface_energies_parquet=None, run_type='probe',
                   in_complex_pdb=None, grid_pdb=None, lig_iupac=None):
        '''
        Reconstruct a GlycanDockEnsemble from previously dumped artifacts,
        skipping the (expensive) parse of the raw pose files entirely.

        This is the complement of from_poses: use it once you have written an
        ensemble PDB (to_pdb), a scores CSV (scores_to_csv), and/or the energy
        Parquets (energies_to_parquet / interface_energies_to_parquet), e.g.
        via save_all(). The ensemble PDB is served lazily through `.universe`;
        no coordinates are held in RAM until the universe is first accessed.

        Parameters
        ----------
        run_id : str, optional
            Identifier for the run. Defaults to the ensemble PDB basename.
        ensemble_pdb : str, optional
            Multimodel ligand ensemble PDB (as written by to_pdb). Backs
            `.universe` and the PyMOL-based analyses.
        scores_csv : str, optional
            Per-pose score table (as written by scores_to_csv), indexed by
            model_num.
        residue_energies_parquet, interface_energies_parquet : str, optional
            Long-format energy tables (as written by the *_to_parquet methods).
        run_type, in_complex_pdb, grid_pdb, lig_iupac : optional
            Passthrough metadata.

        Returns
        -------
        GlycanDockEnsemble
        '''
        if run_id is None and ensemble_pdb is not None:
            run_id = os.path.basename(ensemble_pdb).replace('_ensemble.pdb', '') \
                .replace('.pdb', '')

        ens = cls(run_type=run_type, run_id=run_id,
                  in_complex_pdb=in_complex_pdb, grid_pdb=grid_pdb,
                  lig_iupac=lig_iupac)

        if scores_csv is not None:
            ens.scoredata = pd.read_csv(scores_csv, index_col='model_num')
        if residue_energies_parquet is not None:
            ens.residue_energies = pd.read_parquet(residue_energies_parquet)
        if interface_energies_parquet is not None:
            ens.interface_energies = pd.read_parquet(interface_energies_parquet)
        if ensemble_pdb is not None:
            ens._ensemble_file = os.path.abspath(ensemble_pdb)
            if ens._n_poses is None:
                ens._n_poses = len(ens.universe.trajectory)

        return ens

    def save_all(self, out_dir='.', include_receptor=False):
        '''
        Dump every populated artifact to `out_dir` under run_id-derived names,
        so the ensemble can later be reloaded with from_files() without
        re-parsing poses. Skips artifacts that have not been populated.

        Returns
        -------
        dict : {artifact_name: path} for everything written.
        '''
        os.makedirs(out_dir, exist_ok=True)
        written = {}

        def _p(name):
            return os.path.join(out_dir, name)

        if self._lig_coords is not None or self._ensemble_file:
            written['ensemble_pdb'] = self.to_pdb(
                outname=_p(f'{self.run_id}_ensemble.pdb'),
                include_receptor=include_receptor)
        if self.scoredata is not None:
            written['scores_csv'] = self.scores_to_csv(
                _p(f'{self.run_id}_scores.csv'))
        if self.residue_energies is not None:
            written['residue_energies_parquet'] = self.energies_to_parquet(
                _p(f'{self.run_id}_residue_energies.parquet'))
        if self.interface_energies is not None:
            written['interface_energies_parquet'] = \
                self.interface_energies_to_parquet(
                    _p(f'{self.run_id}_interface_energies.parquet'))

        return written

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
                   parse_energies=True, glycan_energies_only=True,
                   store_receptor=False):
        '''
        Import object data from a collection of Rosetta output poses.

        Populates the per-pose `scoredata` DataFrame and, when
        parse_energies is True, the long-format `residue_energies`
        DataFrame (one row per pose / residue / energy term). In the same
        single pass it also captures the ligand (and, if store_receptor,
        receptor) heavy+hydrogen coordinates into in-memory arrays, which
        become the source of truth for both the `.universe` and `to_pdb()`
        -- so pose files are only ever read once.

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
        glycan_energies_only : bool, optional
            Scope of the parsed REF15 energy table: if True (default), only
            store energy rows for glycan ligand residues. Set False to also
            store protein residues (much larger table). Does not affect the
            captured structure -- ligand coordinates are always stored.
        store_receptor : bool, optional
            Also capture receptor (chain _rec_chain_id) coordinates per pose
            so to_pdb(include_receptor=True) can serialize them from memory
            (default False).
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

        # Per-model coordinate accumulators for the in-memory structure. Atom
        # identity (names/resids/elements/order) is constant across poses, so
        # the topology is captured once from model 1; only coordinates vary.
        lig_coords_per_model = []
        rec_coords_per_model = []

        # Parse each pose file to extract data:
        for i, file in enumerate(self.pose_files):
            model_num = i + 1
            scoredata.loc[model_num, 'pose_file'] = os.path.abspath(file)

            with open(file, 'r') as f:
                lines = f.readlines()

            glycan_labels = []

            # Capture ligand (and optionally receptor) atom records for the
            # structural ensemble, in file order.
            lig_recs = [ln for ln in lines
                        if ln.startswith(('ATOM', 'HETATM'))
                        and ln[21] == self._lig_chain_id]
            if not lig_recs:
                raise ValueError(
                    f'No chain-{self._lig_chain_id} (ligand) atoms found in '
                    f'{file}. Check _lig_chain_id.')
            if model_num == 1:
                self._capture_ligand_topology(lig_recs)
            self._check_atom_count(lig_recs, self._lig_template, file, 'ligand')
            lig_coords_per_model.append(self._coords_from_records(lig_recs))

            if store_receptor:
                rec_recs = [ln for ln in lines
                            if ln.startswith(('ATOM', 'HETATM'))
                            and ln[21] == self._rec_chain_id]
                if model_num == 1:
                    self._rec_template = list(rec_recs)
                self._check_atom_count(rec_recs, self._rec_template, file,
                                       'receptor')
                rec_coords_per_model.append(
                    self._coords_from_records(rec_recs))

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
                self._parse_energy_table(lines, model_num, glycan_energies_only,
                                         cols, source_file=file)

        self.scoredata = scoredata

        # Stack per-model coordinates into (n_models, n_atoms, 3) arrays.
        self._lig_coords = np.asarray(lig_coords_per_model, dtype=np.float32)
        if store_receptor:
            self._rec_coords = np.asarray(rec_coords_per_model, dtype=np.float32)
        # Invalidate any previously built universe now that coords changed.
        self._universe = None

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

    # ---- in-memory structure: capture, universe, serialization ----------
    @staticmethod
    def _coords_from_records(records):
        '''Parse the (N, 3) xyz array out of a list of PDB ATOM/HETATM lines.'''
        return np.array(
            [[float(r[30:38]), float(r[38:46]), float(r[46:54])]
             for r in records], dtype=np.float32)

    @staticmethod
    def _check_atom_count(records, template, source_file, which):
        '''Guard that every pose has the same atom count/order as model 1.'''
        if template is not None and len(records) != len(template):
            raise ValueError(
                f'{which} atom count in {source_file} ({len(records)}) differs '
                f'from model 1 ({len(template)}); poses must share topology.')

    def _capture_ligand_topology(self, records):
        '''
        Record the ligand's per-atom identity from the first pose. These are
        constant across poses, so they are stored once and reused to build the
        universe and to re-emit PDB lines in to_pdb().
        '''
        self._lig_template = list(records)
        self._lig_names = np.array([r[12:16].strip() for r in records])
        self._lig_resids_atom = np.array([int(r[22:26]) for r in records],
                                          dtype=np.int64)
        self._lig_resnames_atom = np.array([r[17:20].strip() for r in records])
        self._lig_elements = np.array([r[76:78].strip() for r in records])

    def _build_universe(self):
        '''
        Build a ligand-only MDAnalysis Universe from the captured coordinate
        arrays via MemoryReader -- no PDB file is written. Frame f (0-based)
        corresponds to model_num f + 1, matching scoredata's index, and the
        atom selection / resid semantics match the old file-based universe so
        the mapper is unaffected.
        '''
        if self._lig_coords is None:
            raise ValueError(
                'No in-memory ligand coordinates; call read_poses/from_poses '
                'first, or use from_files() with an ensemble PDB.')

        resids_atom = self._lig_resids_atom
        # Residues in first-appearance (file) order.
        ridx_of = {}
        atom_resindex = np.empty(len(resids_atom), dtype=np.int64)
        unique_resids = []
        resnames = []
        for a, rid in enumerate(resids_atom):
            if rid not in ridx_of:
                ridx_of[rid] = len(unique_resids)
                unique_resids.append(rid)
                resnames.append(self._lig_resnames_atom[a])
            atom_resindex[a] = ridx_of[rid]

        n_atoms = self._lig_coords.shape[1]
        u = mda.Universe.empty(
            n_atoms, n_residues=len(unique_resids),
            atom_resindex=atom_resindex, trajectory=True)
        u.add_TopologyAttr('names', self._lig_names)
        u.add_TopologyAttr('types', self._lig_elements)
        u.add_TopologyAttr('resids', np.asarray(unique_resids, dtype=np.int64))
        u.add_TopologyAttr('resnames', np.asarray(resnames))
        u.add_TopologyAttr('segids', [self._lig_chain_id])
        u.load_new(self._lig_coords, format=MemoryReader)
        return u

    @property
    def universe(self):
        '''
        Lazily built, cached MDAnalysis Universe of the ligand ensemble.

        Prefers the in-memory coordinate arrays captured by read_poses; falls
        back to reading a cached/loaded multimodel PDB (_ensemble_file), which
        is how a from_files()-constructed ensemble is served. This is the
        object a Mapper should consume -- no to_pdb() call is required.
        '''
        if self._universe is None:
            if self._lig_coords is not None:
                self._universe = self._build_universe()
            elif self._ensemble_file and os.path.exists(self._ensemble_file):
                self._universe = mda.Universe(self._ensemble_file)
            else:
                raise ValueError(
                    'No structure available to build a universe: read_poses '
                    'has not run and no ensemble PDB is set. Use from_poses() '
                    'or from_files(ensemble_pdb=...).')
        return self._universe

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

        Notes
        -----
        This serializes the coordinate arrays captured by read_poses -- pose
        files are NOT re-read. Each model's line is the model-1 template with
        fresh coordinates spliced into columns 31-54, so occupancy, B-factor,
        element, and atom naming are preserved exactly. include_receptor=True
        requires read_poses(store_receptor=True); otherwise it falls back to
        re-reading the original pose files.
        '''
        if outname is None:
            outname = f'{self.run_id}_ensemble.pdb'

        # include_receptor without captured receptor coords -> legacy fallback.
        if include_receptor and self._rec_coords is None:
            return self._to_pdb_from_files(outname, include_receptor=True)

        if self._lig_coords is None:
            raise ValueError(
                f'No structural data in {self}; run read_poses/from_poses '
                f'(or use from_files with an ensemble PDB).')

        ensemble_lines = []
        for m in range(self._lig_coords.shape[0]):
            ensemble_lines.append(f'MODEL     {m + 1:4d}\n')
            if include_receptor:
                self._emit_model_lines(ensemble_lines, self._rec_template,
                                       self._rec_coords[m])
            self._emit_model_lines(ensemble_lines, self._lig_template,
                                   self._lig_coords[m])
            ensemble_lines.append('ENDMDL\n')

        with open(outname, 'w') as f:
            f.writelines(ensemble_lines)

        self._ensemble_file = os.path.abspath(outname)  # Cache the path
        return outname

    @staticmethod
    def _emit_model_lines(out, template, coords):
        '''
        Append one model's PDB lines to `out` by splicing `coords` (N, 3) into
        the fixed coordinate columns (31-54) of each template line.
        '''
        for line, (x, y, z) in zip(template, coords):
            out.append(f'{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}')

    def _to_pdb_from_files(self, outname, include_receptor=False):
        '''
        Legacy path: build the multimodel ensemble by re-reading the original
        pose files. Used only when receptor coordinates were not captured in
        memory but include_receptor=True is requested.
        '''
        if not self.pose_files:
            raise ValueError(
                f'Cannot write receptor ensemble for {self}: no captured '
                f'receptor coords and no pose_files to re-read. Re-run '
                f'read_poses with store_receptor=True.')

        ensemble_lines = []
        for i, file in enumerate(self.pose_files):
            ensemble_lines.append(f'MODEL     {i + 1:4d}\n')
            with open(file, 'r') as f:
                for line in f:
                    if not line.startswith(('ATOM', 'HETATM')):
                        continue
                    chain = line[21]
                    if chain == self._lig_chain_id or (
                            include_receptor and chain == self._rec_chain_id):
                        ensemble_lines.append(line)
            ensemble_lines.append('ENDMDL\n')

        with open(outname, 'w') as f:
            f.writelines(ensemble_lines)

        self._ensemble_file = os.path.abspath(outname)
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

    def extract_interface_energies(self, models=None, score_function=None,
                                   init_flags='-include_sugars -mute all',
                                   store=True, verbose=True):
        '''
        Compute per-term interaction energies between each glycan ligand
        residue and the protein residues it contacts, by re-scoring poses in
        PyRosetta and decomposing the two-body energies per residue pair.

        REQUIRES PyRosetta -- run this on the Ubuntu partition / HPC. The
        import is local so dock.py still imports on systems without it.

        Why this and not the POSE_ENERGIES_TABLE: that table stores each
        residue's *total* energy with every two-body term summed and split
        50/50 between partners, so the residue->residue pairing is lost and
        cannot be recovered. ScoreFunction.residue_pair_energy recomputes the
        full two-body energy for one (glycan, protein) pair -- including
        hydrogen bonds -- which is exactly the Newhouse-style decomposition.

        Re-scoring is the expensive step, so pass a filtered/clustered subset
        of `models` (e.g. from filter_by_score or a dominant cluster).

        Parameters
        ----------
        models : list of int, optional
            scoredata index (model_num) values to process. Defaults to all.
        score_function : pyrosetta ScoreFunction, optional
            Defaults to ref2015 (the GlycanDock scoring function). Pass the
            exact score function used for docking for full consistency.
        init_flags : str
            Flags for pyrosetta.init() if PyRosetta is not already
            initialized. Must include sugar support to read glycan PDBs.
        store : bool
            Store the result on self.interface_energies.

        Returns
        -------
        pd.DataFrame (long format) with columns:
            model_num, glycan_resnum, glycan_label,
            protein_resnum, protein_label, term, weighted
        '''
        import pyrosetta
        from pyrosetta import pose_from_pdb
        from pyrosetta.rosetta.core.scoring import (
            EMapVector, name_from_score_type)

        # Initialize PyRosetta only if needed (defensive about API name):
        try:
            initialized = pyrosetta.rosetta.basic.was_init_called()
        except Exception:
            initialized = False
        if not initialized:
            pyrosetta.init(init_flags)

        sfxn = score_function or pyrosetta.get_fa_scorefxn()
        weights = sfxn.weights()
        nonzero_terms = list(sfxn.get_nonzero_weighted_scoretypes())

        if self.scoredata is None:
            raise ValueError('No scoredata; run read_poses/from_poses first.')
        if models is None:
            models = list(self.scoredata.index)
        file_for = dict(zip(self.scoredata.index, self.scoredata['pose_file']))

        # Reuse the IUPAC labels already parsed from the energies table:
        gly_label_map = {}
        if self.residue_energies is not None:
            gsub = self.residue_energies[self.residue_energies.is_glycan]
            gly_label_map = dict(zip(gsub.residue_num.astype(int),
                                     gsub.residue_label.astype(str)))

        cols = {k: [] for k in ('model_num', 'glycan_resnum', 'glycan_label',
                                 'protein_resnum', 'protein_label', 'term',
                                 'weighted')}

        for m in models:
            pose = pose_from_pdb(file_for[m])
            sfxn(pose)
            energy_graph = pose.energies().energy_graph()
            pdb_info = pose.pdb_info()
            n = pose.total_residue()

            glycan_res = [i for i in range(1, n + 1)
                          if pose.residue(i).is_carbohydrate()]

            for i in glycan_res:
                res_i = pose.residue(i)
                gly_label = gly_label_map.get(i, f'glycan_{i}')
                for j in range(1, n + 1):
                    res_j = pose.residue(j)
                    if res_j.is_carbohydrate() or not res_j.is_protein():
                        continue
                    # Only pairs with a scoring edge can have nonzero
                    # two-body energy -- this is the natural contact filter.
                    if energy_graph.find_energy_edge(i, j) is None:
                        continue
                    emap = EMapVector()
                    sfxn.residue_pair_energy(res_i, res_j, pose, emap)

                    if pdb_info is not None:
                        prot_label = f'{res_j.name3()}{pdb_info.number(j)}'
                    else:
                        prot_label = f'{res_j.name3()}{j}'

                    for st in nonzero_terms:
                        val = emap[st] * weights[st]
                        if val == 0.0:
                            continue
                        cols['model_num'].append(m)
                        cols['glycan_resnum'].append(i)
                        cols['glycan_label'].append(gly_label)
                        cols['protein_resnum'].append(j)
                        cols['protein_label'].append(prot_label)
                        cols['term'].append(name_from_score_type(st))
                        cols['weighted'].append(val)

            if verbose:
                print(f'Decomposed interface energies for model {m}')

        interface_energies = pd.DataFrame({
            'model_num': np.asarray(cols['model_num'], dtype=np.int32),
            'glycan_resnum': np.asarray(cols['glycan_resnum'], dtype=np.int32),
            'glycan_label': pd.Categorical(cols['glycan_label']),
            'protein_resnum': np.asarray(cols['protein_resnum'], dtype=np.int32),
            'protein_label': pd.Categorical(cols['protein_label']),
            'term': pd.Categorical(cols['term']),
            'weighted': np.asarray(cols['weighted'], dtype=np.float32),
        })

        if store:
            self.interface_energies = interface_energies
        return interface_energies

    def interface_energies_to_parquet(self, outname=None):
        '''
        Dump the long-format glycan-protein interface energy table to Parquet.
        Requires a Parquet engine (pyarrow or fastparquet).
        '''
        if self.interface_energies is None:
            raise ValueError(
                f'No interface energy data found in {self}; run '
                f'extract_interface_energies() first (needs PyRosetta).')
        if outname is None:
            outname = f'{self.run_id}_interface_energies.parquet'
        self.interface_energies.to_parquet(outname, index=False)
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

    def scale_inteng(self, method='minmax'):
        '''
        Scale the interaction energy values of all poses of the ensemble
        by a specific metric (Minmax scale by default where 0 is the worst
        score and 1 is the best)
        '''
        vals = self.scoredata['interaction_energy'].to_numpy()
        vals = np.clip(vals, a_min=np.min(vals), a_max=0)
        vals = -vals

        if method == 'minmax':
            scaler = MinMaxScaler()
        
        scaled_vals = scaler.fit_transform(vals.reshape([-1, 1]))
        self.scoredata['scaled_interaction_energy'] = scaled_vals

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
    
    def get_ensemble_file(self, include_receptor=False):
        '''
        Return a path to the multimodel ensemble PDB, writing it on demand.

        Downstream tools that need a file on disk (the PyMOL-based
        cluster_poses / rmsd_from_reference) call this. If a file was already
        written or loaded it is reused; otherwise to_pdb() materializes one
        from the in-memory coordinates. Mapping does NOT need this -- use the
        in-memory `.universe` instead.
        '''
        if self._ensemble_file and os.path.exists(self._ensemble_file):
            return self._ensemble_file
        if self._lig_coords is not None:
            return self.to_pdb(include_receptor=include_receptor)
        print('No ensemble file has been generated and no in-memory '
              'coordinates are available to create one.')
        return self._ensemble_file