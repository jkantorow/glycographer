'''
Sampling grids: the set of starting coordinates GlycanDock places a probe at.

A grid is a point cloud sitting in a thin shell above the receptor surface,
written as a PDB of `PSD` pseudo-atoms on chain P. sample.GlycanDockRunner
hands that file to Rosetta's StartFrom mover, which moves the glycoligand's
center of mass onto one of the points before each decoy is docked -- so a
grid point is a candidate *ligand center*, not a candidate contact.

    from glycographer.grid import (load_receptor, SamplingRegion,
                                   build_sampling_grid, write_grid)

    rec = load_receptor('1pmh_rec.pdb')
    region = SamplingRegion.around(rec.site_coords('chain A and resi 110-130'),
                                   radius=12.0)
    grid = build_sampling_grid(rec, region=region)
    write_grid(grid, '1pmh_grid_cbm.pdb')

Geometry
--------
Every distance is measured from the *solvent-excluded* (molecular) surface of
the atoms in `surface_selection`, computed directly from atom coordinates and
van der Waals radii. There is no surface mesh and no STL file: the old
pipeline exported a PyMOL surface to STL and raycast it with open3d, which
needed incentive PyMOL (open-source PyMOL refuses `save *.stl`) and an open3d
build that will not solve on Windows.

Two radii define the geometry, and they do different jobs:

  probe_radius (default 1.4 A, water)
      Rolls over the atoms to define the surface the shell is measured from.
      It fills in crevices narrower than a water molecule, exactly as a
      Connolly surface does.

  fit_radius (default 2.5 A, roughly a pyranose ring)
      Rolls over the atoms to define which space a sugar could actually
      occupy. Points that only a sphere smaller than this can reach -- the
      bottom of a narrow groove, a sealed interior cavity -- are dropped.
      This is the "cryptic pocket" filter, done as one morphological pass
      rather than per-point raycasting, and it subsumes the interior-cavity
      case for free (a cavity is a component of free space that does not
      reach bulk solvent).

Both are the same computation at two radii, on one lattice:

  1. d_vdw(x) = min_i(|x - c_i| - r_i), exact, from one KD-tree per distinct
     radius (nearest *center* is not nearest *surface* when radii differ).
  2. The probe centers that fit are d_vdw >= r. Label them; the components
     that reach the edge of the lattice are bulk solvent, the rest are sealed
     cavities.
  3. Solvent is everything within r of a bulk probe center (one Euclidean
     distance transform); everything else is receptor interior.
  4. Distance above the surface is the distance transform of the interior,
     clipped by d_vdw -- which is exact and is the answer wherever the
     surface is convex, i.e. almost everywhere. The lattice only matters
     inside crevices, and only ever makes them look slightly more open.

Lattice discretization therefore costs accuracy only in concavities, and in
the direction of keeping points rather than dropping them.

Where to sample
---------------
A SamplingRegion restricts where points may be placed:

    SamplingRegion()                                 whole receptor surface
    SamplingRegion.from_box(bb_min, bb_max)          an explicit box
    SamplingRegion.around(coords, radius)            within R of some atoms

`around` is the usual one: pass the coordinates of a PyMOL selection
(`Receptor.site_coords`) and every point within `radius` of any selected atom
is kept. The kept volume is a union of spheres, not a box, so two selections
far apart give two disjoint patches of grid with nothing sampled in between.

PyMOL is used only to resolve selections and read coordinates; it is imported
lazily, so the rest of this module works from a plain coordinate array.
'''

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from glycographer import naming

# --- defaults ---------------------------------------------------------------

# Everything that is physically in the way of a glycoligand. Solvent is not
# (it is stripped before docking); hydrogens are not, because receptors
# prepared by scripts/utils/prepare_complexes.py have none and a grid must not
# shift depending on whether the input happened to be protonated.
DEFAULT_SURFACE_SELECTION = 'not (solvent or hydro)'

DEFAULT_SPACING = 2.0           # A between sampling points
DEFAULT_SHELL = (1.0, 4.0)      # A above the molecular surface
DEFAULT_PROBE_RADIUS = 1.4      # A, water
DEFAULT_FIT_RADIUS = 2.5        # A, smallest sphere a sugar ring needs
DEFAULT_SITE_RADIUS = 10.0      # A around a --site selection
MAX_LATTICE_SPACING = 1.0       # A; the geometry lattice is never coarser

# A lattice this big already costs ~2 GB across the intermediate arrays.
MAX_VOXELS = 40_000_000

# van der Waals radii (Bondi 1964; Mantina 2009 for the halogens).
#
# Metal ions use Shannon ionic radii (CN 6), NOT neutral-atom vdW radii, and
# this is a deliberate, load-bearing choice: a C-type lectin coordinates a
# sugar's O3/O4 at ~2.4 A from the Ca(2+) center. With the neutral-atom radius
# (Ca 2.31 A) that oxygen sits *inside* the ion, the coordination site reads as
# receptor interior, and no grid point is ever generated in the one place the
# glycan is known to bind. With the ionic radius (Ca(2+) 1.14 A) the site is
# open and a ligand center lands 2.1-5.1 A from the ion, which is where the
# ring of a coordinating sugar belongs.
ELEMENT_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80,
    'F': 1.47, 'CL': 1.75, 'BR': 1.85, 'I': 1.98, 'SE': 1.90, 'B': 1.92,
    'LI': 0.76, 'NA': 1.02, 'K': 1.38, 'MG': 0.72, 'CA': 1.14,
    'MN': 0.83, 'FE': 0.78, 'CO': 0.75, 'NI': 0.69, 'CU': 0.73, 'ZN': 0.74,
}
DEFAULT_RADIUS = 1.70           # unknown element: treat as carbon


class GridError(RuntimeError):
    '''Raised when a grid cannot be built from the given inputs.'''


# --- receptor ---------------------------------------------------------------

def _pymol():
    '''The PyMOL cmd module, launching a headless session on first use.'''
    import pymol
    from pymol import cmd
    if not getattr(_pymol, '_launched', False):
        pymol.finish_launching(['pymol', '-qc'])
        _pymol._launched = True
    return cmd


def _residue_rows(cmd, selection) -> List[Tuple[str, str, str]]:
    '''[(chain, resi, resn), ...] for a selection, each residue once, in order.'''
    rows: List[Tuple[str, str, str]] = []
    cmd.iterate(selection, 'rows.append((chain, resi, resn))',
                space={'rows': rows})
    seen, out = set(), []
    for row in rows:
        if row not in seen:
            seen.add(row)
            out.append(row)
    return out


def fmt_residues(rows: Sequence[Tuple[str, str, str]], limit: int = 12) -> str:
    '''"A/ASP366 A/GLU354 ..." for printing what a selection matched.'''
    shown = [f'{c}/{n}{i}' for c, i, n in rows[:limit]]
    if len(rows) > limit:
        shown.append(f'... (+{len(rows) - limit} more)')
    return ' '.join(shown)


@dataclass
class Receptor:
    '''
    The atoms a grid is built against: their coordinates, their van der Waals
    radii, and a live PyMOL object to resolve further selections with.
    '''
    path: str
    selection: str
    coords: np.ndarray                      # (N, 3) float
    radii: np.ndarray                       # (N,) float
    residues: List[Tuple[str, str, str]] = field(default_factory=list)
    object_name: Optional[str] = None

    @property
    def n_atoms(self) -> int:
        return len(self.coords)

    @property
    def stem(self) -> str:
        '''Receptor basename without .pdb, used to name the grid file.'''
        return os.path.basename(self.path).rsplit('.pdb', 1)[0]

    def bounds(self, margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        '''Axis-aligned bounds of the atom *surfaces*, grown by `margin`.'''
        r = self.radii[:, None]
        return ((self.coords - r).min(axis=0) - margin,
                (self.coords + r).max(axis=0) + margin)

    def site_coords(self, selection: str, *, report: bool = True) -> np.ndarray:
        '''
        Coordinates of a PyMOL selection within this receptor, e.g.
        'chain A and resi 365+366' or 'byres (polymer within 5 of resn CA)'.

        The selection is intersected with the receptor object, so it can be
        written without naming the object.
        '''
        if self.object_name is None:
            raise GridError(
                'this Receptor was built from coordinates, not from a PDB, '
                'so PyMOL selections cannot be resolved against it.')
        cmd = _pymol()
        sel = f'({self.object_name}) and ({selection})'
        n = cmd.count_atoms(sel)
        if n == 0:
            raise GridError(
                f'selection {selection!r} matched no atoms in {self.path}. '
                f'Check chain IDs and residue numbers -- prepare_complexes.py '
                f'may have renumbered them (see its *_resmap.csv).')
        coords = np.asarray(cmd.get_coords(sel), dtype=float)
        if report:
            rows = _residue_rows(cmd, sel)
            print(f'  site {selection!r}: {n} atoms, {len(rows)} residues '
                  f'[{fmt_residues(rows)}]')
        return coords

    def close(self):
        '''Delete the PyMOL object backing this receptor.'''
        if self.object_name is not None:
            _pymol().delete(self.object_name)
            self.object_name = None


def atom_radii(elements: Iterable[str],
               overrides: Optional[Dict[str, float]] = None) -> np.ndarray:
    '''
    van der Waals / ionic radii for a list of element symbols, from
    ELEMENT_RADII (see its note on metal ions). Unknown elements fall back to
    DEFAULT_RADIUS and are reported once each.
    '''
    table = dict(ELEMENT_RADII)
    for k, v in (overrides or {}).items():
        table[k.strip().upper()] = float(v)

    radii, unknown = [], {}
    for e in elements:
        key = (e or '').strip().upper()
        if key in table:
            radii.append(table[key])
        else:
            unknown[key] = unknown.get(key, 0) + 1
            radii.append(DEFAULT_RADIUS)
    for key, count in unknown.items():
        print(f'  warning: no radius for element {key or "(blank)"!r} '
              f'({count} atoms); using {DEFAULT_RADIUS} A. Set one with '
              f'--radius {key or "X"}=<value> if that is wrong.',
              file=sys.stderr)
    return np.asarray(radii, dtype=float)


_STANDARD_RESN = frozenset((
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'))


def load_receptor(pdb_path: str,
                  selection: str = DEFAULT_SURFACE_SELECTION, *,
                  radius_overrides: Optional[Dict[str, float]] = None,
                  report: bool = True) -> Receptor:
    '''
    Read a receptor PDB with PyMOL and keep the atoms matching `selection` as
    the surface the grid is measured from.

    Build the grid from the *prepared* receptor (the one that goes into
    docking), not the raw PDB: prepare_complexes.py renames chains and may
    renumber residues, so selections written against the raw file can silently
    match the wrong atoms -- or nothing.
    '''
    cmd = _pymol()
    obj = 'glycog_receptor'
    cmd.delete(obj)
    cmd.load(pdb_path, obj)
    if cmd.count_atoms(obj) == 0:
        raise GridError(f'{pdb_path}: no atoms read.')

    sel = f'({obj}) and ({selection})'
    n = cmd.count_atoms(sel)
    if n == 0:
        raise GridError(
            f'surface selection {selection!r} matched no atoms in {pdb_path}.')

    coords = np.asarray(cmd.get_coords(sel), dtype=float)
    elements: List[str] = []
    cmd.iterate(sel, 'elements.append(elem)', space={'elements': elements})
    radii = atom_radii(elements, radius_overrides)
    residues = _residue_rows(cmd, sel)

    if report:
        dropped = cmd.count_atoms(obj) - n
        print(f'Receptor {pdb_path}: {n} atoms, {len(residues)} residues '
              f'(surface selection {selection!r}'
              + (f', {dropped} atoms excluded)' if dropped else ')'))
        hetero = sorted({r[2] for r in residues if r[2] not in _STANDARD_RESN})
        if hetero:
            print(f'  non-protein residues on the surface: {" ".join(hetero)}')

    return Receptor(path=pdb_path, selection=selection, coords=coords,
                    radii=radii, residues=residues, object_name=obj)


# --- where to sample --------------------------------------------------------

@dataclass(frozen=True)
class SamplingRegion:
    '''
    Where sampling points are allowed. An empty region means the whole
    receptor surface; a box and a set of site spheres can be combined, in
    which case a point must satisfy both.
    '''
    box_min: Optional[Tuple[float, float, float]] = None
    box_max: Optional[Tuple[float, float, float]] = None
    site_coords: Optional[np.ndarray] = None
    site_radius: float = DEFAULT_SITE_RADIUS
    labels: Tuple[str, ...] = ()

    @classmethod
    def from_box(cls, bb_min, bb_max) -> 'SamplingRegion':
        lo = np.asarray(bb_min, dtype=float)
        hi = np.asarray(bb_max, dtype=float)
        if lo.shape != (3,) or hi.shape != (3,):
            raise GridError('a box needs three numbers for each corner.')
        if np.any(hi <= lo):
            raise GridError(
                f'box max {tuple(hi)} is not greater than box min {tuple(lo)} '
                f'on every axis.')
        return cls(box_min=tuple(lo), box_max=tuple(hi))

    @classmethod
    def around(cls, coords, radius: float = DEFAULT_SITE_RADIUS,
               labels: Sequence[str] = ()) -> 'SamplingRegion':
        '''Within `radius` of any of `coords` -- a union of spheres.'''
        coords = np.atleast_2d(np.asarray(coords, dtype=float))
        if len(coords) == 0 or coords.shape[1] != 3:
            raise GridError('site coordinates must be an (N, 3) array.')
        if radius <= 0:
            raise GridError('--site-radius must be positive.')
        return cls(site_coords=coords, site_radius=float(radius),
                   labels=tuple(labels))

    def with_box(self, bb_min, bb_max) -> 'SamplingRegion':
        box = SamplingRegion.from_box(bb_min, bb_max)
        return SamplingRegion(box_min=box.box_min, box_max=box.box_max,
                              site_coords=self.site_coords,
                              site_radius=self.site_radius, labels=self.labels)

    @property
    def is_global(self) -> bool:
        return self.box_min is None and self.site_coords is None

    def bounds(self, receptor: Receptor,
               margin: float) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Lattice bounds: the region grown by `margin` (so the surface geometry
        around its edge is computed correctly), never larger than the receptor
        plus the same margin.
        '''
        lo, hi = receptor.bounds(margin)
        if self.site_coords is not None:
            reach = self.site_radius + margin
            lo = np.maximum(lo, self.site_coords.min(axis=0) - reach)
            hi = np.minimum(hi, self.site_coords.max(axis=0) + reach)
        if self.box_min is not None:
            lo = np.maximum(lo, np.asarray(self.box_min) - margin)
            hi = np.minimum(hi, np.asarray(self.box_max) + margin)
        if np.any(hi <= lo):
            raise GridError(
                'the requested region does not overlap the receptor: '
                f'bounds collapsed to {tuple(np.round(lo, 2))} .. '
                f'{tuple(np.round(hi, 2))}.')
        return lo, hi

    def contains(self, points: np.ndarray) -> np.ndarray:
        '''Boolean mask of which points the region allows.'''
        mask = np.ones(len(points), dtype=bool)
        if self.box_min is not None:
            mask &= np.all((points >= np.asarray(self.box_min)) &
                           (points <= np.asarray(self.box_max)), axis=1)
        if self.site_coords is not None:
            tree = cKDTree(self.site_coords)
            near = tree.query_ball_point(points, self.site_radius,
                                         return_length=True)
            mask &= near > 0
        return mask

    def describe(self) -> str:
        if self.is_global:
            return 'whole receptor surface'
        parts = []
        if self.site_coords is not None:
            sel = '; '.join(self.labels) if self.labels \
                else f'{len(self.site_coords)} atoms'
            parts.append(f'within {self.site_radius:g} A of [{sel}]')
        if self.box_min is not None:
            parts.append(f'inside box {tuple(np.round(self.box_min, 2))} .. '
                         f'{tuple(np.round(self.box_max, 2))}')
        return ' and '.join(parts)

    def to_dict(self) -> Dict:
        return {
            'type': 'global' if self.is_global else 'local',
            'box_min': list(self.box_min) if self.box_min else None,
            'box_max': list(self.box_max) if self.box_max else None,
            'site_selections': list(self.labels) or None,
            'site_atoms': (None if self.site_coords is None
                           else len(self.site_coords)),
            'site_radius': (None if self.site_coords is None
                            else self.site_radius),
        }


# --- surface geometry -------------------------------------------------------

def vdw_distance(points: np.ndarray, coords: np.ndarray, radii: np.ndarray, *,
                 chunk: int = 2_000_000) -> np.ndarray:
    '''
    Exact distance from each point to the union of atom spheres,
    min_i(|p - c_i| - r_i); negative inside an atom.

    One KD-tree per distinct radius. Querying a single tree of centers and
    subtracting a radius afterwards would be wrong whenever radii differ: the
    nearest center is not necessarily the nearest *surface*.
    '''
    points = np.ascontiguousarray(points, dtype=float)
    out = np.full(len(points), np.inf)
    for r in np.unique(radii):
        tree = cKDTree(coords[radii == r])
        for start in range(0, len(points), chunk):
            sl = slice(start, start + chunk)
            d, _ = tree.query(points[sl], k=1, workers=-1)
            np.minimum(out[sl], d - r, out=out[sl])
    return out


def _lattice_axes(lo, hi, spacing) -> List[np.ndarray]:
    # +0.5*spacing so the upper bound is included when it lands on a node.
    return [np.arange(lo[k], hi[k] + 0.5 * spacing, spacing) for k in range(3)]


def _lattice_points(axes) -> np.ndarray:
    mesh = np.meshgrid(*axes, indexing='ij')
    return np.stack([m.ravel() for m in mesh], axis=1)


_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=int)


def _interior_mask(d_vdw: np.ndarray, radius: float,
                   spacing: float) -> np.ndarray:
    '''
    Voxels a probe of `radius` rolling in from bulk solvent cannot cover:
    receptor interior, plus any crevice or sealed cavity too tight for it.

    `d_vdw` is the exact van der Waals distance field on the lattice.
    '''
    fits = d_vdw >= radius
    if not fits.any():
        raise GridError(
            f'no room anywhere in the lattice for a probe of radius '
            f'{radius:g} A. The region is probably buried inside the receptor.')

    labels, _ = ndimage.label(fits, structure=_CONNECTIVITY_26)
    edge = np.concatenate([
        labels[0].ravel(), labels[-1].ravel(),
        labels[:, 0].ravel(), labels[:, -1].ravel(),
        labels[:, :, 0].ravel(), labels[:, :, -1].ravel()])
    bulk_labels = np.setdiff1d(np.unique(edge), [0])
    if bulk_labels.size == 0:
        raise GridError(
            f'no free space of radius {radius:g} A reaches the edge of the '
            f'lattice, so bulk solvent could not be identified. Widen the '
            f'region (--site-radius / --box) or lower --fit-radius.')
    bulk = np.isin(labels, bulk_labels)
    del labels

    # Solvent = swept volume of the bulk-connected probe centers.
    covered = ndimage.distance_transform_edt(~bulk, sampling=spacing) <= radius
    return ~covered


def surface_fields(receptor: Receptor, lo, hi, spacing: float, *,
                   probe_radius: float = DEFAULT_PROBE_RADIUS,
                   fit_radius: Optional[float] = DEFAULT_FIT_RADIUS,
                   verbose: bool = True):
    '''
    Compute, on a lattice spanning [lo, hi] at `spacing`:

      height     distance above the solvent-excluded surface (0 inside)
      reachable  True where a sphere of `fit_radius` rolling in from bulk
                 solvent can reach (all True when fit_radius is None)

    Returns (axes, height, reachable).
    '''
    axes = _lattice_axes(lo, hi, spacing)
    shape = tuple(len(a) for a in axes)
    n_voxels = int(np.prod(shape))
    if verbose:
        print(f'Surface lattice: {shape[0]} x {shape[1]} x {shape[2]} '
              f'= {n_voxels:,} voxels at {spacing:g} A')

    d = vdw_distance(_lattice_points(axes), receptor.coords,
                     receptor.radii).reshape(shape)

    interior = _interior_mask(d, probe_radius, spacing)
    if not interior.any():
        raise GridError(
            'no receptor interior found inside the lattice -- the region does '
            'not appear to touch the receptor at all.')
    height = ndimage.distance_transform_edt(~interior, sampling=spacing)
    # The exact van der Waals distance is an upper bound on the distance to
    # the molecular surface, and is exactly it wherever the surface is convex.
    np.minimum(height, np.maximum(d, 0.0), out=height)

    if fit_radius:
        reachable = ~_interior_mask(d, fit_radius, spacing)
    else:
        reachable = np.ones(shape, dtype=bool)

    return axes, height, reachable


# --- the grid ---------------------------------------------------------------

@dataclass
class SamplingGrid:
    '''A built grid: its points and everything needed to reproduce them.'''
    points: np.ndarray
    receptor: str
    spacing: float
    shell: Tuple[float, float]
    params: Dict = field(default_factory=dict)
    counts: Dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.points)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    def to_dict(self) -> Dict:
        lo, hi = self.bounds() if self.n_points else (np.zeros(3), np.zeros(3))
        return {
            'receptor': self.receptor,
            'n_points': self.n_points,
            'spacing': self.spacing,
            'shell_min': self.shell[0],
            'shell_max': self.shell[1],
            'extent_min': [round(float(v), 3) for v in lo],
            'extent_max': [round(float(v), 3) for v in hi],
            'counts': self.counts,
            **self.params,
        }

    def summary(self) -> str:
        c = self.counts
        lines = [
            f'{self.n_points:,} sampling points '
            f'({self.spacing:g} A apart, {self.shell[0]:g}-{self.shell[1]:g} A '
            f'above the surface)']
        if self.n_points:
            lo, hi = self.bounds()
            lines.append('  extent  ' + '  '.join(
                f'{a:.1f}..{b:.1f}' for a, b in zip(lo, hi)))
        order = [('candidates', 'lattice nodes in the region'),
                 ('in_shell', 'in the surface shell'),
                 ('reachable', 'also reachable by the fit probe'),
                 ('in_region', 'also inside the region')]
        for key, label in order:
            if key in c:
                lines.append(f'  {c[key]:>10,}  {label}')
        return '\n'.join(lines)


def _resolve_lattice(spacing: float,
                     lattice_spacing: Optional[float]) -> Tuple[float, int]:
    '''Geometry-lattice pitch and the stride from it to the sampling pitch.'''
    if lattice_spacing is None:
        n = max(1, int(np.ceil(spacing / MAX_LATTICE_SPACING - 1e-9)))
        return spacing / n, n
    g = float(lattice_spacing)
    if g <= 0:
        raise GridError('--lattice-spacing must be positive.')
    ratio = spacing / g
    stride = int(round(ratio))
    if stride < 1 or abs(ratio - stride) > 1e-6:
        raise GridError(
            f'--spacing ({spacing:g}) must be a whole multiple of '
            f'--lattice-spacing ({g:g}); got a ratio of {ratio:.4f}.')
    return g, stride


def _site_clusters(coords, cutoff):
    '''
    Split site atoms into groups further than `cutoff` apart, so each group
    can get its own lattice.

    Single linkage: any two atoms within `cutoff` stay together, so two groups
    can never want overlapping lattices. The point of --site is to sample two
    faces of a receptor without paying for the volume in between, and one
    lattice spanning both would cost exactly that volume.
    '''
    if len(coords) == 1:
        return [coords]
    from scipy.sparse.csgraph import connected_components
    tree = cKDTree(coords)
    n, labels = connected_components(tree.sparse_distance_matrix(tree, cutoff),
                                     directed=False)
    return [coords[labels == k] for k in range(n)]


def _build_patch(receptor: Receptor, region: SamplingRegion, *,
                 spacing: float, shell: Tuple[float, float],
                 probe_radius: float, fit_radius: Optional[float],
                 g: float, stride: int, margin: float, max_voxels: int,
                 verbose: bool) -> Tuple[np.ndarray, Dict[str, int]]:
    '''Build the points for one lattice. Returns (points, stage counts).'''
    shell_min, shell_max = shell
    lo, hi = region.bounds(receptor, margin)
    # Snap to a global lattice of pitch `spacing` -- not of pitch g -- so that
    # two patches, or two grids built for the same receptor, put their
    # sampling points on exactly the same nodes. Snapping to g alone would
    # align the geometry lattices but still let the every-stride-th sampling
    # sub-lattice land on a different parity in each patch.
    pitch = g * stride
    lo = np.floor(lo / pitch) * pitch

    shape = tuple(len(a) for a in _lattice_axes(lo, hi, g))
    n_voxels = int(np.prod(shape))
    if n_voxels > max_voxels:
        raise GridError(
            f'the surface lattice would be {shape[0]} x {shape[1]} x '
            f'{shape[2]} = {n_voxels:,} voxels, over the {max_voxels:,} limit. '
            f'Restrict the region (--site/--box), widen --spacing or '
            f'--lattice-spacing, or raise --max-voxels if you have the memory '
            f'(~50 bytes/voxel).')

    axes, height, reachable = surface_fields(
        receptor, lo, hi, g, probe_radius=probe_radius,
        fit_radius=fit_radius, verbose=verbose)

    # Sampling points are every `stride`-th lattice node, so they inherit the
    # computed geometry without any interpolation.
    sub = tuple(slice(None, None, stride) for _ in range(3))
    points = _lattice_points([a[::stride] for a in axes])
    height_s = height[sub].ravel()
    reach_s = reachable[sub].ravel()

    counts = {'candidates': int(len(points))}
    mask = (height_s >= shell_min) & (height_s <= shell_max)
    counts['in_shell'] = int(mask.sum())
    if fit_radius:
        mask &= reach_s
        counts['reachable'] = int(mask.sum())
    if not region.is_global:
        mask &= region.contains(points)
        counts['in_region'] = int(mask.sum())
    return points[mask], counts


def _margin(shell_max: float, probe_radius: float,
            fit_radius: Optional[float], g: float) -> float:
    '''
    How far outside the region the geometry lattice has to extend: enough for
    the surface, the bulk-solvent flood fill and the fit probe to be resolved
    correctly right up to the region's edge.
    '''
    return shell_max + 2 * max(probe_radius, fit_radius or 0.0) + 2 * g


def plan_lattice(receptor: Receptor,
                 region: Optional[SamplingRegion] = None, *,
                 spacing: float = DEFAULT_SPACING,
                 shell: Tuple[float, float] = DEFAULT_SHELL,
                 probe_radius: float = DEFAULT_PROBE_RADIUS,
                 fit_radius: Optional[float] = DEFAULT_FIT_RADIUS,
                 lattice_spacing: Optional[float] = None,
                 split_sites: bool = True) -> List[Dict]:
    '''
    What build_sampling_grid would allocate, without allocating it: one dict
    per lattice patch, with its bounds, shape, voxel count and the number of
    sampling nodes before any filtering. Backs --dry-run.
    '''
    region = region or SamplingRegion()
    g, stride = _resolve_lattice(spacing, lattice_spacing)
    margin = _margin(float(shell[1]), probe_radius, fit_radius, g)
    pitch = g * stride

    patches = [region]
    if split_sites and region.site_coords is not None:
        groups = _site_clusters(region.site_coords,
                                2 * (region.site_radius + margin))
        if len(groups) > 1:
            patches = [SamplingRegion(box_min=region.box_min,
                                      box_max=region.box_max, site_coords=c,
                                      site_radius=region.site_radius,
                                      labels=region.labels) for c in groups]

    plans = []
    for patch in patches:
        lo, hi = patch.bounds(receptor, margin)
        lo = np.floor(lo / pitch) * pitch
        shape = [len(a) for a in _lattice_axes(lo, hi, g)]
        plans.append({
            'lo': lo, 'hi': hi, 'shape': shape,
            'n_voxels': int(np.prod(shape)),
            'lattice_spacing': g, 'stride': stride,
            'n_nodes': int(np.prod([len(range(0, n, stride)) for n in shape])),
        })
    return plans


def build_sampling_grid(receptor: Receptor, *,
                        region: Optional[SamplingRegion] = None,
                        spacing: float = DEFAULT_SPACING,
                        shell: Tuple[float, float] = DEFAULT_SHELL,
                        probe_radius: float = DEFAULT_PROBE_RADIUS,
                        fit_radius: Optional[float] = DEFAULT_FIT_RADIUS,
                        lattice_spacing: Optional[float] = None,
                        max_voxels: int = MAX_VOXELS,
                        split_sites: bool = True,
                        verbose: bool = True) -> SamplingGrid:
    '''
    Build the sampling grid for `receptor` inside `region` (default: the whole
    receptor surface).

    Points sit on a cubic lattice of pitch `spacing`, in the shell
    `shell = (min, max)` angstroms above the solvent-excluded surface, and --
    unless fit_radius is None -- only where a sphere of `fit_radius` rolling
    in from bulk solvent can reach.

    `lattice_spacing` is the pitch of the geometry lattice the surface is
    computed on; it must divide `spacing`. By default it is the largest
    divisor of `spacing` that is <= 1 A.

    With `split_sites` (the default), site selections far enough apart get a
    lattice each instead of one lattice spanning the gap between them. The
    points come out identical either way -- every lattice is snapped to the
    same global one -- but a grid over two faces of a large receptor then
    costs the two neighbourhoods rather than their whole bounding box.
    '''
    region = region or SamplingRegion()
    shell_min, shell_max = float(shell[0]), float(shell[1])
    if shell_max <= shell_min:
        raise GridError(
            f'shell max ({shell_max}) must exceed shell min ({shell_min}).')
    if spacing <= 0:
        raise GridError('spacing must be positive.')

    g, stride = _resolve_lattice(spacing, lattice_spacing)

    margin = _margin(shell_max, probe_radius, fit_radius, g)

    patches = [region]
    if split_sites and region.site_coords is not None:
        groups = _site_clusters(region.site_coords,
                                2 * (region.site_radius + margin))
        if len(groups) > 1:
            patches = [SamplingRegion(box_min=region.box_min,
                                      box_max=region.box_max,
                                      site_coords=c,
                                      site_radius=region.site_radius,
                                      labels=region.labels)
                       for c in groups]

    if verbose:
        print(f'Region: {region.describe()}')
        if len(patches) > 1:
            print(f'  {len(patches)} separated patches, one lattice each')

    counts: Dict[str, int] = {}
    chunks = []
    for n, patch in enumerate(patches, 1):
        if verbose and len(patches) > 1:
            print(f'  patch {n}/{len(patches)}:')
        pts, c = _build_patch(
            receptor, patch, spacing=spacing, shell=(shell_min, shell_max),
            probe_radius=probe_radius, fit_radius=fit_radius, g=g,
            stride=stride, margin=margin, max_voxels=max_voxels,
            verbose=verbose)
        chunks.append(pts)
        for key, value in c.items():
            counts[key] = counts.get(key, 0) + value

    kept = np.vstack(chunks) if chunks else np.zeros((0, 3))
    # np.unique sorts lexicographically and drops any point that two
    # overlapping patches both produced, so the file is stable and
    # duplicate-free: a duplicated start point would quietly bias StartFrom
    # toward that one spot.
    if len(kept):
        kept = np.unique(np.round(kept, 4), axis=0)
    counts['points'] = int(len(kept))

    params = {
        'surface_selection': receptor.selection,
        'probe_radius': probe_radius,
        'fit_radius': fit_radius,
        'lattice_spacing': g,
        'patches': len(patches),
        'receptor_atoms': receptor.n_atoms,
        'region': region.to_dict(),
    }
    grid = SamplingGrid(points=kept, receptor=os.path.abspath(receptor.path),
                        spacing=spacing, shell=(shell_min, shell_max),
                        params=params, counts=counts)
    if verbose:
        print(grid.summary())
    if grid.n_points == 0:
        print('warning: the grid is empty. Widen --shell, enlarge the region, '
              'or check that the selection is on the receptor surface.',
              file=sys.stderr)
    return grid


# --- output -----------------------------------------------------------------

PDB_SERIAL_LIMIT = 99999


def grid_filename(receptor_stem: str, grid_id: str, ext: str = '.pdb') -> str:
    '''<receptor>_grid_<grid_id>.pdb -- e.g. 1o7v_grid_global.pdb.'''
    return f'{receptor_stem}_grid_{naming.validate_grid_id(grid_id)}{ext}'


def _remark_lines(grid: SamplingGrid) -> List[str]:
    out = []
    for key, value in grid.to_dict().items():
        if key in ('counts', 'region'):
            continue
        out.append(f'{key} {value}')
    for key, value in (grid.params.get('region') or {}).items():
        if value is not None:
            out.append(f'region.{key} {value}')
    for key, value in grid.counts.items():
        out.append(f'count.{key} {value}')
    return out


def write_grid_pdb(grid: SamplingGrid, path: str) -> str:
    '''
    Write the points as PSD pseudo-atoms on chain P -- the format Rosetta's
    StartFrom mover reads (StartFrom.parse_pdb_file) and the one every
    existing glycographer grid uses.
    '''
    if grid.n_points > PDB_SERIAL_LIMIT:
        print(f'warning: {grid.n_points:,} points exceed the '
              f'{PDB_SERIAL_LIMIT:,} PDB atom serials; serials will wrap. Use '
              f'a coarser --spacing or a smaller region.', file=sys.stderr)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as f:
        f.write('HEADER    GLYCOGRAPHER SAMPLING GRID\n')
        for line in _remark_lines(grid):
            f.write(f'REMARK    {line}\n')
        for i, p in enumerate(grid.points, 1):
            serial = (i - 1) % PDB_SERIAL_LIMIT + 1
            resi = (i - 1) % 9999 + 1
            f.write(f'ATOM  {serial:5d}  O   PSD P{resi:4d}    '
                    f'{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}'
                    f'  1.00 20.00           O  \n')
        f.write('END\n')
    print(f'Wrote {grid.n_points:,} points to {path}')
    return path


def write_grid_xyz(grid: SamplingGrid, path: str) -> str:
    '''Write the points as an XYZ file (one "X x y z" line per point).'''
    with open(path, 'w') as f:
        f.write(f'{grid.n_points}\n')
        f.write('glycographer sampling grid; '
                + '; '.join(_remark_lines(grid)) + '\n')
        for p in grid.points:
            f.write(f'X {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n')
    print(f'Wrote {grid.n_points:,} points to {path}')
    return path


def write_grid_manifest(grid: SamplingGrid, path: str) -> str:
    '''Write the grid's full provenance as JSON beside the PDB.'''
    with open(path, 'w') as f:
        json.dump(grid.to_dict(), f, indent=2, sort_keys=False)
        f.write('\n')
    print(f'Wrote grid manifest to {path}')
    return path


def write_grid(grid: SamplingGrid, pdb_path: str, *, xyz: bool = False,
               manifest: bool = True) -> Dict[str, str]:
    '''Write the grid PDB and, by default, its JSON manifest.'''
    stem = pdb_path.rsplit('.pdb', 1)[0]
    out = {'pdb': write_grid_pdb(grid, pdb_path)}
    if xyz:
        out['xyz'] = write_grid_xyz(grid, f'{stem}.xyz')
    if manifest:
        out['manifest'] = write_grid_manifest(grid, f'{stem}.json')
    return out


def read_grid_points(pdb_path: str) -> np.ndarray:
    '''Read the point coordinates back out of a grid PDB.'''
    pts = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('ATOM  ', 'HETATM')):
                pts.append((float(line[30:38]), float(line[38:46]),
                            float(line[46:54])))
    return np.asarray(pts, dtype=float)


def read_grid_params(pdb_path: str) -> Dict[str, str]:
    '''Read the REMARK key/value provenance out of a grid PDB.'''
    params = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('REMARK'):
                continue
            fields = line[6:].strip().split(None, 1)
            if len(fields) == 2:
                params[fields[0]] = fields[1]
    return params
