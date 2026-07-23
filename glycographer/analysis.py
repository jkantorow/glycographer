'''
Analysis tools for the output of a Glycographer simulation.

These functions mostly operate on data contained within a VolMap or
GlycanDockEnsemble instance and do not require PyRosetta to execute.
'''

import numpy as np
import pandas as pd
from scipy import ndimage

# seaborn / matplotlib are imported lazily inside the plotting functions so
# that importing this module (for map_stats, standardize_favorability,
# find_hotspots, ...) stays lightweight -- map.py and dock.py depend on it.


def map_stats(volmap, occupied_only=True):
    '''
    Distribution statistics of a volume map, computed directly from the voxel
    array in numpy (ground truth).

    Prefer this over PyMOL's cmd.get_volume_histogram, whose reported bounds are
    histogram bin edges rather than true min/max and therefore disagree with the
    values seen at map load. Empty voxels carry a sentinel of 0.0; with
    occupied_only=True (the default) they are excluded so the mean/std describe
    the actual sampled field rather than being diluted toward zero.

    Parameters
    ----------
    volmap : VolMap or np.ndarray
        A VolMap (its .values are used) or a raw flat/3D value array.
    occupied_only : bool
        Exclude exactly-zero (empty) voxels from the statistics.

    Returns
    -------
    dict
        {'min', 'max', 'mean', 'std', 'n_occupied', 'n_voxels'}. The min/max/
        mean/std are None when there are no qualifying voxels.
    '''
    values = np.asarray(getattr(volmap, 'values', volmap), dtype=float).ravel()
    n_voxels = values.size

    sample = values[values != 0.0] if occupied_only else values
    if sample.size == 0:
        return {'min': None, 'max': None, 'mean': None, 'std': None,
                'n_occupied': 0, 'n_voxels': n_voxels}

    return {
        'min': float(np.min(sample)),
        'max': float(np.max(sample)),
        'mean': float(np.mean(sample)),
        'std': float(np.std(sample)),
        'n_occupied': int(np.count_nonzero(values != 0.0)),
        'n_voxels': n_voxels,
    }


def standardize_favorability(values, method='percentile'):
    '''
    Map raw energies (favorable = negative) to a per-distribution favorability
    score in which higher = more favorable, so values can be compared *across*
    probes/ensembles whose absolute REU ranges differ.

    Why: raw REU is the right scale for absolute-depth questions (is this
    location bindable), but for cross-probe comparison (which probe is
    unusually good here, how selective is a site) it is biased -- a larger or
    charged fragment has systematically deeper interaction energies everywhere.
    Standardizing each probe against its *own* distribution removes that offset,
    so the score means "favorable relative to this probe" rather than "deep in
    absolute REU". This is the input the best_probe / support / selectivity
    consensus reductions should use; keep raw REU for consensus_min/mean.

    Only favorable (negative) values carry signal; non-negative values (the
    0.0 empty sentinel and any positive residue) map to 0.0.

    method
      'percentile' (default): favorability percentile in (0, 1] over the
          favorable values, 1 = most favorable in this distribution. Robust,
          nonparametric, and directly interpretable as "top X% for this probe"
          -- which is what makes a support-count threshold comparable across
          probes.
      'robust': (depth - median)/(1.4826*MAD) over favorable values; a signed,
          outlier-resistant z-score of favorability.
      'minmax': depth linearly scaled to [0, 1] between the least and most
          favorable value (the legacy scale_inteng behavior; outlier-sensitive).

    Returns an array shaped like ``values``.
    '''
    v = np.asarray(values, dtype=float)
    out = np.zeros_like(v)
    fav_mask = v < 0
    if not fav_mask.any():
        return out
    depth = -v[fav_mask]                       # positive favorability

    if method == 'percentile':
        from scipy.stats import rankdata
        r = rankdata(depth, method='average')
        out[fav_mask] = r / r.max()            # (0, 1], 1 = most favorable
    elif method == 'robust':
        med = np.median(depth)
        mad = np.median(np.abs(depth - med)) or 1.0
        out[fav_mask] = (depth - med) / (1.4826 * mad)
    elif method == 'minmax':
        lo, hi = depth.min(), depth.max()
        out[fav_mask] = (depth - lo) / (hi - lo) if hi > lo else 1.0
    else:
        raise ValueError(f"Unknown method {method!r}; choose 'percentile', "
                         "'robust', or 'minmax'.")
    return out


def _favorable_field(volmap, smooth_sigma=None):
    '''
    Return the 3D voxel field prepared for contouring / segmentation.

    Empty voxels carry a 0.0 sentinel and favorable binding is negative, so the
    field is used as-is (thresholds are negative). Optional Gaussian smoothing
    makes isosurfaces and segment boundaries less jagged; note it pulls edge
    voxels toward the 0 background and so slightly shrinks features, which is
    fine for display but keep sigma small (~0.5-1.0 voxel) for detection.
    '''
    field = np.asarray(volmap._values_3d, dtype=float)
    if smooth_sigma:
        field = ndimage.gaussian_filter(field, sigma=smooth_sigma)
    return field


def component_sweep(volmap, levels=None, n_sweep=60, min_voxels=3,
                    smooth_sigma=None):
    '''
    Count distinct favorable components as a function of contour level.

    Sweeps candidate (negative) levels and, at each, thresholds the map
    (value < level) and labels connected voxel blobs, counting those with at
    least ``min_voxels`` voxels. As the level rises from the map minimum the
    count grows (more distinct hotspots resolve) then falls (neighboring
    hotspots merge -- the "bleeding" you see as diamonds-within-diamonds). The
    level at peak count is the most a contour can be raised while keeping sites
    separate, and is what the 'components' mode of choose_contour_levels uses as
    its upper bound.

    Returns
    -------
    (levels, counts) : two np.ndarrays of equal length.
    '''
    field = _favorable_field(volmap, smooth_sigma)
    fav = field[field < 0]
    if fav.size == 0:
        return np.array([]), np.array([])

    if levels is None:
        # Sweep from the deepest voxel up to the 60th percentile of favorable
        # values -- merging almost always happens well below the median.
        lo = float(fav.min())
        hi = float(np.percentile(fav, 60))
        levels = np.linspace(lo, hi, n_sweep)

    counts = np.empty(len(levels), dtype=int)
    for i, lv in enumerate(levels):
        labels, n = ndimage.label(field < lv)
        if n == 0:
            counts[i] = 0
            continue
        sizes = np.bincount(labels.ravel())[1:]  # drop background (label 0)
        counts[i] = int(np.count_nonzero(sizes >= min_voxels))
    return np.asarray(levels), counts


def choose_contour_levels(volmap, n=4, mode='absolute', step=1.0,
                          quantiles=(0.001, 0.005, 0.02, 0.05),
                          min_gap=0.5, smooth_sigma=None):
    '''
    Choose isocontour levels for a favorable (negative) volume map.

    Modes
    -----
    'absolute' (default)
        Tail-anchored: start at the map minimum and step up by ``step`` REU
        (e.g. -20, -19, -18, ...). Matches the by-hand method and keeps levels
        on a shared absolute scale, so contours are directly comparable across
        probes.
    'quantile'
        Levels at tail-hugging quantiles of the favorable-voxel distribution
        (default the deepest 0.1-5%). Self-normalizing, so it adapts to maps of
        very different depth without re-picking ``step``; not comparable in
        absolute REU across probes.
    'components'
        Data-driven anti-bleed: use component_sweep to find the level where the
        most distinct hotspots resolve, then place ``n`` levels between the map
        minimum and that level. Automatically avoids the range where sites merge.

    ``min_gap`` drops levels closer than that many REU to the previous kept
    level, preventing nested shells within a voxel. Levels are returned most
    negative (deepest) first.
    '''
    field = _favorable_field(volmap, smooth_sigma)
    fav = field[field < 0]
    if fav.size == 0:
        return []

    lo = float(fav.min())
    if mode == 'absolute':
        raw = lo + np.arange(n) * step
    elif mode == 'quantile':
        raw = np.quantile(fav, list(quantiles)[:n])
    elif mode == 'components':
        levels, counts = component_sweep(volmap, min_voxels=3,
                                         smooth_sigma=smooth_sigma)
        if counts.size == 0 or counts.max() == 0:
            raw = np.linspace(lo, float(np.percentile(fav, 30)), n)
        else:
            l_peak = float(levels[int(np.argmax(counts))])
            raw = np.linspace(lo, l_peak, n)
    else:
        raise ValueError(f"Unknown mode {mode!r}; choose 'absolute', "
                         "'quantile', or 'components'.")

    # Keep only favorable levels, ascending (deepest first), spaced by min_gap.
    raw = np.sort(np.asarray(raw, dtype=float))
    kept = []
    for lv in raw:
        if lv >= 0:
            continue
        if not kept or abs(lv - kept[-1]) >= min_gap:
            kept.append(float(lv))
    return kept


_HOTSPOT_COLS = ['rank', 'label_id', 'peak_value', 'mean_value', 'persistence',
                 'n_voxels', 'volume_A3', 'x', 'y', 'z']

# Face-connectivity (6-neighbour) offsets for the persistence sweep.
_NEIGHBOR_OFFSETS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                     (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _voxel_size(volmap):
    '''Isotropic voxel edge length from a VolMap's spacing.'''
    return (volmap.spacing[0][0]
            if isinstance(volmap.spacing[0], (list, np.ndarray))
            else float(volmap.spacing))


def _minima_persistence(field, mask):
    '''
    Topological persistence of the local minima of a favorable field.

    Sweeps the favorable voxels (``mask``) from most favorable (most negative
    ``field``) upward with a union-find, growing one basin per local minimum.
    When a voxel bridges two basins the *shallower* one (smaller depth) dies,
    and its persistence is fixed at (its own depth - the depth at the merge
    saddle). This is 0-D persistent homology of the sublevel sets: it measures
    how deep a minimum is before it merges into a more favorable neighbor --
    i.e. the contour-depth interval over which it survives as a *separate*
    site. That is the formal version of the by-hand "diamonds within diamonds
    until they merge" ranking, and it distinguishes a real isolated pocket
    (high persistence) from a shallow dimple on the flank of a deeper basin
    (low persistence) even when the dimple's absolute value is very negative.

    Depth here is favorability = -field (positive, larger = more favorable).

    Returns
    -------
    (seeds, persistence, depth)
        seeds : (K, 3) int array, one seed voxel index per local minimum.
        persistence : (K,) float, depth-until-merge of each minimum.
        depth : (K,) float, the seed's own favorability (-field at the seed).
    '''
    fav = -np.asarray(field, dtype=float)          # favorable -> positive depth
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return np.empty((0, 3), int), np.empty(0), np.empty(0)

    order = np.argsort(-fav[mask])                 # deepest first
    coords = coords[order]
    vals = fav[mask][order]
    shape = field.shape

    comp = np.full(shape, -1, dtype=np.int64)      # component id per voxel
    parent, birth, seed = [], [], []               # union-find state per root
    persistence = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j, k), v in zip(coords, vals):
        roots = set()
        for di, dj, dk in _NEIGHBOR_OFFSETS:
            ni, nj, nk = i + di, j + dj, k + dk
            if (0 <= ni < shape[0] and 0 <= nj < shape[1]
                    and 0 <= nk < shape[2] and comp[ni, nj, nk] != -1):
                roots.add(find(int(comp[ni, nj, nk])))
        if not roots:                              # new local minimum is born
            cid = len(parent)
            parent.append(cid); birth.append(v); seed.append((i, j, k))
            comp[i, j, k] = cid
        else:                                      # join / merge basins
            survivor = max(roots, key=lambda r: birth[r])
            comp[i, j, k] = survivor
            for r in roots:
                if r != survivor:
                    persistence[r] = birth[r] - v  # shallower basin dies here
                    parent[r] = survivor

    n = len(parent)
    seeds = np.array(seed, dtype=int) if n else np.empty((0, 3), int)
    # Surviving basins never merged -> persist over their full depth.
    pers = np.array([persistence.get(r, birth[r]) for r in range(n)])
    depth = np.asarray(birth, dtype=float)
    return seeds, pers, depth


def find_hotspots(volmap, level=None, min_voxels=3, min_persistence=None,
                  top_n=None, smooth_sigma=None, connectivity=1,
                  return_labels=False):
    '''
    Segment a volume map into discrete hotspots and rank them by persistence.

    Single-threshold connected-component labeling is unsuitable for a
    near-global probe scan: the favorable region is a continuous shell over the
    receptor, so thresholding merges every pocket connected by a favorable ridge
    into one surface-spanning blob. Instead this:

      1. finds the local minima of the favorable field and their topological
         persistence (_minima_persistence),
      2. uses *all* the minima as markers for a marker-controlled watershed
         (skimage), which splits the favorable region into one basin per
         minimum -- separating pockets that share a ridge and keeping each
         basin tight (bounded by its neighbours),
      3. filters the resulting basins by size and persistence and ranks them.

    Significance filtering is applied to the basins, NOT to the markers: pruning
    markers before the watershed would let the few survivors flood the entire
    favorable region into a handful of giant basins (the very pathology this
    replaces). All minima seed the flood; ``min_persistence`` then decides which
    basins to report.

    Each hotspot carries its peak (most favorable) and mean voxel value, its
    persistence (depth-until-merge -- the significance score), its size in
    voxels and cubic Angstroms, and the world-space centroid (x, y, z). Rows are
    ranked by persistence (rank 1 = most persistent/robust site), which reflects
    "deep AND separate" rather than raw depth alone. ``label_id`` is the basin
    id in the returned label array.

    Parameters
    ----------
    level : float, optional
        Restrict segmentation to voxels with value < level. Default None uses
        all favorable (value < 0) voxels; set a negative level to focus on the
        tail.
    min_voxels : int
        Drop basins smaller than this many voxels.
    min_persistence : float, optional
        Drop basins whose persistence is below this (favorability units, i.e.
        positive REU-like depth). None keeps all.
    top_n : int, optional
        Keep only the ``top_n`` most persistent basins after filtering. On a
        near-global scan the surface has many small favorable micro-pockets, so
        capping to the most persistent handful is usually how you get a
        readable shortlist. None keeps all.
    smooth_sigma : float, optional
        Gaussian pre-smoothing (voxels). Leave None for these Boltzmann maps:
        they are already smooth, and smoothing flattens ridges, which balloons
        each watershed catchment and pulls peaks toward zero. Use a small value
        only for visibly noisy fields.
    connectivity : int
        Voxel connectivity for the watershed (1 = faces/6-neighbour,
        3 = full/26-neighbour).

    Returns
    -------
    pd.DataFrame with columns:
        rank, label_id, peak_value, mean_value, persistence, n_voxels,
        volume_A3, x, y, z
    If return_labels=True, returns (df, labels) with the (nx, ny, nz) integer
    basin-label array (0 = background / non-favorable).
    '''
    from skimage.segmentation import watershed

    field = _favorable_field(volmap, smooth_sigma)
    thr = 0.0 if level is None else float(level)
    mask = field < thr
    empty = pd.DataFrame(columns=_HOTSPOT_COLS)
    if not mask.any():
        return (empty, np.zeros(field.shape, int)) if return_labels else empty

    seeds, pers, _ = _minima_persistence(field, mask)
    if seeds.shape[0] == 0:
        return (empty, np.zeros(field.shape, int)) if return_labels else empty

    # Marker-controlled watershed from ALL minima (see docstring): pruning
    # markers first would let survivors flood the whole favorable region.
    markers = np.zeros(field.shape, dtype=np.int64)
    markers[seeds[:, 0], seeds[:, 1], seeds[:, 2]] = np.arange(1, len(seeds) + 1)
    structure = ndimage.generate_binary_structure(3, connectivity)
    labels = watershed(field, markers=markers, mask=mask, connectivity=structure)

    idx = np.arange(1, len(seeds) + 1)
    sizes = np.bincount(labels.ravel(), minlength=len(seeds) + 1)[1:]
    peaks = ndimage.minimum(field, labels, idx)
    means = ndimage.mean(field, labels, idx)
    coms = np.atleast_2d(ndimage.center_of_mass(np.ones(field.shape), labels, idx))

    origin = np.asarray(volmap.origin, dtype=float)
    v = _voxel_size(volmap)
    centers = origin + (coms + 0.5) * v            # cell lower corner + half

    df = pd.DataFrame({
        'label_id': idx,
        'peak_value': np.asarray(peaks, dtype=float),
        'mean_value': np.asarray(means, dtype=float),
        'persistence': np.asarray(pers, dtype=float),
        'n_voxels': sizes.astype(int),
        'volume_A3': sizes.astype(float) * (v ** 3),
        'x': centers[:, 0], 'y': centers[:, 1], 'z': centers[:, 2],
    })
    df = df[df['n_voxels'] >= min_voxels]
    if min_persistence is not None:
        df = df[df['persistence'] >= float(min_persistence)]
    df = df.sort_values('persistence', ascending=False)
    if top_n is not None:
        df = df.head(int(top_n))
    df = df.reset_index(drop=True)
    df.insert(0, 'rank', np.arange(1, len(df) + 1))
    df.attrs['level'] = thr
    return (df, labels) if return_labels else df


def attribute_hotspots_to_residues(volmap, receptor_pdb, level=None,
                                   min_voxels=3, min_persistence=None,
                                   top_n=None, radius=4.0, smooth_sigma=None,
                                   selection='not name *H*'):
    '''
    Attribute each ranked hotspot to the receptor residues that line it.

    This is the *geometric* attribution: for every hotspot, the receptor heavy
    atoms whose nearest hotspot voxel is within ``radius`` Angstroms are found,
    and their residues are reported ranked by proximity. It answers "which
    residues surround this hotspot", using only the map and the receptor PDB --
    no poses or PyRosetta. (Contact- and interaction-type attribution is a
    separate, ensemble-coupled step that folds into interface_energies.)

    Parameters
    ----------
    volmap : VolMap
        The probe map to segment (favorable = negative).
    receptor_pdb : str
        Receptor structure the map was built around.
    level, min_voxels, min_persistence, top_n, smooth_sigma
        Passed through to find_hotspots for segmentation. Set top_n (and/or
        min_persistence) to attribute only the most significant sites rather
        than every watershed basin.
    radius : float
        A receptor atom lines a hotspot if within this distance (A) of any of
        the hotspot's voxels.
    selection : str
        MDAnalysis selection for the receptor atoms considered (default heavy).

    Returns
    -------
    pd.DataFrame, long-format, one row per (hotspot, lining residue):
        hotspot_rank, peak_value, chain, resid, resname, min_dist, n_atoms
    ranked by hotspot then ascending min_dist. Empty if no hotspots.
    '''
    import MDAnalysis as mda
    from scipy.spatial import cKDTree

    cols = ['hotspot_rank', 'peak_value', 'chain', 'resid', 'resname',
            'min_dist', 'n_atoms']
    df, labels = find_hotspots(volmap, level=level, min_voxels=min_voxels,
                               min_persistence=min_persistence, top_n=top_n,
                               smooth_sigma=smooth_sigma, return_labels=True)
    if df.empty:
        return pd.DataFrame(columns=cols)

    # Receptor heavy-atom positions + per-atom residue identity.
    u = mda.Universe(receptor_pdb)
    atoms = u.select_atoms(selection)
    pos = atoms.positions
    resids = atoms.resids
    resnames = atoms.resnames
    try:
        chains = atoms.chainIDs
    except (AttributeError, mda.exceptions.NoDataError):
        chains = atoms.segids

    origin = np.asarray(volmap.origin, dtype=float)
    v = _voxel_size(volmap)

    rows = []
    for _, hs in df.iterrows():
        # World-space centers of this hotspot's member voxels.
        ijk = np.argwhere(labels == int(hs['label_id']))
        vox_centers = origin + (ijk + 0.5) * v
        # Nearest hotspot-voxel distance for every receptor atom.
        tree = cKDTree(vox_centers)
        dists, _ = tree.query(pos)
        within = dists <= radius
        if not np.any(within):
            continue
        lining = pd.DataFrame({
            'chain': chains[within], 'resid': resids[within],
            'resname': resnames[within], 'dist': dists[within],
        })
        agg = (lining.groupby(['chain', 'resid', 'resname'], observed=True)
                     .agg(min_dist=('dist', 'min'), n_atoms=('dist', 'size'))
                     .reset_index())
        agg.insert(0, 'peak_value', float(hs['peak_value']))
        agg.insert(0, 'hotspot_rank', int(hs['rank']))
        rows.append(agg.sort_values('min_dist'))

    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.concat(rows, ignore_index=True)[cols]
    out.attrs['level'] = df.attrs.get('level')
    out.attrs['radius'] = radius
    return out


def plot_component_sweep(volmap, min_voxels=3, smooth_sigma=None, ax=None):
    '''
    Plot distinct-hotspot count vs contour level (a tuning aid for how many
    contours to draw and where they start to bleed). Marks the peak-resolution
    level chosen by 'components' mode.
    '''
    import matplotlib.pyplot as plt

    levels, counts = component_sweep(volmap, min_voxels=min_voxels,
                                     smooth_sigma=smooth_sigma)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(levels, counts, marker='.')
    if counts.size and counts.max() > 0:
        l_peak = levels[int(np.argmax(counts))]
        ax.axvline(l_peak, color='crimson', ls='--',
                   label=f'peak @ {l_peak:.2f}')
        ax.legend()
    ax.set_xlabel('contour level (REU, favorable = negative)')
    ax.set_ylabel(f'distinct hotspots (>= {min_voxels} voxels)')
    ax.set_title('Component sweep')
    return ax


def aggregate_interface(iface, scoredata=None, models=None, by_cluster=False):
    '''
    Mean per-term interaction energy for each (glycan, protein) residue pair.

    Averaging glycan-protein interaction energies is only physically
    meaningful within a single binding mode -- across a near-global probe
    scan the glycan occupies different pockets in different poses. Pass
    by_cluster=True (with scoredata carrying a cluster_id column from
    GlycanDockEnsemble.cluster_poses) to get one matrix per cluster.

    Parameters
    ----------
    iface : pd.DataFrame
        Long-format interface energy table (GlycanDockEnsemble.interface_energies)
        with columns: model_num, glycan_label, protein_label, term, weighted.
    scoredata : pd.DataFrame, optional
        Per-pose score table indexed by model_num. Required (with a
        'cluster_id' column) when by_cluster is True.
    models : list of int, optional
        Restrict to these model_num values (e.g. top-scoring poses) before
        averaging. Defaults to all poses present in iface.
    by_cluster : bool
        Average within each pose cluster instead of pooling all poses.

    Returns
    -------
    pd.DataFrame
        Columns: [cluster_id,] glycan_label, protein_label, term, weighted
        where 'weighted' is the mean interaction energy (kcal/mol).
    '''
    df = iface if models is None else iface[iface.model_num.isin(models)]
    keys = ['glycan_label', 'protein_label', 'term']
    if by_cluster and scoredata is not None:
        df = df.merge(scoredata[['cluster_id']], left_on='model_num',
                      right_index=True)
        df = df[df['cluster_id'].notna()]
        keys = ['cluster_id'] + keys
    return df.groupby(keys, observed=True)['weighted'].mean().reset_index()


def plot_term_heatmaps(agg, terms=None, cluster=None, cmap='vlag',
                       figsize=None):
    '''
    Grid of glycan x protein heatmaps, one panel per REF15 energy term.

    Uses a diverging colormap centered at zero: favorable (negative)
    interactions are blue, unfavorable (positive) are red. The color scale
    is shared across panels so terms are directly comparable.

    Parameters
    ----------
    agg : pd.DataFrame
        Output of aggregate_interface.
    terms : list of str, optional
        Which energy terms to plot (one panel each). Defaults to all present.
    cluster : hashable, optional
        If agg has a cluster_id column, plot only this cluster.
    cmap : str
        Diverging matplotlib/seaborn colormap name.
    figsize : tuple, optional
        Overrides the auto-computed figure size.

    Returns
    -------
    matplotlib.figure.Figure
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns

    if cluster is not None and 'cluster_id' in agg.columns:
        agg = agg[agg.cluster_id == cluster]
    if terms is None:
        terms = list(pd.unique(agg['term']))

    ncols = min(3, len(terms))
    nrows = int(np.ceil(len(terms) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (4.5 * ncols, 3.5 * nrows),
                             squeeze=False)

    # Shared symmetric color scale across all term panels:
    vmax = float(np.nanmax(np.abs(agg['weighted']))) or 1.0

    for ax, term in zip(axes.flat, terms):
        mat = (agg[agg.term == term]
               .pivot_table(index='glycan_label', columns='protein_label',
                            values='weighted'))
        sns.heatmap(mat, ax=ax, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
                    cbar_kws={'label': 'kcal/mol'})
        ax.set_title(term)
        ax.set_xlabel('protein residue')
        ax.set_ylabel('glycan residue')

    # Hide any unused panels in the grid:
    for ax in axes.flat[len(terms):]:
        ax.axis('off')

    fig.tight_layout()
    return fig


def plot_interface_bars(agg):
    '''
    Newhouse-style summary: per protein residue, total interaction energy
    grouped by glycan residue and faceted by cluster (sums over all terms).

    Parameters
    ----------
    agg : pd.DataFrame
        Output of aggregate_interface. If it carries a cluster_id column the
        plot is faceted into one panel per cluster.

    Returns
    -------
    seaborn.axisgrid.FacetGrid
    '''
    import seaborn as sns

    group_cols = [c for c in agg.columns if c not in ('term', 'weighted')]
    summed = (agg.groupby(group_cols, observed=True)['weighted']
                 .sum().reset_index())

    col = 'cluster_id' if 'cluster_id' in summed.columns else None
    g = sns.catplot(data=summed, x='protein_label', y='weighted',
                    hue='glycan_label', col=col, kind='bar',
                    height=4, aspect=1.4, errorbar=None)
    g.set_axis_labels('protein residue', 'interaction energy (kcal/mol)')
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=90)
        ax.axhline(0, color='k', lw=0.5)
    g.tight_layout()
    return g