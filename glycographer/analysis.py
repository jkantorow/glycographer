'''
Analysis tools for the output of a Glycographer simulation.

These functions mostly operate on data contained within a VolMap or
GlycanDockEnsemble instance and do not require PyRosetta to execute.
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import ndimage


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


_HOTSPOT_COLS = ['rank', 'label_id', 'peak_value', 'mean_value', 'n_voxels',
                 'volume_A3', 'x', 'y', 'z']


def _voxel_size(volmap):
    '''Isotropic voxel edge length from a VolMap's spacing.'''
    return (volmap.spacing[0][0]
            if isinstance(volmap.spacing[0], (list, np.ndarray))
            else float(volmap.spacing))


def _auto_level(volmap, fav, min_voxels, smooth_sigma):
    '''Peak-resolution contour level from the component sweep (fallback: q30).'''
    levels, counts = component_sweep(volmap, min_voxels=min_voxels,
                                     smooth_sigma=smooth_sigma)
    if counts.size and counts.max() > 0:
        return float(levels[int(np.argmax(counts))])
    return float(np.percentile(fav, 30))


def find_hotspots(volmap, level=None, min_voxels=3, smooth_sigma=None,
                  return_labels=False):
    '''
    Segment a volume map into discrete hotspots and rank them.

    Thresholds the map at ``level`` (value < level), labels connected voxel
    blobs, and returns one ranked row per blob. If ``level`` is None it is
    chosen automatically as the peak-resolution level from component_sweep --
    the level at which the most distinct sites appear -- so hotspots come out
    separated rather than merged.

    Each hotspot carries its peak (most favorable) and mean voxel value, its
    size in voxels and in cubic Angstroms, and the world-space centroid (x, y, z)
    of its voxels, which locates the site on the receptor for downstream residue
    attribution. Rows are ranked by peak favorability (rank 1 = most negative).
    ``label_id`` is the id of the blob in the labeled array, so a caller with
    the labels (return_labels=True) can recover a hotspot's member voxels.

    Returns
    -------
    pd.DataFrame with columns:
        rank, label_id, peak_value, mean_value, n_voxels, volume_A3, x, y, z
    If return_labels=True, returns (df, labels) where labels is the (nx, ny, nz)
    integer label array (0 = background).
    '''
    field = _favorable_field(volmap, smooth_sigma)
    fav = field[field < 0]
    empty = pd.DataFrame(columns=_HOTSPOT_COLS)
    if fav.size == 0:
        return (empty, np.zeros(field.shape, dtype=int)) if return_labels else empty

    if level is None:
        level = _auto_level(volmap, fav, min_voxels, smooth_sigma)

    labels, n = ndimage.label(field < level)
    if n == 0:
        return (empty, labels) if return_labels else empty

    idx = np.arange(1, n + 1)
    sizes = np.bincount(labels.ravel())[1:]
    peaks = ndimage.minimum(field, labels, idx)          # most favorable voxel
    means = ndimage.mean(field, labels, idx)
    coms = ndimage.center_of_mass(field < level, labels, idx)  # (i, j, k) each

    origin = np.asarray(volmap.origin, dtype=float)
    v = _voxel_size(volmap)
    # Cell i has its lower corner at origin + i*v; use the voxel center.
    coms = np.atleast_2d(coms)
    centers = origin + (coms + 0.5) * v

    df = pd.DataFrame({
        'label_id': idx,
        'peak_value': np.asarray(peaks, dtype=float),
        'mean_value': np.asarray(means, dtype=float),
        'n_voxels': sizes.astype(int),
        'volume_A3': sizes.astype(float) * (v ** 3),
        'x': centers[:, 0], 'y': centers[:, 1], 'z': centers[:, 2],
    })
    df = df[df['n_voxels'] >= min_voxels].copy()
    df = df.sort_values('peak_value').reset_index(drop=True)
    df.insert(0, 'rank', np.arange(1, len(df) + 1))
    df.attrs['level'] = level
    return (df, labels) if return_labels else df


def attribute_hotspots_to_residues(volmap, receptor_pdb, level=None,
                                   min_voxels=3, radius=4.0, smooth_sigma=None,
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
    level, min_voxels, smooth_sigma
        Passed through to find_hotspots for segmentation.
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