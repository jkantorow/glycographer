'''
Analysis tools for the output of a Glycographer simulation.

These functions mostly operate on data contained within a VolMap or
GlycanDockEnsemble instance and do not require PyRosetta to execute.
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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