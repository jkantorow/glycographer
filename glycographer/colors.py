'''
Glycan / probe coloring utilities.

Two distinct coloring jobs are supported and deliberately kept separate:

1. Chemical-identity coloring (SNFG). ``get_snfg_color`` maps a single glycan
   residue name to its Symbol Nomenclature for Glycans (SNFG) color. Use this
   for crystal ligands and other single-molecule, single-residue contexts where
   SNFG faithfulness is the convention. Note that SNFG encodes identity by
   color *and* shape; a surface/mesh has no shape channel, so residues that
   share a color (e.g. GLC and NAG are both blue) are not distinguishable here.

2. Probe-comparison coloring (categorical). ``probe_palette`` assigns each probe
   in a run a maximally-distinct, colorblind-safe color (Okabe-Ito). Use this
   for best_probe surfaces and multi-probe mesh overlays, where perceptual
   separation matters more than chemical mnemonic. Do NOT color a probe overlay
   by SNFG (two blues / two greens defeat the comparison), and do NOT mix
   constituent colors for composite probes (blue GLC + yellow GAL mixes to green
   and reads as MAN).

This module is pure (matplotlib + colorsys only); it imports no PyMOL and can be
used from analysis code, notebooks, or CLI scripts.
'''

import colorsys
from typing import Dict, List, Sequence, Tuple

import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# SNFG chemical-identity colors (single residue -> PyMOL color name)
# ---------------------------------------------------------------------------
SNFG_COLORS: Dict[str, str] = {
    'MAN': 'green',
    'BMA': 'green',
    'GLC': 'blue',
    'GAL': 'yellow',
    'NGA': 'yellow',
    'A2G': 'yellow',
    'GUL': 'orange',
    'NAG': 'blue',
    'BGC': 'blue',
    'NDG': 'blue',
    'FUC': 'red',
    'SIA': 'magenta',
    'XYL': 'orange',
    'ALL': 'purple',
    'ALT': 'pink',
    'RHA': 'pink',
    'ARA': 'green',
    'RIB': 'blue',
}


def get_snfg_color(resname: str, default: str = 'green') -> str:
    '''Return the SNFG PyMOL color name for a glycan residue (case-insensitive).'''
    return SNFG_COLORS.get(resname.upper(), default)


# ---------------------------------------------------------------------------
# Categorical probe palette (colorblind-safe, for probe comparison)
# ---------------------------------------------------------------------------
# Okabe-Ito qualitative palette: eight colors chosen for distinguishability
# under the common forms of color-vision deficiency. Grey is reserved for
# "no / undefined probe" (e.g. best_probe index 0).
OKABE_ITO: List[str] = [
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
    '#000000',  # black
]
NO_PROBE_COLOR = '#BBBBBB'  # grey: voxels no probe visits


def probe_palette(labels: Sequence[str]) -> Dict[str, Tuple[float, float, float]]:
    '''
    Assign each probe label a distinct, colorblind-safe RGB color.

    Colors are assigned by position in ``labels`` (deterministic) and cycle
    through the Okabe-Ito palette if there are more than eight probes. Returns
    a {label: (r, g, b)} dict with fractional RGB in [0, 1], ready for
    ``cmd.set_color`` in PyMOL or a matplotlib legend.
    '''
    palette = {}
    for i, label in enumerate(labels):
        hexcol = OKABE_ITO[i % len(OKABE_ITO)]
        palette[label] = mcolors.to_rgb(hexcol)
    return palette


# ---------------------------------------------------------------------------
# Magnitude scaling (color a value by its position in a score range)
# ---------------------------------------------------------------------------
def color_by_magnitude(base_color, score: float, score_range,
                       negative_is_better: bool = True
                       ) -> Tuple[float, float, float]:
    '''
    Scale a base color by a score's position within a range: the base color at
    full strength for the most favorable score, fading to white for the least.

    This colors a single flat object (e.g. one isomesh contour drawn at one
    isolevel) by the strength that level represents -- something a two-color
    ``ramp_new`` does not do, since a ramp colors a surface by an *underlying*
    per-vertex map value, whereas an isomesh is a single-value object.

    Implemented by sampling a white -> base_color colormap, which is more robust
    than hand-rolled HLS math and matches the intent: t = 1 places the full base
    color at the favorable end.

    Parameters
    ----------
    base_color : str or RGB tuple
        Any matplotlib-recognized color (PyMOL color names like 'red', 'blue'
        resolve fine).
    score : float
        The value to color.
    score_range : (min, max)
        The (min, max) bounds the score is scaled within.
    negative_is_better : bool
        If True, the range minimum is the favorable (full-color) end; if False,
        the maximum is.

    Returns
    -------
    (r, g, b) : fractional RGB in [0, 1].
    '''
    score_min, score_max = score_range[0], score_range[1]
    span = score_max - score_min
    ratio = 0.0 if span == 0 else (score_max - score) / span

    # t = 1 -> full base color (favorable end), t = 0 -> white.
    t = ratio if negative_is_better else (1.0 - ratio)
    t = float(min(1.0, max(0.0, t)))

    cmap = LinearSegmentedColormap.from_list('magnitude', ['white', base_color])
    r, g, b, _ = cmap(t)
    return (r, g, b)


def glycolor_by_magnitude(resname: str, score: float, score_range,
                          negative_is_better: bool = True
                          ) -> Tuple[float, float, float]:
    '''
    ``color_by_magnitude`` keyed on a residue's SNFG base color. Convenience for
    coloring a single-residue contour/patch by its SNFG hue and strength.
    '''
    return color_by_magnitude(get_snfg_color(resname), score, score_range,
                              negative_is_better)


def rgb_to_hex(r: float, g: float, b: float) -> str:
    '''Fractional RGB in [0, 1] -> '#rrggbb' hex string.'''
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
