#!/usr/bin/env python3

'''
DEPRECATED shim. The glycan/probe coloring utilities now live in the package at
``glycographer.colors``. This module re-exports them so older scripts/notebooks
keep working; import from ``glycographer.colors`` in new code.
'''

import os
import sys
from typing import Dict, Tuple

import matplotlib.colors as mcolors
import colorsys

# Make the in-development package importable (repo not pip-installed).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from glycographer.colors import (  # noqa: F401  (re-exported for back-compat)
    SNFG_COLORS,
    color_by_magnitude,
    glycolor_by_magnitude,
    probe_palette,
    rgb_to_hex,
)
from glycographer.colors import get_snfg_color as _get_snfg_color


def get_snfg_color(resname: str, mode: str = 'pymol') -> str:
    '''Back-compat wrapper. VMD coloring is on hiatus; only 'pymol' is supported.'''
    if mode != 'pymol':
        print(f"Warning: get_snfg_color mode {mode!r} is no longer supported "
              "(VMD on hiatus); returning the PyMOL color.")
    return _get_snfg_color(resname)


def load_glycolor_schemes() -> Dict:
    '''
    Back-compat wrapper. The JSON scheme file has been retired in favor of the
    inlined table in glycographer.colors; this reconstructs the old structure.
    '''
    return {'snfg_colors': {'pymol': dict(SNFG_COLORS)}}


def atomcolor_by_magnitude(element: str, score: float, score_range,
                           negative_is_better: bool = False
                           ) -> Tuple[float, float, float]:
    '''Color an atom by element and relative score magnitude (legacy helper).'''
    elem_cols = {'C': 'black', 'O': 'red', 'N': 'blue'}
    base_color = elem_cols.get(element.upper(), 'orange')
    return color_by_magnitude(base_color, score, score_range, negative_is_better)


def rgb_to_255(r, g, b):
    '''Legacy alias for rgb_to_hex.'''
    return rgb_to_hex(r, g, b)
