#!/usr/bin/env python3

'''
Tools for quickly and easily applying standard glycan
coloring and gradients based on SNFG format in visualization
software such as pymol and vmd.
'''

import os
import json
import matplotlib.colors as mcolors
import colorsys
from typing import List, Dict, Optional, Tuple

def load_glycolor_schemes() -> Dict:
    '''
    Load JSON data for use in python:
    '''
    glycolor_data = os.path.join(os.path.dirname(__file__), 'glycan_color_schemes.json')
    with open(glycolor_data, 'r') as f:
        return json.load(f)
    
def get_snfg_color(resname: str, mode: str = 'pymol') -> str:
    '''
    Return a color approximately following SNFG standard for
    the input glycan residue name.
    '''
    palettes = load_glycolor_schemes()
    colors = palettes['snfg_colors'][mode]
    return colors.get(resname.upper(), 'green' if mode == 'pymol' else 7)

def color_by_magnitude(base_color: str, score: float, score_range, 
                       negative_is_better: bool = True) -> Tuple[float, float, float]:
    '''
    Return the rgb values of a residue name's SNFG base color
    scaled by saturation/lightness depending on a corresponding
    input score value's position within a score range.
    '''
    # Take the pymol color name and translate it into hls:
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Scale the saturation value by the relative score magnitude:
    score_min = score_range[0]
    score_max = score_range[1]
    ratio = (score_max - score) / (score_max - score_min)

    if negative_is_better:
        s = s * ratio
        l = 0.90 - (l * ratio)
    else:
        s = s * (1 - ratio)
        l = 0.10 + (l * ratio)
        
    # Convert the saturation-scaled color back to rgb format:
    r_scaled, g_scaled, b_scaled = colorsys.hls_to_rgb(h, l, s)

    return (r_scaled, g_scaled, b_scaled)

def glycolor_by_magnitude(resname: str, score: float, score_range, 
                           negative_is_better: bool = True) -> Tuple[float, float, float]:
    '''
    Return the rgb values of a residue name's SNFG base color
    scaled by saturation/lightness depending on a corresponding
    input score value's position within a score range.
    '''
    base_color = get_snfg_color(resname=resname, mode='pymol')

    # Take the pymol color name and translate it into hls:
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Scale the saturation value by the relative score magnitude:
    score_min = score_range[0]
    score_max = score_range[1]
    ratio = (score_max - score) / (score_max - score_min)

    if negative_is_better:
        s = s * ratio
        l = 0.90 - (l * ratio)
    else:
        s = s * (1 - ratio)
        l = 0.10 + (l * ratio)
        
    # Convert the saturation-scaled color back to rgb format:
    r_scaled, g_scaled, b_scaled = colorsys.hls_to_rgb(h, l, s)

    return (r_scaled, g_scaled, b_scaled)

def atomcolor_by_magnitude(element: str, score: float, score_range,
                           negative_is_better: bool = False) -> Tuple[float, float, float]:
    '''
    Return a custom color for the input atom name and relative score
    in rgb format.
    '''
    elem_cols = {
        'C' : 'black',
        'O' : 'red',
        'N' : 'blue',
    }

    base_color = elem_cols.get(element.upper(), 'orange')

    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Scale the saturation value by the relative score magnitude:
    score_min = score_range[0]
    score_max = score_range[1]
    ratio = (score_max - score) / (score_max - score_min)

    if negative_is_better:
        s = s * ratio
        l = 0.90 - (l * ratio)
    else:
        s = s * (1 - ratio)
        l = 0.10 + (l * ratio)
        
    # Convert the saturation-scaled color back to rgb format:
    r_scaled, g_scaled, b_scaled = colorsys.hls_to_rgb(h, l, s)

    return (r_scaled, g_scaled, b_scaled)

def rgb_to_255(r, g, b):
    '''
    Helper function for translating between fractional rgb output
    and integers between 0 and 255 for better vmd compatibility.
    '''
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
