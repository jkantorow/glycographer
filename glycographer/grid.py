from dataclasses import dataclass, field
import sys

from pymol import cmd
import open3d as o3d
import numpy as np

'''
Classes and functions to manage gridbox generation around a receptor
'''