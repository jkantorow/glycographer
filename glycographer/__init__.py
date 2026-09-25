'''
glycographer: simulation tools and pipeline scaffolding for receptor-glycoligand
binding affinity maps.

Submodules are deliberately NOT imported here. Several of them pull in heavy or
platform-specific dependencies -- pymol (grid, vis, dock), open3d (grid), and
pyrosetta (sample, and one method on dock) -- and importing them eagerly would
make a bare `import glycographer` fail on any machine missing one of them,
including the Windows workstation used for development and any analysis-only
environment.

Import what you need explicitly:

    from glycographer.sample import GlycanDockConfig
    from glycographer.dock import GlycanDockEnsemble
'''

__version__ = '0.1.0'
