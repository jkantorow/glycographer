'''
Dataclasses and tools for glycan conformation sampling during docking.
'''

from dataclasses import dataclass, field
import os

@dataclass(frozen=True)
class GlycanDockConfig:

    # Full protocol parameters:
    partners: str = 'A_X' # Set the chain IDs identifying the protein-glycoligand complex protein is chain A, glycoligand is chain X –> partners = A_X
    refine_only: bool = False # Skip the Stage 1 initial perturbation and employ only the uniform ± 15° glycosidic torsion angle perturbations using the SmallBBSampler during Stage 2 docking and refinement
    prepack_only: bool = False # Perform only Stage 0 pre-packing
    intf_pack_dist: float = 16.0 # (Angstroms) Cutoff distance to define protein-glycan interface residues to be considered during packing
    slide: bool = True # Allow the glycoligand’s center-of-mass to be periodically translated toward the protein receptor’s center-of-mass during rigid-body sampling
    rand_glyc_jump: bool = False # Set whether to use a random, non-branch-point residue of the glycoligand to set as the Jump residue for the docking FoldTree

    # Stage 1-specific parameters:
    stage1_rand_rot: bool = False # During Stage 1 initial perturbation, randomly rotate the glycoligand about its center-of-mass in 360˚ space
    stage1_trans_mag: float = 0.5 # (Angstroms) The translational magnitude of the Gaussian perturbation on the glycoligand’s center-of-mass performed during Stage 1
    stage1_rot_mag: float = 7.5 # (Degrees) The rotational magnitude of the Gaussian perturbation on the glycoligand’s center-of-mass performed during Stage 1
    stage1_tor_mag: float = 12.5 # (Degrees) The magnitude of the uniform perturbation on each glycosidic torsion angle
    
    # Stage 2-specific parameters:
    n_repeats: int = 3 # Number of times to repeat Stage 2 of GlycanDock if the final model does not have a negative interface score
    n_cycles: int = 10 # Number of stage 2 outer cycles to perform
    rb_rounds: int = 8 # The number of inner cycles of rigid-body perturbations to perform
    tor_rounds: int = 8 # The number of inner cycles of glycosidic torsion perturbations to perform
    stage2_trans_mag: float = 0.5 # (Angstroms) During Stage 2 docking and refinement, this is the magnitude for performing a translational Gaussian perturbation on the glycoligand's center-of-mass
    stage2_rot_mag: float = 7.5 # (Degrees) During Stage 2 docking and refinement, this is the magnitude for performing a rotational Gaussian perturbation on the glycoligand's center-of-mass 
    full_pack_freq: int = 8 # When packing after each sampling perturbation in Stage 2, at what frequency should the PackRotamersMover be called? Set to 0 to use only the EnergyCutRotamerTrialsMover after every perturbation. Set to the value of n_rigid_body_rounds and n_torsion_rounds to use the PackRotamersMover only after the last perturbation
    mc_kt: float = 0.6 # (a.u.) During Stage 2 docking and refinement, the value of kT used to accept or reject moves based on the Metropolis criterion
    ramp_sf: bool = True # Set whether to ramp the fa_atr and fa_rep score terms of the ScoreFunction used during Stage 2 sampling and optimization