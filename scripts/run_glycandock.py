#!/usr/bin/env python3

'''
Perform a full glycandocking protocol between a receptor and
glycoligand.

- By default, the script can be used to simply generate n predicted
  docked structures between a receptor target and a single, manually
  positioned glycoligand. Keep in mind that Rosetta's GlycanDock
  protocol is intended for bound glycan refinement and as such has
  an effective docking range of 7 Angstroms or less -- meaning that
  the input pose's ring RMSD compared to the known/theoretical "true"
  binding pose is assumed to be at most 7 Angstroms. Because of this,
  while glycosidic torsions are exhaustively sampled during docking,
  rigid body moves are only sparingly sampled.

- The main function of this script is to output an ensemble of complexes
  between a receptor and glycoligand fragment(s) over a predetermined
  set of spatial sampling positions across a putative binding site on
  the receptor target. If a sampling grid is provided along with the
  receptor and glycoligand fragment(s) of interest, docking will be
  performed for:
    - l number of ligand fragments
    - p number of sampling points in the provided grid
    - n number of output structures for each sampling point
'''

from pyrosetta import init, pose_from_pdb, Vector1
from pyrosetta.rosetta.protocols.rigid import RigidBodyRandomizeMover, partner_downstream
from pyrosetta.rosetta.protocols.docking import setup_foldtree
from pyrosetta.rosetta.protocols.ligand_docking import StartFrom
from pyrosetta.rosetta.protocols.glycan_docking import GlycanDockProtocol
import argparse
import os

def init_glycandock(complex, nstruct=1, mc_cycles=1, options=None):
    '''
    Start Rosetta session with necessary structure and
    options input.
    '''
    in_flags = f'''
    -in:file:s {complex}
    -nstruct {nstruct}
    -n_cycles {mc_cycles}
    '''
    if not options:
        options = '''
        -include_sugars
        -auto_detect_glycan_connections
        -alternate_3_letter_codes pdb_sugar
        -write_pdb_link_records
        -maintain_links
        -docking:partners A_X

        -ex1
        -ex2

        -ignore_unrecognized_res
        -ignore_zero_occupancy false
        -load_PDB_components false
        -no_fconfig
        '''
        options = ' '.join(options.split('\n'))
        init(f'{in_flags} {options}')
    else:
        init(f'{in_flags} @{options}')
    
# How do we best define the movers for startfrom, prepack, and full protocol?
# Do we just run the process mostly in the main script?
    # I'm not seeing great ways to wrap a lot of this into functions.
# Better established outfile naming convention (rec_lig_startpoint_iter)?

def main():
    
    parser = argparse.ArgumentParser(description='' \
    'Output predicted docking poses between a receptor and glycoligand' \
    'or glycoligand fragments. Scored by Rosetta GlycanDockProtocol mover' \
    'which uses the REF15 scoring function.')

    parser.add_argument('complex', type=str,
                        help='Filename of the receptor to analyze in PDB format. Receptor and glycoligand chain id' \
                        'should be A and X respectively if glycoligand is already included in the receptor pdb. If no' \
                        'chain id X is found, include the glycoligand structure as a pdb file with the -lig flag.')
    parser.add_argument('-n', '--nstruct', type=int, default=1,
                        help='The desired number of sampled and scored poses to output in PDB format (default: 1)')
    parser.add_argument('-grid', '--meshgrid', type=str, required=False,
                        help='Filename of the sampling grid to use for energy landscape mapping (Not required if refining a predetermined input pose).')
    parser.add_argument('-o', '--outprefix', type=str, required=False,
                        help='Prefix appended to the beginning of all output files.')
    parser.add_argument('--no-random-start', action='store_true',
                        help='Specify to not randomize input glycoligand orientation before running GlycanDockProtocol')
    parser.add_argument('--start-count-from', type=int, default=1,
                        help='Numeric id value to start output complex files from (use for checkpointing from a previous run) (default: 1).')
    parser.add_argument('--refine-only', action='store_true',
                        help='Only perform GlycanDockProtocol stage 2 on input structure (prepacking is performed regardless).')
    parser.add_argument('--options', type=str, default=None,
                        help='Rosetta input options to provide as a file or as a string of flags (default flags are used for glycan docking if not provided).')
    parser.add_argument('--mc-cycles', type=int, default=1,
                        help='Number of Monte Carlo cycles to perform before outputting the final structure for each instance of GlycanDock' \
                        'Note: the GlycanDockProtocol mover only outputs structures with an interface score less than 0 by default' \
                        'and uses a filter to automatically repeat the protocol up to three times if a positive interface' \
                        'score is obtained before sampling a new pose. So, for example, if you specify a cycle number of 3,' \
                        'the protocol might re-execute up to 9 times if repeated unfavorable scores are obtained.')
    
    args = parser.parse_args()

    if args.nstruct < 1:
        raise ValueError('Number of output structures must be at least 1.')
    if args.start_count_from < 1:
        raise ValueError('Output pose count must begin at least from 1.')

    # Define GlycanDockProtocol Stage 1 parameters:
    s1_rot_com = False if args.refine_only else True # Whether or not to rotate the glycoligand around its com during stage 1
    s1_t_mag = 0.5 # Angstroms max (?) from start coord
    s1_r_mag = 180.0 if args.meshgrid else 7.5 # Degrees max (?) about start coord in roh, theta, or phi (?)
    s1_tor_pert_mag = 12.5 # Degrees max (?) each glycosidic linkage is torsionally rotated

    # Specify how GlycanDockProtocol will perform Monte Carlo moves:
    n_repeats = 3 # Number of protocol repeats to perform on a Monte Carlo pose if interaction energy score is > 0
    mc_kt = 0.6 # Statistical value representing the extent of Monte Carlo randomization (?)

    # Define GlycanDockProtcol Stage 2 parameters:
    s2_only = True if args.refine_only else False # Bypass stage 1 sampling
    s2_t_mag = 0.2 if args.meshgrid else 0.5 # Angstroms max (?) from pose com obtained after stage 1
    s2_r_mag = 45.0 if args.meshgrid else 7.5 # Degrees max (?) about pose com obtained after stage 1
    n_rb_rounds = 20 if args.meshgrid else 8 # Number of rigid body translation/rotation MC moves to perform during stage 2
    n_tor_rounds = 20 # Number of torsional MC moves to perform during stage 2

    # Specify how Rosetta will prepack each structure and deploy the score function: 
    full_pack_freq = 8 # How many times to apply the full repacking protocol via PackRotamersMover (higher accuracy / higher computational expense)
    interface_pack_dist = 16.0 # Distance between rec and lig defined as interface for sidechain repacking (?)
    ramp_sfxn = True # Alter REF15 params for stage 2 based on FlexPepDock accuracy

    # We have to init Rosetta for each unique receptor-ligand combination:
        
    init_glycandock(args.complex, nstruct=args.nstruct, mc_cycles=args.mc_cycles, options=args.options)

    # Instantiate StartFrom mover if a docking grid is provided:
    if args.meshgrid:
        start_from_grid = StartFrom()
        start_from_grid.chain('X')
        start_from_grid.parse_pdb_file(args.meshgrid)
    
    # Instantiate prepacking mover:
    gdock_prepack = GlycanDockProtocol()
    gdock_prepack.set_partners('A_X')
    gdock_prepack.set_prepack_only(True)
    gdock_prepack.set_slide_glycan_into_contact(True)

    # Instantiate full protocol mover:
    gdock_full = GlycanDockProtocol()
    gdock_full.set_partners('A_X')
    gdock_full.set_refine_only(s2_only)
    gdock_full.set_stage1_rotate_glycan_about_com(s1_rot_com)
    gdock_full.set_stage1_perturb_glycan_com_trans_mag(s1_t_mag)
    gdock_full.set_stage1_perturb_glycan_com_rot_mag(s1_r_mag)
    gdock_full.set_stage1_torsion_uniform_pert_mag(s1_tor_pert_mag)
    gdock_full.set_n_repeats(n_repeats)
    gdock_full.set_mc_kt(mc_kt)
    gdock_full.set_n_rigid_body_rounds(n_rb_rounds)
    gdock_full.set_n_torsion_rounds(n_tor_rounds)
    gdock_full.set_stage2_trans_mag(s2_t_mag)
    gdock_full.set_stage2_rot_mag(s2_r_mag)
    gdock_full.set_full_packing_frequency(full_pack_freq)
    gdock_full.set_interface_packing_distance(interface_pack_dist)
    gdock_full.set_ramp_scorefxn(ramp_sfxn)
    gdock_full.set_slide_glycan_into_contact(True)

    # Run each mover for n specified output structures:
    for i in range(int(args.start_count_from), (int(args.nstruct)+int(args.start_count_from))):
        
        # Load the complex into Rosetta as a pose from the input PDB:
        complex_pose = pose_from_pdb(args.complex)
        
        # Move the glycoligand to a starting coordinate if a grid was specified:
        if args.meshgrid:
            start_from_grid.apply(complex_pose)
        
        # Randomize the orientation of the glycoligand before docking if specified:
        if args.no_random_start:
            pass
        else:
            setup_foldtree(complex_pose, "A_X", Vector1([1])) # assign the glycoligand jump number to 1
            randomize = RigidBodyRandomizeMover(complex_pose, 1, partner_downstream)
            randomize.apply(complex_pose)
        
        # Prepack the receptor in context of the oriented glycoligand:
        gdock_prepack.apply(complex_pose)
        
        # Run the full glycan docking protocol on the prepacked complex:
        gdock_full.apply(complex_pose)

        # Extract the sampled pose as a pdb file:
        rec_lig = os.path.basename(args.complex).replace('.pdb', '')
        if not args.outprefix or args.outprefix == "":
            outname = f'{rec_lig}_{str(i).zfill(4)}.pdb'
        else:
            outname = f'{args.outprefix}_{str(i).zfill(4)}.pdb'
        complex_pose.dump_pdb(outname)

        # 1. Need to think about how to make sure this is working
        # the same way as what is specified by the xml protocol.
        # 2. Would it make more sense to generate rotations manually
        # instead of relying on the random rb rotation moves?
        # - 2a. This would allow us to specify the rot sampling more
        #   robustly.
        # 3. StartFrom still chooses coords randomly: can we use a
        # different method to specify exactly how many times each
        # grid point should be sampled and at what rb rotations?
        # 4. Can any of this be refactored into functions?

if __name__ == '__main__':
    exit(main())