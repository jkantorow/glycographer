#!/usr/bin/env python3

import pymol
from pymol import cmd
import numpy as np
import pandas as pd
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

def get_clusters(ensemble, scoredata, split_states=True):
    '''
    Create PyMOL selection objects for each pose
    in a cluster of a loaded ensemble from its
    accompanying scoredata file.
    
    Parameters:
    ensemble (str): Name of the loaded ensemble object in PyMOL
    scoredata (str): Path to the CSV file containing cluster information
    split_states (bool): If True, split ensemble into individual objects for simultaneous display
    '''
    try:
        # Read the scoredata CSV file
        data = pd.read_csv(scoredata)
        
        # Check if required columns exist
        if 'model_num' not in data.columns or 'cluster_id' not in data.columns:
            print("Error: CSV file must contain 'model_num' and 'cluster_id' columns")
            return
        
        # Show some basic info about the data
        max_model_num = data['model_num'].max()
        min_model_num = data['model_num'].min()
        total_models = len(data)
        print(f"Scoredata info: {total_models} models, model numbers {min_model_num}-{max_model_num}")
        
        # Check ensemble info
        ensemble_info = cmd.count_states(ensemble)
        print(f"Ensemble '{ensemble}' has {ensemble_info} states")
        
        if ensemble_info < max_model_num:
            print(f"Warning: Ensemble has {ensemble_info} states but scoredata references up to model {max_model_num}")
        
        # Get unique cluster IDs (excluding NaN/empty values)
        cluster_ids = data['cluster_id'].dropna().unique()
        
        if len(cluster_ids) == 0:
            print("No clusters found in the scoredata file")
            return
        
        if split_states:
            # Split the ensemble into individual objects for each state
            print("Splitting ensemble into individual state objects...")
            cmd.split_states(ensemble)
            
            # Get list of all objects created by split_states to verify which ones exist
            all_objects = cmd.get_names("objects")
            available_state_objects = [obj for obj in all_objects if obj.startswith(f"{ensemble}_")]
            
            print(f"Available state objects: {len(available_state_objects)}")
            
            # Create selection objects for each cluster using individual state objects
            for cluster_id in cluster_ids:
                if pd.isna(cluster_id) or cluster_id == '':
                    continue
                    
                # Get model numbers for this cluster
                cluster_models = data[data['cluster_id'] == cluster_id]['model_num'].tolist()
                
                # Create PyMOL selection string using individual state objects
                selection_name = f"cluster_{int(cluster_id)}"
                
                # Build selection string for all individual state objects in this cluster
                # Only include state objects that actually exist
                state_objects = []
                missing_objects = []
                
                for model in cluster_models:
                    obj_name = f"{ensemble}_{str(model).zfill(4)}"
                    if obj_name in available_state_objects:
                        state_objects.append(obj_name)
                    else:
                        missing_objects.append(obj_name)
                
                if missing_objects:
                    print(f"Warning: Missing state objects for cluster {int(cluster_id)}: {', '.join(missing_objects)}")
                
                if not state_objects:
                    print(f"Error: No valid state objects found for cluster {int(cluster_id)}")
                    continue
                
                selection_string = ' or '.join(state_objects)
                
                # Create the selection object
                cmd.select(selection_name, selection_string)
                
                # Set different colors for each cluster for better visualization
                color_index = int(cluster_id) % 20  # Cycle through PyMOL's built-in colors
                cmd.color(f"auto", selection_name)
                
                print(f"Created selection '{selection_name}' with {len(state_objects)} poses")
                if len(state_objects) <= 10:  # Only show object names for smaller clusters
                    print(f"  State objects: {', '.join(state_objects)}")
                else:
                    print(f"  State objects: {state_objects[0]}, {state_objects[1]}, ... (and {len(state_objects)-2} more)")
        
        else:
            # Original method - create selections from multi-state object
            print("Creating selections from multi-state ensemble...")
            for cluster_id in cluster_ids:
                if pd.isna(cluster_id) or cluster_id == '':
                    continue
                    
                # Get model numbers for this cluster
                cluster_models = data[data['cluster_id'] == cluster_id]['model_num'].tolist()
                
                # Create PyMOL selection string
                selection_name = f"cluster_{int(cluster_id)}"
                
                # Build selection string for all models in this cluster
                state_list = ','.join(map(str, cluster_models))
                selection_string = f"{ensemble} and state {state_list}"
                
                # Create the selection object
                cmd.select(selection_name, selection_string)
                
                print(f"Created selection '{selection_name}' with {len(cluster_models)} poses (states: {state_list})")
        
        print(f"Successfully created selections for {len(cluster_ids)} clusters")
        
        if split_states:
            print("\nTo display all poses in a cluster simultaneously:")
            print("  show cartoon, cluster_1")
            print("  show cartoon, cluster_2")
            print("  etc.")
            print("\nTo hide the original ensemble object:")
            print(f"  hide everything, {ensemble}")
        
    except FileNotFoundError:
        print(f"Error: Could not find scoredata file: {scoredata}")
    except Exception as e:
        print(f"Error processing clusters: {str(e)}")

# Extend PyMOL command set
cmd.extend("get_clusters", get_clusters)

def show_clusters(representation="cartoon", hide_original=True):
    '''
    Convenience function to display all cluster selections simultaneously.
    
    Parameters:
    representation (str): PyMOL representation type (cartoon, sticks, spheres, etc.)
    hide_original (bool): Whether to hide the original ensemble object
    '''
    # Get all objects that start with "cluster_"
    all_objects = cmd.get_names("objects")
    cluster_objects = [obj for obj in all_objects if obj.startswith("cluster_")]
    
    if not cluster_objects:
        print("No cluster selections found. Run get_clusters first.")
        return
    
    # Show all cluster selections
    for cluster in cluster_objects:
        cmd.show(representation, cluster)
        print(f"Showing {cluster} as {representation}")
    
    # Hide the original ensemble if requested
    if hide_original:
        ensemble_objects = [obj for obj in all_objects if not obj.startswith("cluster_") and "_" in obj]
        for obj in ensemble_objects:
            if any(obj.startswith(cluster.replace("cluster_", "")) for cluster in cluster_objects):
                continue  # Don't hide individual state objects
            cmd.hide("everything", obj)
            print(f"Hiding original ensemble: {obj}")

cmd.extend("show_clusters", show_clusters)