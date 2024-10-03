#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# Function to extract coordinates from the docking log (DLG) file
def extract_coordinates(content):
    """Extract coordinates of each docking pose from the log file."""
    poses = []
    current_pose = []
    for line in content:
        if "DOCKED: MODEL" in line:
            if current_pose:
                poses.append(np.array(current_pose))
                current_pose = []
        if "DOCKED: ATOM" in line:
            # Extract the x, y, z coordinates using regex
            match = re.search(r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", line)
            if match:
                x, y, z = map(float, match.groups())
                current_pose.append([x, y, z])
    if current_pose:  # Append the last pose
        poses.append(np.array(current_pose))
    
    return poses

# Function to calculate RMSD between all pairs of poses
def calculate_rmsd(poses):
    """Calculate RMSD between each pair of poses."""
    n_poses = len(poses)
    rmsd_matrix = np.zeros((n_poses, n_poses))

    for i in range(n_poses):
        for j in range(i + 1, n_poses):
            diff = poses[i] - poses[j]
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            rmsd_matrix[i, j] = rmsd_matrix[j, i] = rmsd
    
    return rmsd_matrix

# Function to process all .dlg files in a folder and save their RMSD distributions
def process_dlg_files(folder_path):
    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".dlg"):
            file_path = os.path.join(folder_path, file_name)
            
            # Read the content of the current DLG file
            with open(file_path, 'r') as file:
                content = file.readlines()
            
            # Extract the coordinates from the file
            poses = extract_coordinates(content)
            
            # Calculate the RMSD matrix
            rmsd_matrix = calculate_rmsd(poses)
            
            # Flatten the upper triangular part of the RMSD matrix
            rmsd_values = rmsd_matrix[np.triu_indices_from(rmsd_matrix, k=1)]
            
            # Generate and save the histogram for the RMSD distribution
            plt.figure(figsize=(8, 6))
            plt.hist(rmsd_values, bins=np.arange(0, np.max(rmsd_values) + 0.5, 0.5), color='blue', edgecolor='black')
            plt.title(f'Distribution of RMSD Values for {file_name}')
            plt.xlabel('RMSD (Å)')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # Save the figure with the same base name as the DLG file
            output_file = os.path.join(folder_path, f'{file_name}_rmsd_distribution.png')
            plt.savefig(output_file)
            plt.close()

# Folder containing the .dlg files
folder_path = '/home/paturel/Bureau/rmsd-integrin/4G1M'  # Change this to your folder path

# Process all .dlg files in the folder
process_dlg_files(folder_path)

