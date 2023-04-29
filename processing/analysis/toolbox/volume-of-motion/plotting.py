from scipy.spatial import ConvexHull
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from ellipsoid_fit_modified import ellipsoid_fit

def ellipsoidGen(coords):
    # compute the convex hull of the points
    hull = ConvexHull(coords)

    # extract the vertices of the convex hull
    vertices = coords[hull.vertices,:]

    # fit ellipsoid
    center, evecs, radii, v = ellipsoid_fit(vertices)
    radii *= 1.2 # for better visualization

    # Create the ellipsoid mesh
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x_e = radii[0] * np.outer(np.cos(u), np.sin(v))
    y_e = radii[1] * np.outer(np.sin(u), np.sin(v))
    z_e = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x_e)):
        for j in range(len(x_e)):
            [x_e[i, j], y_e[i, j], z_e[i, j]] = np.dot([x_e[i, j], y_e[i, j], z_e[i, j]], evecs) + center
            
    return x_e, y_e, z_e
    
def extractCoordinates(filepath, armIsLeft):
    
    df = pd.read_csv(filepath)
        
    # extract positions (left / right)
    if armIsLeft:
        X = df[['SLTx']].values
        Y = df[['SLTy']].values
        Z = df[['SLTz']].values
    else: 
        X = df[['SRTx']].values
        Y = df[['SRTy']].values
        Z = df[['SRTz']].values

    # assume (x,y,z) is either all constants or all NaN
    x = [c[0] for c in X if not math.isnan(c[0])]
    y = [c[0] for c in Y if not math.isnan(c[0])]
    z = [c[0] for c in Z if not math.isnan(c[0])]

    return x, y, z

# later merge this with areaOfMotionAnalysis to reduce redundancy
def mergeTrajectories(residentFilePath, surgeonFilePath, armIsLeft):
    
    # extract coordinates
    x_r, y_r, z_r = extractCoordinates(residentFilePath, armIsLeft)
    x_s, y_s, z_s = extractCoordinates(surgeonFilePath, armIsLeft)
    
    # combine the arrays into a 3d numpy array
    all_coords_r = np.vstack((x_r, y_r, z_r)).T
    all_coords_s = np.vstack((x_s, y_s, z_s)).T
    
    # calculate distances between all consecutive points
    distances_r = np.sqrt(np.sum(np.diff(all_coords_r, axis=0)**2, axis=1))
    distances_s = np.sqrt(np.sum(np.diff(all_coords_s, axis=0)**2, axis=1))

    # find the total length of the trajectory
    total_length_r = np.sum(distances_r)
    total_length_s = np.sum(distances_s)
    
    ax = ax2 if armIsLeft else ax5
    
    # find percentage difference
    percentageDifference = ((total_length_r/total_length_s) - 1) * 100
    
    # add a text annotation with the total length
    if percentageDifference > 0:
        ax.text2D(0.5, 0.00, f'You have moved {round(percentageDifference)}%\nmore than needed.', ha='center', transform=ax.transAxes, fontsize=10)
    else:
        ax.text2D(0.5, 0.00, f'Contragulations! You have moved {round(-percentageDifference)}%\nless than needed.', ha='center', transform=ax.transAxes, fontsize=10)
        
    # plot the trajectory
    ax.plot(x_r, y_r, z_r, color='red', label='Your Trajectory', linewidth=1)
    ax.plot(x_s, y_s, z_s, color='blue', label='Ideal Trajectory', linewidth=0.5, alpha=0.5) 
    
    # # Set the plot labels
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # Hide the 3D axes box
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()

    # Add a legend
    ax.legend(loc='upper right', fontsize=8)
    
def extract_top_eighty(x,y,z):
    # centroid for ellipsoid
    average_x = sum(x) / len(x)
    average_y = sum(y) / len(y)
    average_z = sum(z) / len(z)
    # calculate distances from centroid for each coordinate
    distances = [(math.sqrt((c[0]-average_x)**2 + (c[1]-average_y)**2 + (c[2]-average_z)**2), c) for c in zip(x,y,z)]
    # sort distances in ascending order
    distances.sort()
    # get center point
    center_point = distances[0][1]
    # all zipped points
    merged_list = list(zip(x,y,z))
    # determine the number of coordinates to keep (top 80%)
    num_to_keep = int(0.8 * len(distances))
    # extract the top 80% of coordinates
    middle_index = merged_list.index(center_point)
    if middle_index < num_to_keep/2:
        top_coords = merged_list[:num_to_keep]
    elif (len(merged_list) - middle_index) < num_to_keep/2:
        start_index = len(merged_list) - num_to_keep
        top_coords = merged_list[start_index:]
    else:
        start_index = middle_index - int(num_to_keep/2)
        last_index = middle_index + int(num_to_keep/2)
        top_coords = merged_list[start_index:last_index]

    # top_coords = np.array([c[1] for c in distances[:num_to_keep]])
    return top_coords
    
def extract_top_eighty(x,y,z):
    
    # centroid for ellipsoid
    average_x = sum(x) / len(x)
    average_y = sum(y) / len(y)
    average_z = sum(z) / len(z)
    
    # calculate distances from centroid for each coordinate
    distances = [(math.sqrt((c[0]-average_x)**2 + (c[1]-average_y)**2 + (c[2]-average_z)**2), c) for c in zip(x,y,z)]

    # sort distances in ascending order
    distances.sort()
    
    # determine the number of coordinates to keep (top 80%)
    num_to_keep = int(0.8 * len(distances))

    # extract the top 80% of coordinates
    top_coords = np.array([c[1] for c in distances[:num_to_keep]])
    
    return top_coords

def areaOfMotionAnalysis(filepath, userIsResident, armIsLeft):
    
    if armIsLeft:
        ax = ax1 if userIsResident else ax3
    else:
        ax = ax4 if userIsResident else ax6
    
    # extract coordinates
    x, y, z = extractCoordinates(filepath, armIsLeft)

    # combine the arrays into a 3d numpy array
    all_coords = np.vstack((x, y, z)).T

    # add a text annotation with the total length
    if userIsResident:
        ax.text2D(0.5, 0.00, f'Your Performance', ha='center', transform=ax.transAxes, fontsize=10)
    else:
        ax.text2D(0.5, 0.00, f'Ideal Performance', ha='center', transform=ax.transAxes, fontsize=10)

    # plot the trajectory
    ax.plot(x, y, z, color='blue', label='Trajectory', linewidth=0.5)

    # extract the top 80% of coordinates
    top_coords = extract_top_eighty(x, y, z)
    
    # # create an index array indicating the positions of the matching elements in all_coords
    # idx = np.where(np.isin(all_coords, top_coords).all(axis=1))[0]

    # # extract the corresponding elements of top_coords in the order they appear in all_coords
    # top_coords_ordered = top_coords[np.argsort(idx)]
            
    x_top, y_top, z_top = ellipsoidGen(top_coords)
    x_all, y_all, z_all = ellipsoidGen(all_coords)
    
    # Add the ellipsoid wireframe to the plot
    ax.plot_wireframe(x_all, y_all, z_all, color='black', alpha=0.05)
    ax.plot_surface(x_all, y_all, z_all, color='cyan', alpha=0.05, shade=True, rstride=1, cstride=1)

    # ISSUE: generated ellipsoid for 80% coords is too big 
    ax.plot_wireframe(x_top, y_top, z_top, color='red', alpha=0.2, label='80% Coverage')

    # # Set the plot labels
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # Hide the 3D axes box
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()

    # Add a legend
    ax.legend(loc='upper right', fontsize=8)

# code to be executed when the file is run explicitly
if __name__ == '__main__':

    # Create the figure and subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), subplot_kw={'projection': '3d'})

    # get resident / surgeon's performances from CSV
    residentRecord = "Knot_Tying_D001_G13_14_3_1_2_3_2_3.csv"
    surgeonRecord = "Knot_Tying_B001_G13_13_2_2_2_2_2_3.csv"

    #LEFT
    # Area of Motion Analysis
    areaOfMotionAnalysis(filepath=residentRecord, userIsResident=True, armIsLeft=True)
    areaOfMotionAnalysis(filepath=surgeonRecord, userIsResident=False, armIsLeft=True)

    # compare trajectories
    mergeTrajectories(residentFilePath=residentRecord, surgeonFilePath=surgeonRecord, armIsLeft=True)

    #RIGHT
    # Area of Motion Analysis
    areaOfMotionAnalysis(filepath=residentRecord, userIsResident=True, armIsLeft=False)
    areaOfMotionAnalysis(filepath=surgeonRecord, userIsResident=False, armIsLeft=False)

    # compare trajectories
    mergeTrajectories(residentFilePath=residentRecord, surgeonFilePath=surgeonRecord, armIsLeft=False)

    # plot title
    ax2.set_title("Left Hand")
    ax5.set_title("Right Hand")

    # Add a main title to the figure
    fig.suptitle('Area of Motion Analysis', fontweight="bold")

    # Adjust the layout of the subplots
    fig.tight_layout()

    # Show the plot
    plt.show()