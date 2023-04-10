import os
import sys
import math
import numpy as np
import pandas as pd 
import shutil
from scipy.spatial import ConvexHull
import re
            
def no_preprocessed_datasets_OUTPUT_abort(path):
    if not os.path.exists(path):
        print("\n\n")
        print("'SIMMETRIC/preprocessed-datasets/OUTPUT' folder cannot be found from the file path. Please fix and try again.")
        print("\n\n")
        sys.exit(1)

def time_to_completion(frame):
    first_frame = frame.iloc[0]
    last_frame = frame.iloc[-1]
    frame_count = last_frame - first_frame
    time_in_seconds = frame_count / 30 # frame rate = 30 Hz
    return time_in_seconds

def euclidean_distances(input): # PROBLEM: units unknown (m?)
    dist = sum([math.sqrt(
        (input[i+1][0]-input[i][0])**2 + 
        (input[i+1][1]-input[i][1])**2 + 
        (input[i+1][2]-input[i][2])**2) 
                for i in range(len(input)-1)])
    return dist

def compute_ellipsoid_volume(radii):
    r1, r2, r3 = radii
    volume = abs((4/3) * np.pi * r1 * r2 * r3) # radius could be negative
    return volume   
    
def override_make_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# code to be executed when the file is run explicitly
if __name__ == '__main__':
    
    # get the absolute path of the current script
    script_path = os.path.abspath(__file__)

    # import functions from other python files
    processing_path = os.path.abspath(os.path.join(script_path, os.pardir, os.pardir, os.pardir)) # .../SIMMETRIC/processing
    preprocessing_jigsaw_path = os.path.join(processing_path, 'preprocessing')

    analysis_path = os.path.abspath(os.path.join(script_path, os.pardir, os.pardir)) # .../SIMMETRIC/processing/analysis
    volume_of_motion_path = os.path.join(analysis_path, 'toolbox', 'volume-of-motion')

    sys.path.extend([preprocessing_jigsaw_path, volume_of_motion_path])

    from preprocess_jigsaw import get_simmetric_path, no_simmetric_abort
    from ellipsoid_fit_modified import ellipsoid_fit
    from plotting import ellipsoidGen, extractCoordinates, mergeTrajectories, areaOfMotionAnalysis, extract_top_eighty

    def ellipsoidGenRadiiOnly(coords):
        # compute the convex hull of the points
        hull = ConvexHull(coords)

        # extract the vertices of the convex hull
        vertices = coords[hull.vertices,:]

        # fit ellipsoid
        center, evecs, radii, v = ellipsoid_fit(vertices)
        return radii

    # verify path 
    simmetric_path = get_simmetric_path(script_path)
    no_simmetric_abort(simmetric_path) # abort if SIMMETRIC not found
    preprocessed_datasets_path = os.path.join(simmetric_path, 'processed-datasets')
    preprocessed_datasets_OUTPUT_path = os.path.join(preprocessed_datasets_path, 'OUTPUT')
    no_preprocessed_datasets_OUTPUT_abort(preprocessed_datasets_OUTPUT_path) # abort if SIMMETRIC/preprocessed-datasets/OUTPUT d.n.e

    # change path to .../SIMMETRIC/preprocessed-datasets
    os.chdir(preprocessed_datasets_path)

    folder = [f for f in next(os.walk('OUTPUT'))[1] if not f.startswith('.')] # not .DS_Store

    pwd = os.getcwd() # .../SIMMETRIC/preprocessed-datasets

    outputgesturespwd = os.path.join(pwd, 'OUTPUT-GESTURES')
    override_make_folder(outputgesturespwd)

    for i in folder:
        # new folder
        outputtaskpwd = os.path.join(pwd, 'OUTPUT-GESTURES', i)
        override_make_folder(outputtaskpwd)
        path = os.path.join(pwd, 'OUTPUT', i) 
        os.chdir(path) 
        
        subfolder = [f for f in next(os.walk('.'))[1] if not f.startswith('.')] # {E, I, N} subfolders
        
        gestures = []
        for root, dirs, files in os.walk('.'):
            for dir in dirs:
                gestures.append(os.path.join(root, dir))

        gestures_unprocessed = list(set([os.path.basename(path) for path in gestures])) # all ending folders
        gestures_names = [f for f in gestures_unprocessed if re.match(r'^G\d+$', f)] # only gestures
                
        for j in gestures_names:
            gesturepwd = os.path.join(outputtaskpwd, j)
            override_make_folder(gesturepwd) # result: OUTPUT-BREAKDOWN > Knot-Tying > {G1, G11, .. }
            
        performance_info = [] 
        
        for k in subfolder: # k = self-claimed expertise level

            csv_files = []
            for root, dirs, files in os.walk(k): # within subfolder
                for file in files:
                    if file.endswith('.csv') and not file.startswith('~') and not file.startswith('.'):
                        full_path = os.path.join(root, file)
                        if os.path.isfile(full_path) and os.path.splitext(full_path)[1].lower() == '.csv':
                            csv_files.append(full_path) # valid, regular csv file paths
            
            for m in csv_files:
                split_info = os.path.splitext(m)[0].split('_')[1:]
                if len(split_info) < 3: # ignore csv file that doesn't have a specified gesture
                    continue

                if len(split_info) == 3:
                    user_trial, gesture, objscore = split_info[-3:]
                    objsub = [None for i in range(6)]
                else:
                    user_trial, gesture, objscore = split_info[-9:-6] # e.g. ('E003', 'G15', '22'), works with all task types (e.g. Knot_Tying vs. Suturing)
                    objsub = split_info[-6:]

                split = re.match(r'([A-Za-z]+)(\d+)', user_trial)
                if not split: # ignore csv files that do not have a concatenated string of user ID + trial #
                    continue
                user, trial = split.group(1), int(split.group(2))
                if objscore != 'None':
                    objscore = int(objscore)
                info = {
                    'gesture': gesture,
                    'user': user,
                    'trial': trial,
                    'selfscore': k,
                    'objscore': objscore,
                    'objsub': objsub,
                    'path': m
                }
                performance_info.append(info)
                
        performance_info = sorted(performance_info, key=lambda x: (x['gesture'], x['user'], x['trial'])) # sort in gesture -> user -> trial
        
        # unoptimized; fix later (O(gestures_names * performance_info))
        for n in gestures_names:
            
            df_left = pd.DataFrame(columns=[
                'User', 
                'Trial', 
                'Gesture Frequency', 
                'Self-Claimed Level', 
                'GRS', 
                'Respect for Tissue',
                'Suture/Needle Handling',
                'Time and Motion',
                'Flow of Operation',
                'Overall Performance',
                'Quality of Final Product',
                'Volume of Motion',
                '80% Volume of Motion', 
                'Time to Completion', 
                'Economy of Motion'
                ])
            # create df_right with same columns as df_left
            df_right = pd.DataFrame(columns=df_left.columns)
            
            for p in performance_info:

                if p['gesture'] == n:
                    # read csv from 'OUTPUT' folder
                    gesturedata = pd.read_csv(p['path'])
                    
                    # calculate the difference between consecutive values in the 'frame' column
                    diff = gesturedata['frame'].diff()
                    # create a new column 'group' that indicates the group number
                    gesturedata['group'] = (diff != 1).cumsum()
                    # group the data frame by 'group' column
                    gesturedata_grouped = gesturedata.groupby('group')

                    # iterate over groups and use corresponding data frames
                    for gesture_freq, gesture_data in gesturedata_grouped:
                        
                        # expand as needed
                        # assuming units are in meters (convert to centimeters)
                        SLTx, SLTy, SLTz = gesture_data['SLTx']*100, gesture_data['SLTy']*100, gesture_data['SLTz']*100
                        SLTTVx, SLTTVy, SLTTVz = gesture_data['SLTTVx']*100, gesture_data['SLTTVy']*100, gesture_data['SLTTVz']*100
                        SLTRVx, SLTRVy, SLTRVz = gesture_data['SLTRVx']*100, gesture_data['SLTRVy']*100, gesture_data['SLTRVz']*100
                        SLGA = gesture_data['SLGA'] 
                        
                        SRTx, SRTy, SRTz = gesture_data['SRTx']*100, gesture_data['SRTy']*100, gesture_data['SRTz']*100
                        SRTTVx, SRTTVy, SRTTVz = gesture_data['SRTTVx']*100, gesture_data['SRTTVy']*100, gesture_data['SRTTVz']*100
                        SRTRVx, SRTRVy, SRTRVz = gesture_data['SRTRVx']*100, gesture_data['SRTRVy']*100, gesture_data['SRTRVz']*100
                        SRGA = gesture_data['SRGA']
                        
                        frame = gesture_data['frame']
                        
                        # expand as needed -- coordinate data only used
                        SLT = [(x, y, z) for x, y, z in zip(SLTx, SLTy, SLTz)] # list of 3d coordinates for SLAVE LEFT
                        SLT_x, SLT_y, SLT_z = [x for x in SLTx], [y for y in SLTy], [z for z in SLTz] # single axis data
                        
                        SRT = [(x, y, z) for x, y, z in zip(SRTx, SRTy, SRTz)] # list of 3d coordinates for SLAVE RIGHT
                        SRT_x, SRT_y, SRT_z = [x for x in SRTx], [y for y in SRTy], [z for z in SRTz] # single axis data
                                    
                        # PROCESSING (made subroutines for readability)
                        
                        # Time to Completion
                        timeToCompletion = time_to_completion(frame) # in seconds
                        
                        # Economy of Motion
                        econMotion_SLT = euclidean_distances(SLT) # in cm (assumed)
                        econMotion_SRT = euclidean_distances(SRT) 
                        
                        # Volume of Motion
                        volMotion_axis_SLT = ellipsoidGenRadiiOnly(np.array(SLT)) # numpy array expected
                        volMotion_SLT = compute_ellipsoid_volume(volMotion_axis_SLT) # in cm^3 (assumed)
                        volMotion_axis_SRT = ellipsoidGenRadiiOnly(np.array(SRT)) 
                        volMotion_SRT = compute_ellipsoid_volume(volMotion_axis_SRT)
                        
                        # 80% Volume of Motion
                        eighty_percent_points_SLT = extract_top_eighty(SLT_x, SLT_y, SLT_z) # 80% closest points to centroid
                        volMotion_eighty_axis_SLT = ellipsoidGenRadiiOnly(eighty_percent_points_SLT) 
                        volMotion_eighty_SLT = compute_ellipsoid_volume(volMotion_eighty_axis_SLT)
                        eighty_percent_points_SRT = extract_top_eighty(SRT_x, SRT_y, SRT_z) 
                        volMotion_eighty_axis_SRT = ellipsoidGenRadiiOnly(eighty_percent_points_SRT) 
                        volMotion_eighty_SRT = compute_ellipsoid_volume(volMotion_eighty_axis_SRT)
                        
                        # WRITE RESULTS 
                        row_left = {
                            'User': p['user'],
                            'Trial': p['trial'],
                            'Gesture Frequency': gesture_freq,
                            'Self-Claimed Level': p['selfscore'],
                            'GRS': p['objscore'],
                            'Respect for Tissue': p['objsub'][0],
                            'Suture/Needle Handling':p['objsub'][1],
                            'Time and Motion':p['objsub'][2],
                            'Flow of Operation':p['objsub'][3],
                            'Overall Performance':p['objsub'][4],
                            'Quality of Final Product':p['objsub'][5],
                            'Volume of Motion': volMotion_SLT,
                            '80% Volume of Motion': volMotion_eighty_SLT,
                            'Time to Completion': timeToCompletion,
                            'Economy of Motion': econMotion_SLT
                        }
                        df_left.loc[len(df_left)] = row_left

                        row_right = {
                            'User': p['user'],
                            'Trial': p['trial'],
                            'Gesture Frequency': gesture_freq,
                            'Self-Claimed Level': p['selfscore'],
                            'GRS': p['objscore'],
                            'Respect for Tissue': p['objsub'][0],
                            'Suture/Needle Handling':p['objsub'][1],
                            'Time and Motion':p['objsub'][2],
                            'Flow of Operation':p['objsub'][3],
                            'Overall Performance':p['objsub'][4],
                            'Quality of Final Product':p['objsub'][5],
                            'Volume of Motion': volMotion_SRT,
                            '80% Volume of Motion': volMotion_eighty_SRT,
                            'Time to Completion': timeToCompletion,
                            'Economy of Motion': econMotion_SRT
                        }
                        df_right.loc[len(df_right)] = row_right
            
            # export df_left and df_right
            dfleftcsvname = i + '_' + 'Compilation' + '_' + n + '_' + 'Left' + '.csv' # e.g. Knot_Tying_Compilation_G13_Left.csv
            dfrightcsvname = i + '_' + 'Compilation' + '_' + n + '_' + 'Right' + '.csv'
            dfleftexportpath = os.path.join(outputtaskpwd, n, dfleftcsvname) # e.g. (.../processed-datasets/OUTPUT-GESTURES/Knot_Tying_, G13, Knot_Tying_Compilation_G13_Left.csv)
            dfrightexportpath = os.path.join(outputtaskpwd, n, dfrightcsvname)
    
            df_left.to_csv(dfleftexportpath, index=True)
            df_right.to_csv(dfrightexportpath, index=True)
            