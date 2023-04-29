import os
import sys
import math
import numpy as np
import pandas as pd 
import glob
import shutil
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr
import re
            
def correlation_output(X, Y, filepaths):
    vvthreshold = 1
    result_list = []
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        filename = os.path.basename(filepath)
        for x in X: # derived data
            for y in Y: # eval metrics
                dfcopy = df.copy()
                if x == 'S - 80% V / V Ratio' or x == 'M - 80% V / V Ratio': # if evaluating v/v ratio
                    dfcopy = df[df[x] <= vvthreshold] # ignore error values
                if dfcopy.shape[0] <= 2:
                    continue # for correlation, x and y must have length at least 2
                try:
                    x_var = np.var(dfcopy[x])
                    y_var = np.var(dfcopy[y])
                    if x_var == 0 or y_var == 0:
                        continue  # skip if either x or y is constant
                    corr_coeff, _ = pearsonr(dfcopy[x], dfcopy[y])     
                    result_list += [(filename, x, y, corr_coeff)]
                except:
                    continue # skip to next iteration if an exception is raised
    result_sorted = sorted(result_list, key=lambda x: abs(x[3]), reverse=True) # sort by greatest correlation
    return result_sorted            
            
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

    gestures_folder = [f for f in next(os.walk('OUTPUT/Gestures'))[1] if not f.startswith('.')] # not .DS_Store
    individual_folder = [f for f in next(os.walk('OUTPUT/Individual'))[1] if not f.startswith('.')] # not .DS_Store

    pwd = os.getcwd() # .../SIMMETRIC/preprocessed-datasets

    outputgesturespwd = os.path.join(pwd, 'OUTPUT-GESTURES')
    override_make_folder(outputgesturespwd)

    # OUTPUT 1: OUTPUT-GESTURES
    
    for i in gestures_folder:
        # new folder
        outputtaskpwd = os.path.join(pwd, 'OUTPUT-GESTURES', i)
        override_make_folder(outputtaskpwd)
        
        path = os.path.join(pwd, 'OUTPUT/Gestures', i) 
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
                csv_name = os.path.basename(m)
                if 'None' in csv_name:
                    continue # ignore CSV files w/o evaluation metrics
                split_info = os.path.splitext(m)[0].split('_')[1:]
                if (len(split_info) < 3): # ignore csv file that doesn't have a specified gesture
                    continue
                if len(split_info) == 3:
                    user_trial, gesture, objscore = split_info[-3:]
                    objsub = [None for i in range(6)]
                else:
                    user_trial, gesture, objscore = split_info[-10:-7] # e.g. ('E003', 'G15', '22'), works with all task types (e.g. Knot_Tying vs. Suturing)
                    objsub = split_info[-7:-1] # ignore last letter (self-proclaimed expertise level); that will be only used for the OUTPUT-INDIVIDUAL folder

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
                'S - Volume of Motion',
                'S - 80% Volume of Motion', 
                'S - 80% V / V Ratio',
                'M - Volume of Motion',
                'M - 80% Volume of Motion', 
                'M - 80% V / V Ratio',
                'Time to Completion', 
                'S - Economy of Motion',
                'M - Economy of Motion'
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
                        
                        MLTx, MLTy, MLTz = gesture_data['MLTx']*100, gesture_data['MLTy']*100, gesture_data['MLTz']*100
                        MLTTVx, MLTTVy, MLTTVz = gesture_data['MLTTVx']*100, gesture_data['MLTTVy']*100, gesture_data['MLTTVz']*100
                        MLTRVx, MLTRVy, MLTRVz = gesture_data['MLTRVx']*100, gesture_data['MLTRVy']*100, gesture_data['MLTRVz']*100
                        MLGA = gesture_data['MLGA'] 
                        
                        MRTx, MRTy, MRTz = gesture_data['MRTx']*100, gesture_data['MRTy']*100, gesture_data['MRTz']*100
                        MRTTVx, MRTTVy, MRTTVz = gesture_data['MRTTVx']*100, gesture_data['MRTTVy']*100, gesture_data['MRTTVz']*100
                        MRTRVx, MRTRVy, MRTRVz = gesture_data['MRTRVx']*100, gesture_data['MRTRVy']*100, gesture_data['MRTRVz']*100
                        MRGA = gesture_data['MRGA']
                        
                        frame = gesture_data['frame']
                        
                        # expand as needed -- coordinate data only used
                        SLT = [(x, y, z) for x, y, z in zip(SLTx, SLTy, SLTz)] # list of 3d coordinates for SLAVE LEFT
                        SLT_x, SLT_y, SLT_z = [x for x in SLTx], [y for y in SLTy], [z for z in SLTz] # single axis data
                        
                        SRT = [(x, y, z) for x, y, z in zip(SRTx, SRTy, SRTz)] # list of 3d coordinates for SLAVE RIGHT
                        SRT_x, SRT_y, SRT_z = [x for x in SRTx], [y for y in SRTy], [z for z in SRTz] # single axis data
                        
                        MLT = [(x, y, z) for x, y, z in zip(MLTx, MLTy, MLTz)] # list of 3d coordinates for MASTER LEFT
                        MLT_x, MLT_y, MLT_z = [x for x in MLTx], [y for y in MLTy], [z for z in MLTz] # single axis data
                        
                        MRT = [(x, y, z) for x, y, z in zip(MRTx, MRTy, MRTz)] # list of 3d coordinates for MASTER RIGHT
                        MRT_x, MRT_y, MRT_z = [x for x in MRTx], [y for y in MRTy], [z for z in MRTz] # single axis data
                                    
                        # PROCESSING (made subroutines for readability)
                        
                        # Time to Completion
                        timeToCompletion = time_to_completion(frame) # in seconds
                        
                        # Economy of Motion
                        econMotion_SLT = euclidean_distances(SLT) # in cm (assumed)
                        econMotion_SRT = euclidean_distances(SRT) 
                        econMotion_MLT = euclidean_distances(MLT) # in cm (assumed)
                        econMotion_MRT = euclidean_distances(MRT) 
                        
                        # S - Volume of Motion
                        volMotion_axis_SLT = ellipsoidGenRadiiOnly(np.array(SLT)) # numpy array expected
                        volMotion_SLT = compute_ellipsoid_volume(volMotion_axis_SLT) # in cm^3 (assumed)
                        volMotion_axis_SRT = ellipsoidGenRadiiOnly(np.array(SRT)) 
                        volMotion_SRT = compute_ellipsoid_volume(volMotion_axis_SRT)
                        
                        # S - 80% Volume of Motion
                        eighty_percent_points_SLT = extract_top_eighty(SLT_x, SLT_y, SLT_z) # 80% closest points to centroid
                        volMotion_eighty_axis_SLT = ellipsoidGenRadiiOnly(eighty_percent_points_SLT) 
                        volMotion_eighty_SLT = compute_ellipsoid_volume(volMotion_eighty_axis_SLT)
                        eighty_percent_points_SRT = extract_top_eighty(SRT_x, SRT_y, SRT_z) 
                        volMotion_eighty_axis_SRT = ellipsoidGenRadiiOnly(eighty_percent_points_SRT) 
                        volMotion_eighty_SRT = compute_ellipsoid_volume(volMotion_eighty_axis_SRT)
                        
                        # M - Volume of Motion
                        volMotion_axis_MLT = ellipsoidGenRadiiOnly(np.array(MLT)) # numpy array expected
                        volMotion_MLT = compute_ellipsoid_volume(volMotion_axis_MLT) # in cm^3 (assumed)
                        volMotion_axis_MRT = ellipsoidGenRadiiOnly(np.array(MRT)) 
                        volMotion_MRT = compute_ellipsoid_volume(volMotion_axis_MRT)
                        
                        # M - 80% Volume of Motion
                        eighty_percent_points_MLT = extract_top_eighty(MLT_x, MLT_y, MLT_z) # 80% closest points to centroid
                        volMotion_eighty_axis_MLT = ellipsoidGenRadiiOnly(eighty_percent_points_MLT) 
                        volMotion_eighty_MLT = compute_ellipsoid_volume(volMotion_eighty_axis_MLT)
                        eighty_percent_points_MRT = extract_top_eighty(MRT_x, MRT_y, MRT_z) 
                        volMotion_eighty_axis_MRT = ellipsoidGenRadiiOnly(eighty_percent_points_MRT) 
                        volMotion_eighty_MRT = compute_ellipsoid_volume(volMotion_eighty_axis_MRT)
                        
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
                            'S - Volume of Motion': volMotion_SLT,
                            'S - 80% Volume of Motion': volMotion_eighty_SLT, 
                            'S - 80% V / V Ratio': volMotion_eighty_SLT / volMotion_SLT,
                            'M - Volume of Motion': volMotion_MLT,
                            'M - 80% Volume of Motion': volMotion_eighty_MLT, 
                            'M - 80% V / V Ratio': volMotion_eighty_MLT / volMotion_MLT,
                            'Time to Completion': timeToCompletion,
                            'S - Economy of Motion': econMotion_SLT,
                            'M - Economy of Motion': econMotion_MLT
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
                            'S - Volume of Motion': volMotion_SRT,
                            'S - 80% Volume of Motion': volMotion_eighty_SRT, 
                            'S - 80% V / V Ratio': volMotion_eighty_SRT / volMotion_SRT,
                            'M - Volume of Motion': volMotion_MRT,
                            'M - 80% Volume of Motion': volMotion_eighty_MRT, 
                            'M - 80% V / V Ratio': volMotion_eighty_MRT / volMotion_MRT,
                            'Time to Completion': timeToCompletion,
                            'S - Economy of Motion': econMotion_SRT,
                            'M - Economy of Motion': econMotion_MRT
                        }
                        df_right.loc[len(df_right)] = row_right
            
            # export df_left and df_right
            dfleftcsvname = i + '_' + 'Compilation' + '_' + n + '_' + 'Left' + '.csv' # e.g. Knot_Tying_Compilation_G13_Left.csv
            dfrightcsvname = i + '_' + 'Compilation' + '_' + n + '_' + 'Right' + '.csv'
            dfleftexportpath = os.path.join(outputtaskpwd, n, dfleftcsvname) # e.g. (.../processed-datasets/OUTPUT-GESTURES/Knot_Tying_, G13, Knot_Tying_Compilation_G13_Left.csv)
            dfrightexportpath = os.path.join(outputtaskpwd, n, dfrightcsvname)
    
            df_left.to_csv(dfleftexportpath, index=True)
            df_right.to_csv(dfrightexportpath, index=True)
            
    # OUTPUT 2: OUTPUT-INDIVIDUAL
    
    outputindividualtaskpwd = os.path.join(pwd, 'OUTPUT-INDIVIDUAL')
    override_make_folder(outputindividualtaskpwd)
    
    for i in individual_folder: # task type
        # new folder
        outputindividualtaskpwd = os.path.join(pwd, 'OUTPUT-INDIVIDUAL', i)
        override_make_folder(outputindividualtaskpwd)
        
        path = os.path.join(pwd, 'OUTPUT/Individual', i) 
        os.chdir(path) 
        
        csv_files = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.csv') and not file.startswith('~') and not file.startswith('.'):
                    full_path = os.path.join(root, file)
                    if os.path.isfile(full_path) and os.path.splitext(full_path)[1].lower() == '.csv':
                        csv_name = os.path.basename(full_path)
                        if 'None' in csv_name:
                            continue # ignore CSV files w/o evaluation metrics
                        csv_files.append(full_path) # valid, regular csv file paths 
                
        # new csv file
        df_left = pd.DataFrame(columns=[
            'User', 
            'Trial', 
            'Self-Claimed Level', 
            'GRS', 
            'Respect for Tissue',
            'Suture/Needle Handling',
            'Time and Motion',
            'Flow of Operation',
            'Overall Performance',
            'Quality of Final Product',
            'S - Volume of Motion',
            'S - 80% Volume of Motion', 
            'S - 80% V / V Ratio',
            'M - Volume of Motion',
            'M - 80% Volume of Motion', 
            'M - 80% V / V Ratio',
            'Time to Completion', 
            'S - Economy of Motion',
            'M - Economy of Motion'
        ])
        # create df_right with same columns as df_left
        df_right = pd.DataFrame(columns=df_left.columns)
            
        for m in csv_files:
            split_info = os.path.splitext(m)[0].split('_')[1:]
            user_trial, objscore = split_info[-9:-7] # e.g. ('C002', '22'), works with all task types (e.g. Knot_Tying vs. Suturing)
            objsub = split_info[-7:-1]
            selfscore = split_info[-1]
            split = re.match(r'([A-Za-z]+)(\d+)', user_trial)
            if not split: # ignore csv files that do not have a concatenated string of user ID + trial #
                continue
            user, trial = split.group(1), int(split.group(2))
            if objscore != 'None':
                objscore = int(objscore)
                
            p = {
                'user': user,
                'trial': trial,
                'selfscore': selfscore,
                'objscore': objscore,
                'objsub': objsub,
                'path': m
            }
            
            individual_data = pd.read_csv(p['path'])
            
            # expand as needed
            # assuming units are in meters (convert to centimeters)
            SLTx, SLTy, SLTz = individual_data['SLTx']*100, individual_data['SLTy']*100, individual_data['SLTz']*100
            SLTTVx, SLTTVy, SLTTVz = individual_data['SLTTVx']*100, individual_data['SLTTVy']*100, individual_data['SLTTVz']*100
            SLTRVx, SLTRVy, SLTRVz = individual_data['SLTRVx']*100, individual_data['SLTRVy']*100, individual_data['SLTRVz']*100
            SLGA = individual_data['SLGA'] 
            
            SRTx, SRTy, SRTz = individual_data['SRTx']*100, individual_data['SRTy']*100, individual_data['SRTz']*100
            SRTTVx, SRTTVy, SRTTVz = individual_data['SRTTVx']*100, individual_data['SRTTVy']*100, individual_data['SRTTVz']*100
            SRTRVx, SRTRVy, SRTRVz = individual_data['SRTRVx']*100, individual_data['SRTRVy']*100, individual_data['SRTRVz']*100
            SRGA = individual_data['SRGA']
            
            MLTx, MLTy, MLTz = individual_data['MLTx']*100, individual_data['MLTy']*100, individual_data['MLTz']*100
            MLTTVx, MLTTVy, MLTTVz = individual_data['MLTTVx']*100, individual_data['MLTTVy']*100, individual_data['MLTTVz']*100
            MLTRVx, MLTRVy, MLTRVz = individual_data['MLTRVx']*100, individual_data['MLTRVy']*100, individual_data['MLTRVz']*100
            MLGA = individual_data['MLGA'] 
            
            MRTx, MRTy, MRTz = individual_data['MRTx']*100, individual_data['MRTy']*100, individual_data['MRTz']*100
            MRTTVx, MRTTVy, MRTTVz = individual_data['MRTTVx']*100, individual_data['MRTTVy']*100, individual_data['MRTTVz']*100
            MRTRVx, MRTRVy, MRTRVz = individual_data['MRTRVx']*100, individual_data['MRTRVy']*100, individual_data['MRTRVz']*100
            MRGA = individual_data['MRGA']
            
            frame = individual_data['frame']
            
            # expand as needed -- coordinate data only used
            SLT = [(x, y, z) for x, y, z in zip(SLTx, SLTy, SLTz)] # list of 3d coordinates for SLAVE LEFT
            SLT_x, SLT_y, SLT_z = [x for x in SLTx], [y for y in SLTy], [z for z in SLTz] # single axis data
            
            SRT = [(x, y, z) for x, y, z in zip(SRTx, SRTy, SRTz)] # list of 3d coordinates for SLAVE RIGHT
            SRT_x, SRT_y, SRT_z = [x for x in SRTx], [y for y in SRTy], [z for z in SRTz] # single axis data
            
            MLT = [(x, y, z) for x, y, z in zip(MLTx, MLTy, MLTz)] # list of 3d coordinates for MASTER LEFT
            MLT_x, MLT_y, MLT_z = [x for x in MLTx], [y for y in MLTy], [z for z in MLTz] # single axis data
            
            MRT = [(x, y, z) for x, y, z in zip(MRTx, MRTy, MRTz)] # list of 3d coordinates for MASTER RIGHT
            MRT_x, MRT_y, MRT_z = [x for x in MRTx], [y for y in MRTy], [z for z in MRTz] # single axis data
                        
            # PROCESSING (made subroutines for readability)
            
            # Time to Completion
            timeToCompletion = time_to_completion(frame) # in seconds
            
            # Economy of Motion
            econMotion_SLT = euclidean_distances(SLT) # in cm (assumed)
            econMotion_SRT = euclidean_distances(SRT) 
            econMotion_MLT = euclidean_distances(MLT) # in cm (assumed)
            econMotion_MRT = euclidean_distances(MRT) 
            
            # S - Volume of Motion
            volMotion_axis_SLT = ellipsoidGenRadiiOnly(np.array(SLT)) # numpy array expected
            volMotion_SLT = compute_ellipsoid_volume(volMotion_axis_SLT) # in cm^3 (assumed)
            volMotion_axis_SRT = ellipsoidGenRadiiOnly(np.array(SRT)) 
            volMotion_SRT = compute_ellipsoid_volume(volMotion_axis_SRT)
            
            # S - 80% Volume of Motion
            eighty_percent_points_SLT = extract_top_eighty(SLT_x, SLT_y, SLT_z) # 80% closest points to centroid
            volMotion_eighty_axis_SLT = ellipsoidGenRadiiOnly(eighty_percent_points_SLT) 
            volMotion_eighty_SLT = compute_ellipsoid_volume(volMotion_eighty_axis_SLT)
            eighty_percent_points_SRT = extract_top_eighty(SRT_x, SRT_y, SRT_z) 
            volMotion_eighty_axis_SRT = ellipsoidGenRadiiOnly(eighty_percent_points_SRT) 
            volMotion_eighty_SRT = compute_ellipsoid_volume(volMotion_eighty_axis_SRT)
            
            # M - Volume of Motion
            volMotion_axis_MLT = ellipsoidGenRadiiOnly(np.array(MLT)) # numpy array expected
            volMotion_MLT = compute_ellipsoid_volume(volMotion_axis_MLT) # in cm^3 (assumed)
            volMotion_axis_MRT = ellipsoidGenRadiiOnly(np.array(MRT)) 
            volMotion_MRT = compute_ellipsoid_volume(volMotion_axis_MRT)
            
            # M - 80% Volume of Motion
            eighty_percent_points_MLT = extract_top_eighty(MLT_x, MLT_y, MLT_z) # 80% closest points to centroid
            volMotion_eighty_axis_MLT = ellipsoidGenRadiiOnly(eighty_percent_points_MLT) 
            volMotion_eighty_MLT = compute_ellipsoid_volume(volMotion_eighty_axis_MLT)
            eighty_percent_points_MRT = extract_top_eighty(MRT_x, MRT_y, MRT_z) 
            volMotion_eighty_axis_MRT = ellipsoidGenRadiiOnly(eighty_percent_points_MRT) 
            volMotion_eighty_MRT = compute_ellipsoid_volume(volMotion_eighty_axis_MRT)
            
            # WRITE RESULTS 
            row_left = {
                'User': p['user'],
                'Trial': p['trial'],
                'Self-Claimed Level': p['selfscore'],
                'GRS': p['objscore'],
                'Respect for Tissue': p['objsub'][0],
                'Suture/Needle Handling':p['objsub'][1],
                'Time and Motion':p['objsub'][2],
                'Flow of Operation':p['objsub'][3],
                'Overall Performance':p['objsub'][4],
                'Quality of Final Product':p['objsub'][5],
                'S - Volume of Motion': volMotion_SLT,
                'S - 80% Volume of Motion': volMotion_eighty_SLT, 
                'S - 80% V / V Ratio': volMotion_eighty_SLT / volMotion_SLT,
                'M - Volume of Motion': volMotion_MLT,
                'M - 80% Volume of Motion': volMotion_eighty_MLT, 
                'M - 80% V / V Ratio': volMotion_eighty_MLT / volMotion_MLT,
                'Time to Completion': timeToCompletion,
                'S - Economy of Motion': econMotion_SLT,
                'M - Economy of Motion': econMotion_MLT
            }
            df_left.loc[len(df_left)] = row_left

            row_right = {
                'User': p['user'],
                'Trial': p['trial'],
                'Self-Claimed Level': p['selfscore'],
                'GRS': p['objscore'],
                'Respect for Tissue': p['objsub'][0],
                'Suture/Needle Handling':p['objsub'][1],
                'Time and Motion':p['objsub'][2],
                'Flow of Operation':p['objsub'][3],
                'Overall Performance':p['objsub'][4],
                'Quality of Final Product':p['objsub'][5],
                'S - Volume of Motion': volMotion_SRT,
                'S - 80% Volume of Motion': volMotion_eighty_SRT, 
                'S - 80% V / V Ratio': volMotion_eighty_SRT / volMotion_SRT,
                'M - Volume of Motion': volMotion_MRT,
                'M - 80% Volume of Motion': volMotion_eighty_MRT, 
                'M - 80% V / V Ratio': volMotion_eighty_MRT / volMotion_MRT,
                'Time to Completion': timeToCompletion,
                'S - Economy of Motion': econMotion_SRT,
                'M - Economy of Motion': econMotion_MRT
            }
            df_right.loc[len(df_right)] = row_right
        
        # export df_left and df_right
        dfleftcsvname = i + '_' + 'Compilation' + '_' + 'Total' + '_' + 'Left' + '.csv' # e.g. Knot_Tying_Compilation_Total_Left.csv
        dfrightcsvname = i + '_' + 'Compilation' + '_' + 'Total' + '_' + 'Right' + '.csv'
        dfleftexportpath = os.path.join(outputindividualtaskpwd, dfleftcsvname) # e.g. (.../processed-datasets/OUTPUT-INDIVIDUAL/Knot_Tying_, Knot_Tying_Compilation_Total_Left.csv)
        dfrightexportpath = os.path.join(outputindividualtaskpwd, dfrightcsvname)

        df_left.to_csv(dfleftexportpath, index=True)
        df_right.to_csv(dfrightexportpath, index=True)

    # OUTPUT 3: Correlations for ALL GENERATED CSV FILES
    
    outputgesturespwd = os.path.join(pwd, 'OUTPUT-GESTURES')
    outputindividualtaskpwd = os.path.join(pwd, 'OUTPUT-INDIVIDUAL')
    
    simmetric_metrics = [
        'S - Volume of Motion',
        'S - 80% Volume of Motion', 
        'S - 80% V / V Ratio',
        'M - Volume of Motion',
        'M - 80% Volume of Motion', 
        'M - 80% V / V Ratio',
        'Time to Completion',
        'S - Economy of Motion',
        'M - Economy of Motion'
    ]
    evaluation_metrics = [
        'GRS',
        'Respect for Tissue',
        'Suture/Needle Handling',
        'Time and Motion',
        'Flow of Operation',
        'Overall Performance',
        'Quality of Final Product'
    ]
    
    os.chdir(outputgesturespwd)
    file_list = glob.glob('**', recursive=True)
    child_files = [f for f in file_list if f.endswith('.csv')]
    
    gestures_corr_results = correlation_output(simmetric_metrics, evaluation_metrics, child_files)
    
    with open('summary-gestures.txt', 'w') as f:
        for result in gestures_corr_results:
            result_str = map(str, result)
            line = ','.join(result_str) + '\n'
            f.write(line)
    
    os.chdir(outputindividualtaskpwd)
    file_list = glob.glob('**', recursive=True)
    child_files = [f for f in file_list if f.endswith('.csv')]
    
    individual_corr_results = correlation_output(simmetric_metrics, evaluation_metrics, child_files)

    with open('summary-individual.txt', 'w') as f:
        for result in individual_corr_results:
            result_str = map(str, result)
            line = ','.join(result_str) + '\n'
            f.write(line)

    overall_corr_results = gestures_corr_results + individual_corr_results
    overall_result_sorted = sorted(overall_corr_results, key=lambda x: abs(x[3]), reverse=True) # sort by greatest correlation
    
    outputoverallpwd = os.path.join(pwd, 'OUTPUT-OVERALL')
    override_make_folder(outputoverallpwd)
    os.chdir(outputoverallpwd)
    
    with open('summary-overall.txt', 'w') as f:
        for result in overall_result_sorted:
            result_str = map(str, result)
            line = ' ,'.join(result_str) + '\n'
            f.write(line)