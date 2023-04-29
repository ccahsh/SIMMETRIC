import os
import shutil
import pandas as pd
import re
import sys

def get_simmetric_path(path):
    """
    Returns the path of the directory that ends with "SIMMETRIC"
    """
    while True:
        dir_path, dir_name = os.path.split(path)
        if dir_name == "SIMMETRIC":
            return path
        elif dir_name == "":
            return None
        else:
            path = dir_path
            
def no_simmetric_abort(path):
    if path is None:
        print("\n\n")
        print("'SIMMETRIC' folder cannot be found in the file path. Please fix and try again.")
        print("\n\n")
        sys.exit(1)
        
def no_datasets_JIGSAW_abort(path):
    if not os.path.exists(path):
        print("\n\n")
        print("'SIMMETRIC/datasets/JIGSAW' folder cannot be found from the file path. Please fix and try again.")
        print("\n\n")
        sys.exit(1)
            
def extract_letter_number(strings, ranking):
    pattern = r'([A-Z])(\d+)$'
    matches = [re.search(pattern, s) for s in strings]
    return [(m.group(1), int(m.group(2)), ranking[i]) if m else None for i, m in enumerate(matches)]

def extract_letter_numbers(strings, ranking):
    pattern = r'([A-Z])(\d+)$'
    matches = [re.search(pattern, s) for s in strings]

    return [(m.group(1), int(m.group(2)), tuple(ranking.iloc[i])) if m else None for i, m in enumerate(matches)]

def extract_info(s):
    match = re.match(r"(\w+)_([A-Z])(\d+)\.txt", s)
    if match:
        return (match.group(2), int(match.group(3)))
    else:
        return None
    
def find_element(long_list, short_list):
    for elem in long_list:
        if elem[:2] == short_list:
            return elem[2]
    return None

def find_element_first_element_only(long_list, short_list):
    for elem in long_list:
        if elem[:1] == short_list[:1]:
            return elem[2]
    return None

def group_by_gestures(df, label1, label2, label3):
    # group the DataFrame by the 'group' column and iterate over the groups
    ranges = []
    for group, data in df.groupby(label3):
        # extract the start and end values from the group
        start = data[label1].tolist()
        end = data[label2].tolist()
        # zip the start and end values together to create a list of ranges
        group_ranges = list(zip(start, end))
        # append the group name and ranges to the 'ranges' list
        ranges.append([group, group_ranges])
    return ranges

def override_make_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# code to be executed when the file is run explicitly
if __name__ == '__main__':
        
    # get the absolute path of the current script
    script_path = os.path.abspath(__file__)
    # verify path 
    simmetric_path = get_simmetric_path(script_path)
    no_simmetric_abort(simmetric_path) # abort if SIMMETRIC not found
    datasets_JIGSAW_path = os.path.join(simmetric_path, 'datasets', 'JIGSAW')
    no_datasets_JIGSAW_abort(datasets_JIGSAW_path) # abort if SIMMETRIC/datasets/JIGSAW d.n.e

    # change path to .../SIMMETRIC/datasets/JIGSAW
    os.chdir(datasets_JIGSAW_path)

    # identify folders with tasks
    folder = [f for f in next(os.walk('.'))[1] if os.path.exists(os.path.join(f, f, 'kinematics'))] 

    pwd = os.getcwd() # .../SIMMETRIC/datasets/JIGSAW
    parent_dir = os.path.abspath(os.path.join(pwd, os.pardir, os.pardir)) # path for 'processed-datasets' folder
    output_dir = os.path.join(parent_dir, 'processed-datasets')

    # collect lists for error messages
    emptyError = []
    partialError = []
    totalError = []

    outputpwd = os.path.join(output_dir, 'OUTPUT')
    override_make_folder(outputpwd)
    
    outputindividualpwd = os.path.join(output_dir, 'OUTPUT-temp')
    override_make_folder(outputindividualpwd)

    for i in folder: # later fix the i, j, k... var names
        # new folder
        outputtaskpwd = os.path.join(output_dir, 'OUTPUT', i)
        override_make_folder(outputtaskpwd)
        outputindividualtaskpwd = os.path.join(output_dir, 'OUTPUT-temp', i)
        override_make_folder(outputindividualtaskpwd)
        
        path = os.path.join(os.getcwd(), i, i) # double folder
        os.chdir(path) 
        
        # process meta file
        metacolumns = ['filename', 'skill_proclaimed', 'skill_obj', 'eval1', 'eval2', 'eval3', 'eval4', 'eval5', 'eval6']
        metadf = pd.read_csv('meta_file_'+i+'.txt', sep='\t+', header=None, names=metacolumns, engine='python')
        
        # print(metadf.head())
        
        # OUTPUT subfolders
        skillfolders = set(metadf['skill_proclaimed'])
        for i in skillfolders:
            skillsubfolder = os.path.join(outputtaskpwd, i)
            override_make_folder(skillsubfolder)
            
        metaset = extract_letter_number(metadf['filename'], metadf['skill_proclaimed']) # elements: e.g. ('B', 2, 'N')

        metaset_GRS = extract_letter_numbers(metadf['filename'], metadf.loc[:, "skill_obj":"eval6"]) # elements: e.g. ('B', 2, (9, 1, 2, 3, 4, 5))
        
        for i in metaset:
            userpwd = os.path.join(outputtaskpwd, i[2], i[0], str(i[1])) # e.g. (outputtaskpwd, 'N', 'B', str(1))
            override_make_folder(userpwd)
            
        transfolder = [f for f in next(os.walk('transcriptions'))[2] if f.endswith('.txt')]

        for i in transfolder:
            
            individual_trial = extract_info(i) # e.g. ('D', 4)
            expertise = find_element(metaset, individual_trial)
            
            # process transcriptions files
            # frame starts at 1, not 0 (based on what were written in transcriptions .txt files)
            transcolumns = ['start', 'end', 'type']
            transfilepath = os.path.join('transcriptions', i)
            transdf = pd.read_csv(transfilepath, sep=' ', header=None, names=transcolumns, engine='python', usecols=[0, 1, 2])
            
            # print(transdf.head())
            
            gestureset = set(transdf['type'])
            # print(gestureset)
            for i in gestureset:
                gesturepwd = os.path.join(outputtaskpwd, expertise, individual_trial[0], str(individual_trial[1]), i) # e.g. (outputtaskpwd, 'N', 'B', str(1), 'G8')
                override_make_folder(gesturepwd)
        
        # process datapoints
        kinematicsfolder = [f for f in next(os.walk('kinematics/AllGestures'))[2] if f.endswith('.txt')]

        for i in kinematicsfolder:
            
            individual_trial = extract_info(i) # e.g. ('D', 4)
            expertise = find_element(metaset, individual_trial)
            GRS = str(find_element(metaset_GRS, individual_trial)) # str(num) or 'None'

            # remove extra chars from tuple
            for c in [" ", "(", ")"]:
                GRS = GRS.replace(c, "")

            GRS = GRS.replace(",", "_")
            GRS += "_"
            GRS += str(expertise)
            
            # process files
            kinematicscolumns = [
                'MLTx', 'MLTy', 'MLTz', 
                'MLTR1', 'MLTR2', 'MLTR3', 'MLTR4', 'MLTR5', 'MLTR6', 'MLTR7', 'MLTR8', 'MLTR9',
                'MLTTVx', 'MLTTVy', 'MLTTVz',
                'MLTRVx', 'MLTRVy', 'MLTRVz',
                'MLGA',
                'MRTx', 'MRTy', 'MRTz',
                'MRTR1', 'MRTR2', 'MRTR3', 'MRTR4', 'MRTR5', 'MRTR6', 'MRTR7', 'MRTR8', 'MRTR9',
                'MRTTVx', 'MRTTVy', 'MRTTVz',
                'MRTRVx', 'MRTRVy', 'MRTRVz',
                'MRGA',
                'SLTx', 'SLTy', 'SLTz',
                'SLTR1', 'SLTR2', 'SLTR3', 'SLTR4', 'SLTR5', 'SLTR6', 'SLTR7', 'SLTR8', 'SLTR9',
                'SLTTVx', 'SLTTVy', 'SLTTVz',
                'SLTRVx', 'SLTRVy', 'SLTRVz',
                'SLGA',
                'SRTx', 'SRTy', 'SRTz',
                'SRTR1', 'SRTR2', 'SRTR3', 'SRTR4', 'SRTR5', 'SRTR6', 'SRTR7', 'SRTR8', 'SRTR9',
                'SRTTVx', 'SRTTVy', 'SRTTVz',
                'SRTRVx', 'SRTRVy', 'SRTRVz',
                'SRGA'
            ]
            kinematicsfilepath = os.path.join('kinematics/AllGestures', i)
            kinematicsdf = pd.read_csv(kinematicsfilepath, sep='\s+', header=None, names=kinematicscolumns, engine='python')
            
            if kinematicsdf.shape[0] == 0:
                # emptyError
                emptyfilepath = os.path.join(os.getcwd(), kinematicsfilepath)
                emptyError.append(emptyfilepath)
            
            # modify as suited
            highlights = kinematicscolumns
            # highlights = [
            #     'SLTx', 'SLTy', 'SLTz',
            #     'SLTTVx', 'SLTTVy', 'SLTTVz',
            #     'SLTRVx', 'SLTRVy', 'SLTRVz',
            #     'SLGA',
            #     'SRTx', 'SRTy', 'SRTz',
            #     'SRTTVx', 'SRTTVy', 'SRTTVz',
            #     'SRTRVx', 'SRTRVy', 'SRTRVz',
            #     'SRGA'
            # ]

            highlightsdf = kinematicsdf[highlights]
            highlightindividualsdf = highlightsdf.copy() # separate
            
            # OUTPUT 1 (newly added, so code is clunky): break down by individual

            individualfilepath = os.path.join(outputindividualtaskpwd, i.replace('.txt', '_' + GRS + '.csv'))
            highlightindividualsdf.index += 1 # same frame # as specified in transcriptions .txt files
            highlightindividualsdf.index.name = 'frame'
            highlightindividualsdf.to_csv(individualfilepath, index=True)
        
            # OUTPUT 2: break down by gestures
            
            # split copies of df based on gestures
            transcolumns = ['start', 'end', 'type']
            transfilepath = os.path.join('transcriptions', i)
            
            try: 
                transdf = pd.read_csv(transfilepath, sep=' ', header=None, names=transcolumns, engine='python', usecols=[0, 1, 2])
                transranges = group_by_gestures(df = transdf, label1 = 'start', label2 = 'end', label3 = 'type')
                
                for j in transranges:
                    gesture, ranges = (j[0]).strip(), j[1]
                    # frame starts at 1, not 0 (based on what were written in transcriptions .txt files)
                    partialhighlightsdf = highlightsdf.iloc[[i for r in ranges for i in range(r[0]-1, r[1])]]
                    partialhighlightsdfpwd = os.path.join(outputtaskpwd, expertise, individual_trial[0], str(individual_trial[1]), gesture)
                    override_make_folder(partialhighlightsdfpwd)
                    filepath = os.path.join(partialhighlightsdfpwd, i.replace('.txt', '_' + gesture + '_' + GRS + '.csv'))
                    partialhighlightsdf.index += 1 # same frame # as specified in transcriptions .txt files
                    partialhighlightsdf.index.name = 'frame'
                    partialhighlightsdf.to_csv(filepath, index=True)
            except:
                # unsplit df in newly added folder
                try: 
                    expertise = find_element_first_element_only(metaset, individual_trial) # meta file doesn't mention this .txt file, so trial # ignored 
                    unsplitpwd = os.path.join(outputtaskpwd, expertise, individual_trial[0], str(individual_trial[1])) # e.g. (outputtaskpwd, 'N', 'B', str(1))
                    override_make_folder(unsplitpwd)
                    filepath = os.path.join(unsplitpwd, i.replace('.txt', '_' + GRS + '.csv'))
                    highlightsdf.index += 1 # same frame # as specified in transcriptions .txt files
                    highlightsdf.index.name = 'frame'
                    highlightsdf.to_csv(filepath, index=True)
                    partialError.append(i)
                except:
                    totalError.append(i)

        # video is ignored for now
        
        # end
        os.chdir(pwd)
        
    # move and reorganize
    
    gestures_folder = os.path.join(outputpwd, 'Gestures')
    individual_folder = os.path.join(outputpwd, 'Individual')
    
    items_gestures = [i for i in os.listdir(outputpwd)]
    items_individual = [i for i in os.listdir(outputindividualpwd)]
    
    override_make_folder(gestures_folder)
    override_make_folder(individual_folder)
    
    for item in items_gestures:
        item_path = os.path.join(outputpwd, item)
        new_item_path = os.path.join(gestures_folder, item)
        shutil.move(item_path, new_item_path)
        
    for item in items_individual:
        item_path = os.path.join(outputindividualpwd, item)
        new_item_path = os.path.join(individual_folder, item)
        shutil.move(item_path, new_item_path)
    
    if os.path.exists(outputindividualpwd):
        shutil.rmtree(outputindividualpwd)
        
    # show error messages (in order of priority)

    print("\n\n")

    print("***The following file(s) are in kinematics/AllGestures but not in transcriptions folder. They are also not documented in meta_file_~.txt AT ALL. Folders corresponding to these file(s) cannot be found nor can be made in the OUPUT/Gestures folder. Please check and update the database: \n")
    for i in totalError:
        print(i)
    print("\n\n")

    print("The following file(s) are in kinematics/AllGestures but not in transcriptions folder nor fully documented in meta_file_~.txt. Regardless, folders corresponding to these files were automatically created in OUTPUT/Gestures folder. Please check and update the database: \n")
    for i in partialError:
        print(i)
    print("\n\n")

    print("The following file(s) are empty. Please check and update the file(s): \n")
    for i in emptyError:
        print(i)
    print("\n\n")
