import os
import shutil
import pandas as pd
import re

def extract_letter_number(strings, ranking):
    pattern = r'([A-Z])(\d+)$'
    matches = [re.search(pattern, s) for s in strings]
    return [(m.group(1), int(m.group(2)), ranking[i]) if m else None for i, m in enumerate(matches)]

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

# identify folders with tasks
folder = [f for f in next(os.walk('.'))[1] if os.path.exists(os.path.join(f, f, 'kinematics'))] 

pwd = os.getcwd()

# collect lists for error messages
emptyError = []
partialError = []
totalError = []

outputpwd = os.path.join(pwd, 'OUTPUT')
override_make_folder(outputpwd)

for i in folder:
    # new folder
    outputtaskpwd = os.path.join(pwd, 'OUTPUT', i)
    override_make_folder(outputtaskpwd)
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
    # print(metaset)
    
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
        highlights = [
            'SLTx', 'SLTy', 'SLTz',
            'SLTTVx', 'SLTTVy', 'SLTTVz',
            'SLTRVx', 'SLTRVy', 'SLTRVz',
            'SLGA',
            'SRTx', 'SRTy', 'SRTz',
            'SRTTVx', 'SRTTVy', 'SRTTVz',
            'SRTRVx', 'SRTRVy', 'SRTRVz',
            'SRGA'
        ]

        highlightsdf = kinematicsdf[highlights]
        
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
                filepath = os.path.join(partialhighlightsdfpwd, i.replace('.txt', '_' + gesture + '.csv'))
                partialhighlightsdf.index += 1 # same frame # as specified in transcriptions .txt files
                partialhighlightsdf.index.name = 'frame'
                partialhighlightsdf.to_csv(filepath, index=True)
        except:
            # unsplit df in newly added folder
            try: 
                expertise = find_element_first_element_only(metaset, individual_trial) # meta file doesn't mention this .txt file, so trial # ignored 
                unsplitpwd = os.path.join(outputtaskpwd, expertise, individual_trial[0], str(individual_trial[1])) # e.g. (outputtaskpwd, 'N', 'B', str(1))
                override_make_folder(unsplitpwd)
                filepath = os.path.join(unsplitpwd, i.replace('.txt', '.csv'))
                highlightsdf.index += 1 # same frame # as specified in transcriptions .txt files
                highlightsdf.index.name = 'frame'
                highlightsdf.to_csv(filepath, index=True)
                partialError.append(i)
            except:
                totalError.append(i)

    # video is ignored for now
    
    # end
    os.chdir(pwd)
    
# show error messages (in order of priority)

print("\n\n")

print("***The following file(s) are in kinematics/AllGestures but not in transcriptions folder. They are also not documented in meta_file_~.txt AT ALL. Folders corresponding to these file(s) cannot be found nor can be made in the OUPUT folder. Please check and update the database: \n")
for i in totalError:
    print(i)
print("\n\n")

print("The following file(s) are in kinematics/AllGestures but not in transcriptions folder nor fully documented in meta_file_~.txt. Regardless, folders corresponding to these files were automatically created in OUTPUT folder. Please check and update the database: \n")
for i in partialError:
    print(i)
print("\n\n")

print("The following file(s) are empty. Please check and update the file(s): \n")
for i in emptyError:
    print(i)
print("\n\n")