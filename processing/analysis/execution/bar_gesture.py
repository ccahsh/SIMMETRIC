import os, sys
import re
from numpy import greater, pad
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy

def check_path(path):
    if not os.path.exists(path):
        raise Exception(path + " folder cannot be found from the file path. Please fix and try again.")

def get_simmetric_path(path):
    """
    Returns the path of the directory that ends with "SIMMETRIC"
    """

    match = re.search(r"SIMMETRIC", path)

    if match:
        return match.string[0:match.end()]
    else:
        raise Exception("SIMMETRIC folder cannot be found from the file path. Please fix and try again.")



if __name__ == "__main__":
    # get the absolute path of the current script
    script_path = os.path.abspath(__file__)
    # verify path 
    simmetric_path = get_simmetric_path(script_path)
    check_path(simmetric_path) # abort if SIMMETRIC not found


    gesture_path = os.path.join(simmetric_path, 'processed-datasets', "OUTPUT-GESTURES", "Knot_Tying", "G13", )
    check_path(gesture_path) # abort if trial folder d.n.e

    os.chdir(gesture_path)

    df_left = pd.DataFrame()
    df_right = pd.DataFrame()

    for i, filename in enumerate(next(os.walk("."))[2]):
        hand = "Left" if i==0 else "Right"

        df = pd.read_csv(filename)

        for y_name in df.loc[:, "Volume of Motion" : "Economy of Motion"]:
            # if y_name != "Economy of Motion":
            #     continue
            if y_name == "Time to Completion":
                if hand == "Left":
                    hand = "Both"
                else:
                    continue

            y = df.loc[:, y_name]

            for i, x_name in enumerate(df.loc[:, "Self-Claimed Level" : "Quality of Final Product"]):
                # if x_name != "Self-Claimed Level":
                #     continue

                x = df.loc[:, x_name]
                
                # possible scores for this GRS category
                if x_name == "GRS":
                    x_score = range(1, 31, 1)
                elif x_name == "Self-Claimed Level":
                    x_score = "NIE" # use string for searching dataframe first
                else: 
                    x_score = range(1, 6, 1)

                x_cat = [df.loc[df[x_name] == i].loc[:, y_name] for i in x_score]

                # print(x_cat)

                means = [score.mean() for score in x_cat]

                std_devs = [score.std() for score in x_cat]

                # print(means)
                # print(std_devs)

                # reencode experience level for bar plot
                if x_name == "Self-Claimed Level":
                    x_score = range(1, 4, 1)

                plt.figure()
                plt.title("{} vs. {} ({})".format(y_name, x_name, hand))
                plt.xlabel(x_name)
                plt.ylabel(y_name)

                plt.bar(x_score, means, yerr=std_devs, capsize=10)

                if x_name == "Self-Claimed Level":
                    ax = plt.gca()
                    ax.set_xticks(x_score)
                    ax.set_xticklabels("NIE")

            
                
                
                    

    plt.show()