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
    # while True:
    #     dir_path, dir_name = os.path.split(path)
    #     if dir_name == "SIMMETRIC":
    #         return path
    #     elif dir_name == "":
    #         return None
    #     else:
    #         path = dir_path


if __name__ == "__main__":
    # get the absolute path of the current script
    script_path = os.path.abspath(__file__)
    # verify path 
    simmetric_path = get_simmetric_path(script_path)
    check_path(simmetric_path) # abort if SIMMETRIC not found


    gesture_path = os.path.join(simmetric_path, 'processed-datasets', "OUTPUT-GESTURES", "Knot_Tying", "G13", )
    check_path(gesture_path) # abort if trial folder d.n.e

    os.chdir(gesture_path)

    for i, filename in enumerate(next(os.walk("."))[2]):
        hand = "Left" if i==0 else "Right"

        df = pd.read_csv(filename)

        for y_name in df.loc[:, "Volume of Motion" : "Economy of Motion"]:
            y = df.loc[:, y_name]

            for x_name in df.loc[:, "GRS" : "Quality of Final Product"]:
                x = df.loc[:, x_name]
                m, b, r_value, p_value, std_err = scipy.stats.linregress(x, y)

                plt.figure()
                plt.title("{} vs. {} ({})".format(y_name, x_name, hand))
                plt.xlabel(x_name)
                plt.ylabel(y_name)

                plt.scatter(x.values,y.values)
                plt.plot(x, m*x+b)

                print("{} vs. {} ({})".format(y_name, x_name, hand))
                print(r_value**2)

    plt.show()