import os, sys
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
    while True:
        dir_path, dir_name = os.path.split(path)
        if dir_name == "SIMMETRIC":
            return path
        elif dir_name == "":
            return None
        else:
            path = dir_path


if __name__ == "__main__":
    # get the absolute path of the current script
    script_path = os.path.abspath(__file__)
    # verify path 
    simmetric_path = get_simmetric_path(script_path)
    check_path(simmetric_path) # abort if SIMMETRIC not found


    gesture_path = os.path.join(simmetric_path, 'processed-datasets', "OUTPUT-GESTURES", "Knot_Tying", "G13", )
    check_path(gesture_path) # abort if trial folder d.n.e

    os.chdir(gesture_path)

    fig, ((ax1, ax2)) = plt.subplots(ncols=2)

    files = next(os.walk(gesture_path))[1]

    for i, filename in enumerate(next(os.walk("."))[2]):
        if i == 1:
            continue

        df = pd.read_csv(filename)

        

        # for x_name in df.loc[:, "Volume of Motion" : "Economy of Motion"]:
        #     x_var = df.loc[:, x_name]
            
        #     for y_name in df.loc[:, "GRS" : "Quality of Final Product"]:
        #         x_var = df.loc[:, x_name]

        y = df.loc[:, "Economy of Motion"]
        x = df.loc[:, "GRS"]

        print(x.values)
        print(y.values)


        plt.scatter(x.values,y.values)

        m, b, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        plt.plot(x, m*x+b)
        print(r_value**2)






    # fig.tight_layout(pad=5)

    # fig.subtitle('Gesture Breakdown: User ' + user + ", Trial " + trial, fontweight="bold")

    # fig.legend(gestures)

    plt.show()