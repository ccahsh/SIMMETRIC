import os, sys
from numpy import greater, pad
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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

    user = "B"
    trial = "1"


    trial_path = os.path.join(simmetric_path, 'processed-datasets', "OUTPUT", "Knot_Tying", "N", user, trial)
    check_path(trial_path) # abort if trial folder d.n.e

    fig, ((ax1, ax2)) = plt.subplots(ncols=2, subplot_kw={'projection': '3d'})

    gestures = next(os.walk(trial_path))[1]

    for i, subfolder in enumerate(os.walk(trial_path)):
        if i == 0:
            continue

        check_path(subfolder[0])
        os.chdir(subfolder[0])

        filename = subfolder[2][0]

        df = pd.read_csv(filename)

        

        ax1.plot(df[['SLTx']], df[['SLTy']], df[['SLTz']])    
        ax2.plot(df[['SRTx']], df[['SRTy']], df[['SRTz']])  

        # plot title
        ax1.set_title("Left Hand")
        ax2.set_title("Right Hand")

        ax1.set_xlabel('$X$')
        ax1.set_ylabel('$Y$')
        ax1.set_zlabel('$Z$')

        ax2.set_xlabel('$X$')
        ax2.set_ylabel('$Y$')
        ax2.set_zlabel('$Z$')


    fig.tight_layout(pad=5)

    fig.suptitle('Gesture Breakdown: User ' + user + ", Trial " + trial, fontweight="bold")

    fig.legend(gestures)

    plt.show()