import pandas as pd
import numpy as np
import math

def distance(p1, p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2+ (p1.z-p2.z)**2)

class Pose:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

data = pd.read_csv("X01_Pea_on_a_Peg_01.csv")

p_old = []

economy = [0, 0]
for index, line in data.iterrows():
    x1 = line["PSM1_position_x"]
    y1 = line["PSM1_position_y"]
    z1 = line["PSM1_position_z"] 
    x2 = line["PSM2_position_x"]
    y2 = line["PSM2_position_y"]
    z2 = line["PSM2_position_z"] 

    p = [Pose(x1, y1, z1), Pose(x2, y2, z2)]
    
    if (index != 0 and not (math.isnan(p[0].x) or math.isnan(p[0].y) or math.isnan(p[0].z) or math.isnan(p[1].x) or math.isnan(p[1].y) or math.isnan(p[1].z))):
        economy[0] += distance(p[0], p_old[0])
        economy[1] += distance(p[1], p_old[1])

    p_old = p  

print(economy[0]*100/2.54, 'in')
print(economy[1]*100/2.54, 'in')
