from scipy import misc
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
from itertools import product
from scipy.ndimage.measurements import label

dict_file = open("amodal/dict_file.txt","r")
match_file = open("amodal/match_file.txt","r")

car_color = [0,0,142,255]
person_color = [255,0,0,255]
bike_color = [119,11,32,255]
ped_color = [220,20,60,255]
pet_color = [111,74,0,255]
bus_color = [0,60,100,255]
truck_color = [0,0,90,255]
trail_color = [0,0,110,255]
motor_color = [0,0,230,255]
license_color = [0,0,142,255]
road_color = [128,64,128,255]
side_color = [244,35,232,255]
ground_color = [81,0,81,255]
veg_color = [107,142,35,255]
terrain_color = [152,251,152,255]

def match_grid(mask,c):
    return np.array((mask[:,:,0]==c[0])*(mask[:,:,1]==c[1])*(mask[:,:,2]==c[2]),dtype=np.float32)
  
for i in range(227):
    line = match_file.readline()
    base_id = int(line[:-1])
    
    line = match_file.readline()
    while line != "\n":
        line = line.split(", ")
        folder = line[0][2:]
        folder = folder[:len(folder)-1]
        impath = line[1][1:]
        impath = impath[:-3]
        
        print(i,folder,impath)
        line = match_file.readline()
