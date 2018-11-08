from scipy import misc
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
from itertools import product
from scipy.ndimage.measurements import label

f = open("dict_file.txt","r")
match_file = open("match_file_v2.txt","w")

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

match_scores = []
order = []

base_paths = []
base_images = []
base_masks = []
base_roads = []
base_grounds = []
base_sides = []
base_vegs = []
base_terrains = []

match_scores = []

base_images = []
base_masks = []
base_roads = []
base_grounds = []
base_sides = []
base_vegs = []
base_terrains = []

for i in range(227):
    a = f.readline()
    counter = a[:a.index(",")]
    a = a[:-1].split(",")
    rt = a[1]
    pth = a[2]
    rtpth = rt+"/"+pth
    
    base_image = misc.imread("cityscapes/gtFine" + rtpth + "_gtFine_color.png")
    base_mask = misc.imread("cityscapes/leftImg8bit" + rtpth + "_leftImg8bit.png")
    base_paths.append(rtpth)
    print(rtpth)
    
    base_images.append(base_image)
    base_masks.append(base_mask)
    base_roads.append(match_grid(base_mask,road_color))
    base_grounds.append(match_grid(base_mask,ground_color))
    base_sides.append(match_grid(base_mask,side_color))
    base_vegs.append(match_grid(base_mask,veg_color))
    base_terrains.append(match_grid(base_mask,terrain_color))
    
for i in range(227):
    print(i)
    if i < 8:
        continue
    
    base_image = base_images[i]
    base_mask = base_masks[i]
    base_road = base_roads[i]
    base_ground = base_grounds[i]
    base_side = base_sides[i]
    base_veg = base_vegs[i]
    base_terrain = base_terrains[i]
    base_path = base_paths[i]
    
    match_scores_ = []
    paths = []
    
    for root,_,mask_paths in os.walk("cityscapes/gtFine"):
        if not ("/test" in root):
            for mask_path in mask_paths:
                if mask_path[-9:] == "color.png":
                    if (("val" in base_path and "val" in root) or ("train" in base_path and "train" in root)):
                        subfolder = root[17:]
                        image_id = mask_path[:-17]
                        mask = misc.imread("cityscapes/gtFine/" + subfolder + "/" + image_id+"_gtFine_color.png")
                        paths.append((subfolder,image_id))
                        print("cityscapes/gtFine/" + subfolder + "/" + image_id+"_gtFine_color.png")

                        road = match_grid(mask,road_color)
                        ground = match_grid(mask,ground_color)
                        side = match_grid(mask,side_color)
                        veg = match_grid(mask,veg_color)
                        terrain = match_grid(mask,terrain_color)

                        match_road = np.where(road>base_road,1.0,-1.0)*road
                        match_side = np.where(side>base_side,1.0,-1.0)*side
                        match_ground = np.where(ground>base_ground,1.0,-1.0)*ground
                        match_veg = np.where(veg>base_veg,1.0,-1.0)*veg
                        match_terrain = np.where(terrain>base_terrain,1.0,-1.0)*terrain
                        print(np.sum(match_road))
                        print(np.sum(match_side))
                        print(np.sum(match_ground))
                        print(np.sum(match_veg))
                        print(np.sum(match_terrain))
                        
                        if np.sum(road+side+ground+veg+terrain) == 0:
                            score = np.inf
                        else:
                            score = np.sum(match_road+match_side+match_ground+match_veg+match_terrain)/np.sum(road+side+ground+veg+terrain)
                        match_scores_.append(score)
                    
    order = np.argsort(match_scores_)[:20]
    print(match_scores_)
    print(order)
    match_file.write(str(i)+"\n")
    for j in order:
        match_file.write(str(paths[j])+"\n")
    match_file.write("\n")
