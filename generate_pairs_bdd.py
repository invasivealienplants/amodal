from scipy import misc
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
from itertools import product
from scipy.ndimage.measurements import label

f = open("amodal/dict_file_bdd.txt","r")
match_file = open("/home/pp456/match_file_bdd.txt","w")

person_color = [220,20,60,255]
rider_color = [255,0,0,255]
bike_color = [119,11,32,255]
bus_color = [0,60,100,255]
car_color = [0,0,142,255]
caravan_color = [0,0,90,255]
motor_color = [0,0,230,255]
trail_color = [0,0,110,255]
train_color = [0,80,100,255]
truck_color = [0,0,70,255]

ground_color = [81,0,81,255]
park_color = [250,170,160,255]
road_color = [128,64,128,255]
side_color = [244,35,232,255]
terrain_color = [152,251,152,255]
veg_color = [107,142,35,255]

def match_grid(mask,c):
    return np.array((mask[:,:,0]==c[0])*(mask[:,:,1]==c[1])*(mask[:,:,2]==c[2]),dtype=np.float32)

match_scores = []

base_paths = []
base_images = []
base_masks = []
base_grounds = []
base_parks = []
base_roads = []
base_sides = []
base_terrains = []
base_vegs = []

for i in range(219):
    a = f.readline()
    a = a.split(",")
    counter = a[0]
    im_name = a[1][13:][:-1]
    base_mask = misc.imread("bdd100k_seg/bdd100k/seg/color_labels/" + im_name + "_train_color.png")
    base_image = misc.imread("bdd100k_seg/bdd100k/seg/images/" + im_name + ".jpg")
    
    base_paths.append(im_name)
    base_images.append(base_image)
    base_masks.append(base_mask)
    base_grounds.append(match_grid(base_mask,ground_color))
    base_parks.append(match_grid(base_mask,park_color))
    base_roads.append(match_grid(base_mask,road_color))
    base_sides.append(match_grid(base_mask,side_color))
    base_terrains.append(match_grid(base_mask,terrain_color))
    base_vegs.append(match_grid(base_mask,veg_color))
    
for i in range(271):
    print(i)
    base_path = base_paths[i]
    base_image = base_images[i]
    base_mask = base_masks[i]
    base_ground = base_grounds[i]
    base_park = base_parks[i]
    base_road = base_roads[i]
    base_side = base_sides[i]
    base_terrain = base_terrains[i]
    base_veg = base_vegs[i]
    
    match_scores_ = []
    paths = []
    for root,_,mask_paths in os.walk("bdd100k_seg/bdd100k/seg/color_labels"):
        if not ("/test" in root):
            for mask_path in mask_paths:
                if mask_path[-9:] == "color.png":
                    if (("val" in base_path and "val" in root) or ("train" in base_path and "train" in root)):
                        subfolder = root[24:]
                        image_id = mask_path[:-16]
                        mask = misc.imread(root+"/"+mask_path)
                        ground = match_grid(mask,ground_color)
                        park = match_grid(mask,park_color)
                        road = match_grid(mask,road_color)
                        side = match_grid(mask,side_color)
                        terrain = match_grid(mask,terrain_color)
                        veg = match_grid(mask,veg_color)
                        
                        match_ground = np.where(ground>base_ground,1.0,-1.0)*ground
                        match_park = np.where(park>base_park,1.0,-1.0)*park
                        match_road = np.where(road>base_road,1.0,-1.0)*road
                        match_side = np.where(side>base_side,1.0,-1.0)*side
                        match_terrain = np.where(terrain>base_terrain,1.0,-1.0)*terrain
                        match_veg = np.where(veg>base_veg,1.0,-1.0)*veg
                        
                        if np.sum(ground+park+road+side+terrain+veg) == 0:
                            score = np.inf
                        else:
                            score = np.sum(match_ground+match_park+match_road+match_side+match_terrain+match_veg)/np.sum(ground+park+road+side+terrain+veg)
                        match_scores_.append(score)
    match_scores.append(match_scores_)
    order = np.argsort(match_scores,axis=1)
    order = order[:,:30]
    np.save("order_bdd.npy",order)
    match_file.write(str(i)+"\n")
    for j in order[i]:
        match_file.write(str(paths[j])+"\n")
    match_file.write("\n")
