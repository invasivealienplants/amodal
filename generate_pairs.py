from scipy import misc
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
from itertools import product
from scipy.ndimage.measurements import label

f = open("amodal/dict_file.txt","r")

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

def match_grid(mask,c):
    return np.array((mask[:,:,0]==c[0])*(mask[:,:,1]==c[1])*(mask[:,:,2]==c[2])*(mask[:,:,3]==c[3]),dtype=np.float32)

for i in range(227):
    print(i)
    a = f.readline()
    counter = a[:a.index(",")]
    match_scores = []
    paths = []
    base_image = misc.imread("base/" + str(counter) + "_image.png")
    base_mask = misc.imread("base/" + str(counter) + "_mask.png")
    base_road = match_grid(base_mask,road_color)
    for root,_,mask_paths in os.walk("gtFine"):
        if not ("/test" in root):
            for mask_path in mask_paths:
                if mask_path[-9:] == "color.png":
                    subfolder = root[6:]
                    image_id = mask_path[:-17]
                    mask = misc.imread("gtFine/" + subfolder + "/" + image_id+"_gtFine_color.png")
                    paths.append((subfolder,image_id))
                    road = match_grid(mask,road_color)
                    match = np.where(road>base_road,-1.0,1.0)*road
                    score = np.sum(match)/np.sum(road)
                    match_scores.append(score)
    order = np.argsort(match_scores)[:20]
    
    total_copies = 0
    for idx in order:
        subfolder,image_id = paths[idx]
        mask = misc.imread("gtFine/" + subfolder + "/" + image_id+"_gtFine_color.png")
        image = misc.imread("leftImg8bit/" + subfolder + "/" + image_id + "_leftImg8bit.png")
        cargrid = match_grid(mask,car_color)
        pergrid = match_grid(mask,person_color)
        bikegrid = match_grid(mask,bike_color)
        pedgrid = match_grid(mask,ped_color)
        petgrid = match_grid(mask,pet_color)
        truckgrid = match_grid(mask,truck_color)
        trailgrid = match_grid(mask,trail_color)
        motorgrid = match_grid(mask,motor_color)
        licensegrid = match_grid(mask,license_color)
        roadgrid = match_grid(mask,road_color)
        fore_grid = np.minimum(cargrid+pergrid+bikegrid+pedgrid+petgrid+truckgrid+trailgrid+motorgrid+licensegrid,1.0)
        labeled_array,num_features = label(fore_grid,structure=np.ones([3,3]))
        
        total_features = 0
        new_base = np.copy(base_image)
        width = len(mask)
        height = len(mask[0])
        coordinates = list(product(range(width), range(height)))
        coordinates = np.reshape(coordinates,[width,height,2])
        for i in range(1,num_features+1):
            grid = np.where(labeled_array==i,1.0,0.0)
            grid_3d = np.tile(np.reshape(grid,[1024,2048,1]),(1,1,3))
            c = np.reshape(grid,[len(mask),len(mask[0]),1])*coordinates
            ground_y = np.max(c[:,:,0])
            ground_x = np.sum(c[:,:,1])/np.sum(grid)
            if np.sum(base_mask[int(ground_y)][int(ground_x)]-np.array(road_color)) == 0:
                new_base = np.where(grid_3d==1,image,new_base)
                total_features += 1
        if total_features > 0:
            misc.imsave("base/"+ str(counter) + "_" + str(total_copies) + ".png",new_base)
            total_copies += 1
       
       



