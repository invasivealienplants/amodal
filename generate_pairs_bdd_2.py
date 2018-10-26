from scipy import misc
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
from itertools import product
from scipy.ndimage.measurements import label

dict_file = open("amodal/dict_file_bdd.txt","r")
match_file = open("amodal/match_file_bdd.txt","r")

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
  
for i in range(219):
    baseline = dict_file.readline()[:-1]
    baseline = baseline.split(",")
    base_mask_path = 'bdd100k_seg/bdd100k/seg/color_labels/'+baseline[1]+'_train_color.png'
    base_im_path = 'bdd100k_seg/bdd100k/seg/images/'+baseline[1]+'.jpg'
    
    base_mask = misc.imread(base_mask_path)
    base_im = misc.imread(base_im_path)
    
    line = match_file.readline()
    base_id = int(line[:-1])
    
    total_gen = 0
    
    line = match_file.readline()
    while line != "\n":
        maskpath = line[:-1]
        imname = line[:line.index("_")]

        mask_path = 'bdd100k_seg/bdd100k/seg/color_labels/train'+maskpath
        if not os.path.isfile(maskpath):
            mask_path = 'bdd100k_seg/bdd100k/seg/color_labels/val'+maskpath
            im_path = 'bdd100k_seg/bdd100k/seg/images/val'+imname+'.jpg'
        else:
            im_path = 'bdd100k_seg/bdd100k/seg/images/train'+imname+'.jpg'
        
        mask = misc.imread(mask_path)
        im = misc.imread(im_path)
        
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
            grid_3d = np.tile(np.reshape(grid,[720,1280,1]),(1,1,3))
            c = np.reshape(grid,[len(mask),len(mask[0]),1])*coordinates
            ground_y = np.max(c[:,:,0])
            ground_x1 = np.min(np.where(c[:,:,1]==0,np.inf,c[:,:,1]))
            ground_x2 = np.max(c[:,:,1])
            
            line = np.array(base_mask[int(ground_y)][int(ground_x1):int(ground_x2)])
            line_match = np.array(line[:,0]==road_color[0])*np.array(line[:,1]==road_color[1])*np.array(line[:,2]==road_color[2])*np.array(line[:,3]==road_color[3])
            line_match = np.array(line_match,dtype=np.float32)
            
            if np.mean(line_match) > 0.85:
                new_base = np.where(grid_3d==1,image,new_base)
                total_features += 1
                
        if total_features > 0:
            misc.imsave('base_pairs_bdd/'+str(base_id)+'_'+str(total_copies)+'.png')
            total_copies += 1
        
        line = match_file.readline()
