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
    print(i)
    baseline = dict_file.readline()[:-1]
    baseline = baseline.split(",")
    base_mask_path = 'cityscapes/gtFine'+baseline[1]+'/'+baseline[2]+'_gtFine_color.png'
    base_im_path = 'cityscapes/leftImg8bit'+baseline[1]+'/'+baseline[2]+'_leftImg8bit.png'
    
    base_mask = misc.imread(base_mask_path)
    base_im = misc.imread(base_im_path)
    
    line = match_file.readline()
    base_id = int(line[:-1])
    
    total_gen = 0
    
    line = match_file.readline()
    while line != "\n":
        line = line.split(", ")
        folder = line[0][2:]
        folder = folder[:len(folder)-1]
        impath = line[1][1:]
        impath = impath[:-3]
        
        mask_path = 'cityscapes/gtFine'+folder+'/'+impath+'_gtFine_color.png'
        im_path = 'cityscapes/leftImg8bit'+folder+'/'+impath+'_leftImg8bit.png'
        
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
        new_base = np.copy(base_im)
        new_base_mask = np.copy(base_mask)
        width = len(mask)
        height = len(mask[0])
        coordinates = list(product(range(width), range(height)))
        coordinates = np.reshape(coordinates,[width,height,2])
        for i in range(1,num_features+1):
            grid = np.where(labeled_array==i,1.0,0.0)
            grid_3d = np.tile(np.reshape(grid,[1024,2048,1]),(1,1,3))
            grid_4d = np.tile(np.reshape(grid,[1024,2048,1]),(1,1,4))
            c = np.reshape(grid,[len(mask),len(mask[0]),1])*coordinates
            ground_y = np.max(c[:,:,0])
            ground_x1 = np.min(np.where(c[:,:,1]==0,np.inf,c[:,:,1]))
            ground_x2 = np.max(c[:,:,1])
            
            line = np.array(base_mask[int(ground_y)][int(ground_x1):int(ground_x2)])
            line_match = np.array(line[:,0]==road_color[0])*np.array(line[:,1]==road_color[1])*np.array(line[:,2]==road_color[2])*np.array(line[:,3]==road_color[3])
            line_match = np.array(line_match,dtype=np.float32)
            
            if len(line_match) > 0 and np.mean(line_match) > 0.85:
                new_base = np.where(grid_3d==1,im,new_base)
                new_base_mask = np.where(grid_4d==1,mask,new_base_mask)
                total_features += 1
                
        if total_features > 0:
            if 'train' in folder:
    #             misc.imsave('base_pairs/train/'+str(base_id)+'_'+str(total_gen)+'.png',new_base)
                misc.imsave('modal_masks/train/'+str(base_id)+'_'+str(total_gen)+'.png',new_base_mask)
            else:
    #             misc.imsave('base_pairs/train/'+str(base_id)+'_'+str(total_gen)+'.png',new_base) 
                misc.imsave('modal_masks/val/'+str(base_id)+'_'+str(total_gen)+'.png',new_base_mask)
            total_gen += 1
        
        line = match_file.readline()
