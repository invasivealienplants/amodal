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
sky_color = [70,130,180,255]

def match_grid(mask,c):
    return np.array((mask[:,:,0]==c[0])*(mask[:,:,1]==c[1])*(mask[:,:,2]==c[2]),dtype=np.float32)

def get_bottom_pixels(binary,mask):
    r = []
    columns = np.sum(binary,0)
    for i,c in enumerate(columns):
        if c > 0:
            column = np.flip(binary[:,i],0)
            j = len(column)-np.argmax(column)-1
            r.append(mask[j][i])
    return np.array(r)
  
for i in range(271):
    print(i)
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
        subfolder,pth = line[:-1].split(",")
        mask_path = 'bdd100k_seg/bdd100k/seg/'+subfolder+'/'+pth+'_train_color.png'
        if 'train' in subfolder:
            im_path = 'bdd100k_seg/bdd100k/seg/images/train/'+pth+'.jpg'
        else:
            im_path = 'bdd100k_seg/bdd100k/seg/images/val/'+pth+'.jpg'
        
        mask = misc.imread(mask_path)
        im = misc.imread(im_path)

        pergrid = match_grid(mask,person_color)
        ridergrid = match_grid(mask,rider_color)
        bikegrid = match_grid(mask,bike_color)
        busgrid = match_grid(mask,bus_color)
        cargrid = match_grid(mask,car_color)
        carvgrid = match_grid(mask,caravan_color)
        motorgrid = match_grid(mask,motor_color)
        trailgrid = match_grid(mask,trail_color)
        traingrid = match_grid(mask,train_color)
        truckgrid = match_grid(mask,truck_color)
        roadgrid = match_grid(mask,road_color)
        skygrid = match_grid(mask,sky_color)
        foregrid = np.minimum(pergrid+ridergrid+bikegrid+busgrid+cargrid+carvgrid+motorgrid+trailgrid+traingrid+truckgrid,1.0)
        labeled_array,num_features = label(fore_grid,structure=np.ones([3,3]))
        
        skygrid = match_grid(mask,sky_color)
        basesky = match_grid(basemask,sky_color)
        skyscore = False
        if np.sum(skygrid) > 0 and np.sum(basesky) > 0:
            meansky = np.mean(im[skygrid==1],axis=(0,1))
            meanskybase = np.mean(baseim[basesky==1],axis=(0,1))
            skyscore = np.mean((meansky-meanskybase)**2)
            
        if np.mean(roadgrid) < 0.05 or skyscore > 10000 or skyscore == False:
            line = match_file.readline()
            continue
        
        total_features = 0
        new_base = np.copy(base_im)
        new_base_mask = np.copy(base_mask)
        width = len(mask)
        height = len(mask[0])
        coordinates = list(product(range(width), range(height)))
        coordinates = np.reshape(coordinates,[width,height,2])
        for i in range(1,num_features+1):
            grid = np.where(labeled_array==i,1.0,0.0)
            grid_3d = np.tile(np.reshape(grid,[720,1280,1]),(1,1,3))
            grid_4d = np.tile(np.reshape(grid,[720,1280,1]),(1,1,4))
            c = np.reshape(grid,[len(mask),len(mask[0]),1])*coordinates
            ground_y = np.max(c[:,:,0])
            ground_x1 = np.min(np.where(c[:,:,1]==0,np.inf,c[:,:,1]))
            ground_x2 = np.max(c[:,:,1])
            
            line = np.array(base_mask[int(ground_y)][int(ground_x1):int(ground_x2)])
            line_match = np.array(line[:,0]==road_color[0])*np.array(line[:,1]==road_color[1])*np.array(line[:,2]==road_color[2])*np.array(line[:,3]==road_color[3])
            line_match = np.array(line_match,dtype=np.float32)
            
            bottom = get_bottom_pixels(grid,base_mask)
            bottom_match = np.array(bottom[:,0]==road_color[0])*np.array(bottom[:,1]==road_color[1])*np.array(bottom[:,2]==road_color[2])*np.array(bottom[:,3]==road_color[3])
            bottom_match = np.array(bottom_match,dtype=np.float32)
            
            if len(line_match) > 0 and np.mean(bottom_match) > 0.75:
                new_base = np.where(grid_3d==1,im,new_base)
                new_base_mask = np.where(grid_4d==1,mask,new_base_mask)
                total_features += 1
                
        if total_features > 0:
            if 'train' in subfolder:
                misc.imsave('base_pairs_bdd/train/'+str(base_id)+'_'+str(total_gen)+'.png',new_base)
                misc.imsave('modal_masks_bdd/train/'+str(base_id)+'_'+str(total_gen)+'.png',new_base_mask)
            else:
                misc.imsave('base_pairs_bdd/val/'+str(base_id)+'_'+str(total_gen)+'.png',new_base)
                misc.imsave('modal_masks_bdd/val/'+str(base_id)+'_'+str(total_gen)+'.png',new_base_mask)
            total_gen += 1
        
        line = match_file.readline()
