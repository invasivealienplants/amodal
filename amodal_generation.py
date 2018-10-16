from scipy import misc
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2

def match_grid(mask,c):
    return np.array((mask[:,:,0]==c[0])*(mask[:,:,1]==c[1])*(mask[:,:,2]==c[2])*(mask[:,:,3]==c[3]),dtype=np.float32)

car_color = [0,0,142,255]
person_color = [255,0,0,255]
bike_color = [119,11,32,255]
ped_color = [220,20,60,255]
pet_color = [111,74,0,255]
truck_color = [0,60,100,255]
road_color = [128,64,128,255]

def calculate_foreground(images,masks):
    # foreground
    percent_foreground = []
    for i,(image,mask) in enumerate(zip(images,masks)):
        image = np.array(image)
        mask = np.array(mask)
        cargrid = match_grid(mask,car_color)
        pergrid = match_grid(mask,person_color)
        bikegrid = match_grid(mask,bike_color)
        pedgrid = match_grid(mask,ped_color)
        petgrid = match_grid(mask,pet_color)
        truckgrid = match_grid(mask,truck_color)
        roadgrid = match_grid(mask,road_color)
        fore_grid = np.minimum(cargrid+pergrid+bikegrid+pedgrid+petgrid+truckgrid-0*roadgrid,1.0)
        percent_foreground.append(np.mean(fore_grid))
    return percent_foreground

def get_top_k(images,masks,subfolders,image_ids,k=250):
    percent_foreground = calculate_foreground(images,masks)
    order = np.argsort(percent_foreground)[:k]
    i_ = []
    m_ = []
    s_ = []
    id_ = []
    for idx in order:
        i_.append(images[idx])
        m_.append(masks[idx])
        s_.append(subfolders[idx])
        id_.append(image_ids[idx])
    return i_,m_,s_,id_

total_count = 0
base_images = []
base_masks = []
base_subfolders = []
base_image_ids = []
for root,_,mask_paths in os.walk("gtFine"):
    for mask_path in mask_paths:
        if mask_path[-9:] == "color.png":
            total_count += 1
            subfolder = root[6:]
            image_id = mask_path[:-17]
            base_masks.append(misc.imread("gtFine/" + subfolder + "/" + image_id+"_gtFine_color.png"))
            base_images.append(misc.imread("leftImg8bit/" + subfolder + "/" + image_id + "_leftImg8bit.png"))
            base_subfolders.append(subfolder)
            base_image_ids.append(image_id)
            if len(base_images) % 1000 == 0:
                print("Images passed through : " + str(len(base_images)))
                base_images,base_masks,base_subfolders,base_image_ids = get_top_k(base_images,base_masks,base_subfolders,base_image_ids,k=25)
               
base_images,base_masks,base_subfolders,base_image_ids = get_top_k(base_images,base_masks,base_subfolders,base_image_ids,k=250)

# save images
f = open("dict_file.txt","a")
for i,(base_image,base_mask,base_sub,base_id) in enumerate(zip(base_images,base_masks,base_subfolders,base_image_ids)):
    f.write(str(i)+","+base_sub+","+base_id+"\n")
    misc.imsave(str(i)+"_image.png",base_image)
    misc.imsave(str(i)+"_mask.png",base_mask)
