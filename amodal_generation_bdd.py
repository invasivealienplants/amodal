from scipy import misc
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2

f = open("base_bdd/dict_file_bdd.txt","w")

def match_grid(mask,c):
    return np.array((mask[:,:,0]==c[0])*(mask[:,:,1]==c[1])*(mask[:,:,2]==c[2])*(mask[:,:,3]==c[3]),dtype=np.float32)
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
        busgrid = match_grid(mask,bus_color)
        truckgrid = match_grid(mask,truck_color)
        trailgrid = match_grid(mask,trail_color)
        motorgrid = match_grid(mask,motor_color)
        licensegrid = match_grid(mask,license_color)
        fore_grid = np.minimum(cargrid+pergrid+bikegrid+pedgrid+petgrid+busgrid+truckgrid+trailgrid+motorgrid+licensegrid,1.0)
        percent_foreground.append(np.mean(fore_grid))
    return percent_foreground
  
def get_top_k(images,masks,image_ids,k=500):
    percent_foreground = calculate_foreground(images,masks)
    order = np.argsort(percent_foreground)[:k]
    i_ = []
    m_ = []
    id_ = []
    for idx in order:
        i_.append(images[idx])
        m_.append(masks[idx])
        id_.append(image_ids[idx])
    return i_,m_,id_

total_count = 0
base_images = []
base_masks = []
base_image_ids = []
for root,_,mask_paths in os.walk("bdd100k_seg/bdd100k/seg/color_labels/"):
    for mask_path in mask_paths:
        if mask_path[-9:] == "color.png":
            total_count += 1
            print(root,mask_path)
            
            subfolder = root[24:]
            image_id = mask_path[:-16]
            base_masks.append(misc.imread("bdd100k_seg/bdd100k/seg/color_labels/"+image_id+"_train_color.png"))
            base_images.append(misc.imread("bdd100k_seg/bdd100k/seg/images/"+image_id+".jpg"))
            image_id = subfolder+"/"+image_id
            base_image_ids.append(image_id)
            if len(base_images) % 1000 == 0:
                print("Images passed through : " + str(len(base_images)))
                base_images,base_masks,base_image_ids = get_top_k(base_images,base_masks,base_image_ids,k=800)
               
base_images,base_masks,base_image_ids = get_top_k(base_images,base_masks,base_image_ids,k=800)

# save images
for i,(base_image,base_mask,base_id) in enumerate(zip(base_images,base_masks,base_image_ids)):
    f.write(str(i)+","+base_id+"\n")
    misc.imsave("base_bdd/"+str(i)+"_image.png",base_image)
    misc.imsave("base_bdd/"+str(i)+"_mask.png",base_mask)
