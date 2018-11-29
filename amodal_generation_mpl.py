from scipy import misc
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2

f = open("base_mpl/dict_file.txt","w")

def match_grid(mask,c):
    return np.array((mask[:,:,0]==c[0])*(mask[:,:,1]==c[1])*(mask[:,:,2]==c[2]),dtype=np.float32)
  
trail_color = [0,0,110]
ganimal_color = [0,192,0]
bus_color = [0,60,100]
bike_color = [255,0,0]
vehic_color = [128,64,64]
rider_color = [255,0,200]
motor_color = [255,0,100]
motorc_color = [0,0,230]
per_color = [220,20,60]
ego_color = [120,10,10]
car_color = [0,0,142]
bicy_color = [119,11,32]
carv_color = [0,0,90]
truck_color = [0,0,70]
whslow_color = [0,0,192]

blane_color = [128,64,255]
manhole_color = [100,128,160]
lanemark_color = [255,255,255]
crosswalk_color = [200,128,128]
cwalkp_color = [140,140,200]
road_color = [128,64,128]
servlane_color = [110,110,110]
phole_color = [70,100,150]

veg_color = [107,142,35]
sky_color = [70,130,180]
water_color = [0,170,30]
terrain_color = [152,251,152]

def calculate_foreground(images,masks):
    # foreground
    percent_foreground = []
    for i,(image,mask) in enumerate(zip(images,masks)):
        image = np.array(image)
        mask = np.array(mask)
        
        trailgrid = match_grid(mask,trail_color)
        ganimalgrid = match_grid(mask,ganimal_color)
        busgrid = match_grid(mask,bus_color)
        bikegrid = match_grid(mask,bike_color)
        vehicgrid = match_grid(mask,vehic_color)
        ridergrid = match_grid(mask,rider_color)
        motorgrid = match_grid(mask,motor_color)
        motorcgrid = match_grid(mask,motorc_color)
        pergrid = match_grid(mask,per_color)
        egogrid = match_grid(mask,ego_color)
        cargrid = match_grid(mask,car_color)
        bicygrid = match_grid(mask,bicy_color)
        carvgrid = match_grid(mask,carv_color)
        truckgrid = match_grid(mask,truck_color)
        whslowgrid = match_grid(mask,whslow_color)
        
        manholegrid = match_grid(mask,manhole_color)
        lanemarkgrid = match_grid(mask,lanemark_color)
        crosswalkgrid = match_grid(mask,crosswalk_color)
        cwalkpgrid = match_grid(mask,cwalkp_color)
        roadgrid = match_grid(mask,road_color)
        
        foregrid = np.minimum(1.0,trailgrid+ganimalgrid+busgrid+bikegrid+vehiclegrid+ridergrid+motorgrid+motorcgrid+pergrid+egogrid+cargrid+bicygrid+carvgrid+truckgrid+whslowgrid)
        roadgrid = np.minimum(1.0,manholegrid+lanemarkgrid+crosswalkgrid+cwalkpgrid+roadgrid)
        
        if np.mean(roadgrid) < 0.05:
            percent_foreground.append(np.inf)
        else:
            percent_foreground.append(np.mean(foreground))
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
for root,_,mask_paths in os.walk("data"):
    if not ("testing" in root) and "images" in root:
        for mask_path in mask_paths:
            total_count += 1
            subfolder = root.split("data/")[-1]
            image_id = mask_path.split(".jpg")[0]
            base_images.append(misc.imread(root+"/"+mask_path))
            if "training" in subfolder:
                base_masks.append(misc.imread("data/training/labels/"+image_id+".png"))
            else:
                base_masks.append(misc.imread("data/validation/labels/"+image_id+".png"))
            base_subfolders.append(subfolder)
            base_image_ids.append(image_id)
            if len(base_images) % 1500 == 0:
                print("Images passed through : " + str(total_count))
                base_images,base_masks,base_subfolders,base_image_ids = get_top_k(base_images,base_masks,base_subfolders,base_image_ids,k=750)
base_images,base_masks,base_subfolders,base_image_ids = get_top_k(base_images,base_masks,base_subfolders,base_image_ids,k=750)

# save images
for i,(base_image,base_mask,base_subfolder,base_id) in enumerate(zip(base_images,base_masks,base_subfolders,base_image_ids)):
    f.write(str(i)+","+base_subfolder+","+base_id+"\n")
