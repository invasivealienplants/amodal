import os
from scipy import misc
import numpy as np

files = []
f = open("dict_file.txt","r")
for line in f:
    i = line[:-1].split(",")
    i[0] = int(i[0])
    files.append(i)
    
road_color = [128,64,128,255]
def match_grid(mask,c):
    return np.array((mask[:,:,0]==c[0])*(mask[:,:,1]==c[1])*(mask[:,:,2]==c[2]),dtype=np.float32)
  
train_images_f = open("train_images.txt","w")
train_labels_f = open("train_labels.txt","w")
val_images_f = open("val_images.txt","w")
val_labels_f = open("val_labels.txt","w")

c = 0
for root,_,paths in os.walk("syn_data/images"):
    for p in paths:
        print(c)
        c += 1
        full_path = root+"/"+p
        im = misc.imread(full_path)
        a = p.split("_")
        a = int(a[0])
        cityscapes_path = files[a]
        cityscapes_mask = misc.imread("cityscapes/gtFine/"+cityscapes_path[1]+"/"+cityscapes_path[2]+"_gtFine_color.png")
        cityscapes_im = misc.imread("cityscapes/leftImg8bit/"+cityscapes_path[1]+"/"+cityscapes_path[2]+"_leftImg8bit.png")
        
#         eq = np.array(cityscapes_im==im,dtype=np.float32)
        eq = np.array((cityscapes_im[:,:,0]==im[:,:,0])*(cityscapes_im[:,:,1]==im[:,:,1])*(cityscapes_im[:,:,2]==im[:,:,2]),dtype=np.float32)
        road = match_grid(cityscapes_mask,road_color)
#         binary_mask = np.where(eq[:,:,0]==0,road,255.0)
        binary_mask = road
        print(np.unique(binary_mask))
        
        cutout_im = im*np.reshape(eq,[1024,2048,1])
        
        if "train" in full_path:
            misc.imsave("syn_data/binary_labels/train/" + p,binary_mask)
            misc.imsave("syn_data/cutouts/train/" + p,cutout_im)
            train_labels_f.write("syn_data/binary_labels/train/"+p+"\n")
            train_images_f.write("syn_data/cutouts/train/"+p+"\n")
            misc.imsave("syn_data/paste_mask/train/"+p,eq)
        else:
            misc.imsave("syn_data/binary_labels/val/" + p,binary_mask)
            misc.imsave("syn_data/cutouts/val/" + p,cutout_im)
            val_labels_f.write("syn_data/binary_labels/val/"+p+"\n")
            val_images_f.write("syn_data/cutouts/val/"+p+"\n")   
            misc.imsave("syn_data/paste_mask/val/"+p,eq)
