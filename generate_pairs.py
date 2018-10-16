from scipy import misc
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2

f = open("amodal/dict_file.txt","r")

for i in range(227):
    a = f.next()[:-1]
    c1 = a.index(",")
    c2 = a.index(",",c1+1)
    subfolder = a[c1+1:c2]
    im = a[c2+1:]
    mask_file = "gtFine_trainvaltest/gtFine" + subfolder + "/" + im + "_gtFine_color.png"
    img_file = "leftImg8bit_trainvaltest/leftImg8bit" + subfolder + "/" + im + "_leftImg8bit.png"
