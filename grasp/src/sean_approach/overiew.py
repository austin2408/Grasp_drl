from ast import iter_child_nodes
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import random
import json
import warnings  
warnings.filterwarnings("ignore") 
path = os.getcwd()

hdf5_path = path+'/datasets/Logger05_filter_v2.hdf5'
f = h5py.File(hdf5_path, "r")

angle = [0, 0, 0, 0]
for name in f.keys():
    group = f[name]
    action = group['action']
    angle[int(action.value[0])] += 1

print('90 -45 0 45')
print(angle)

