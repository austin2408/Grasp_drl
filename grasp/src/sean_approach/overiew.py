from ast import iter_child_nodes
import h5py
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import random
import json
import warnings  
warnings.filterwarnings("ignore") 
path = os.getcwd()

hdf5_path = path+'/datasets/Logger_half_v3_2.hdf5'
f = h5py.File(hdf5_path, "r")

# print(len(f.keys()))
# angle = [0, 0, 0, 0]
# for name in f.keys():
#     group = f[name]
#     action = group['action']
#     angle[int(action.value[0])] += 1

# print('90 -45 0 45')
# print(angle)
def angle(idx):
    if idx == 0:
        return 90
    if idx == 1:
        return 45
    if idx == 2:
        return 0
    if idx == 3:
        return -45
id = 'iter_69'
group = f[id]
color = f[id+'/state/color'].value
depth = f[id+'/state/depth'].value
colorn = f[id+'/next_state/color'].value
depthn = f[id+'/next_state/depth'].value
action = group['action']
dis = 20
point1 = (int(action[2]+dis*math.cos(angle(action[0]))), int(action[1]-dis*math.sin(angle(action[0]))))

point2 = (int(action[2]-dis*math.cos(angle(action[0]))), int(action[1]+dis*math.sin(angle(action[0]))))

label = np.zeros((244,244,3))
cv2.line(label, point1, point2, (0, 255, 0), 3, 8)
cv2.circle(label, (action[2], action[1]), 4, (0,0,255), -1) # B

color_ = color.copy()
cv2.line(color_, point1, point2, (0, 255, 0), 3, 8)
cv2.circle(color_, (action[2], action[1]), 4, (0,0,255), -1) # B

_, axarr = plt.subplots(1,4) 
axarr[0].imshow(color)
axarr[1].imshow(label)
axarr[2].imshow(color_)
axarr[3].imshow(colorn)
cv2.imwrite('/home/austin/state_failed.jpg',color[:,:,[2,1,0]])
cv2.imwrite('/home/austin/action_failed.jpg',label[:,:,[2,1,0]])
cv2.imwrite('/home/austin/state+action_failed.jpg',color_[:,:,[2,1,0]])
cv2.imwrite('/home/austin/afterstate_failed.jpg',colorn[:,:,[2,1,0]])


# plt.imshow(color)
plt.show()
print(action.value)
print(point1)
print(point2)
