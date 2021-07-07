from scipy import ndimage
import scipy.misc
import numpy as np
import torch
import os
import sys
import cv2
import copy
from math import *
import h5py
import random
import argparse
from matplotlib import pyplot as plt

from model import reinforcement_net
from utils import plot_figures, preprocessing

net = reinforcement_net(use_cuda=True)

model_name = "/home/austin/Grasp_drl/grasp/src/sean_approach/weight_try_2_sucess/behavior_160_0.056181950867176055.pth"
net.load_state_dict(torch.load(model_name))
net = net.cuda().eval()

hdf5_path = '/home/austin/DataSet/grasp_drl/logger0.hdf5'
f = h5py.File(hdf5_path, "r")
angle = [90, -45, 0, 45]
dis_count = 0
theta_count = 0
file = open('/home/austin/Grasp_drl/grasp/src/sean_approach/Test_record.txt', "a+")
for name in f.keys():
    group = f[name]
    action = group['action']
    color = f[name+'/state/color'].value
    depth = f[name+'/state/depth'].value

    color_tensor, depth_tensor, pad = preprocessing(color, depth)
    color_tensor = color_tensor.cuda()
    depth_tensor = depth_tensor.cuda()
    with torch.no_grad():
        prediction = net.forward(color_tensor, depth_tensor, is_volatile=True)

    size = color.shape[0]
    s = color_tensor.shape[2]
    lower = int(pad/2)
    upper = int(s/2-pad/2)
    
    tool_0 = prediction[0][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
    tool_1 = prediction[1][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
    tool_2 = prediction[2][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
    tool_3 = prediction[3][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy()

    result = plot_figures([tool_0, tool_1, tool_2, tool_3], color, depth)
    theta = angle[int(action.value[0])]
    print(name)
    print('Pred : ',result)
    print('Label : ',[theta, action.value[1], action.value[2]])
    dis_error = abs(sqrt((result[1] - action.value[1])**2 + (result[2] - action.value[2])**2))
    if (int(result[0]) !=  int(theta)):
        theta_count += 1
    print('Dis_error : ',dis_error)
    if dis_error > 20:
        dis_count += 1

    print('===================================================')
    file.write(name+'\n'+'Pred : '+str(result)+'\n'+'Label : '+str([theta, action.value[1], action.value[2]])+'\n'+'Dis_error : '+str(dis_error)+'\n')
    file.write('==================================================='+'\n')

print('Accuracy of position : ',(1 - dis_count/len(f.keys())))
print('Accuracy of theta : ',(1 - theta_count/len(f.keys())))
file.write(str((1 - dis_count/len(f.keys())))+'/'+str((1 - theta_count/len(f.keys()))))
file.close()