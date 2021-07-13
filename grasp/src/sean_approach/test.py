from numpy.lib.npyio import save
from scipy import ndimage
import scipy.misc
import numpy as np
import torch
import os
import sys
import cv2
import copy
import h5py
import random
import argparse
from matplotlib import pyplot as plt

from model import reinforcement_net
from utils import plot_figures, preprocessing

import warnings  
warnings.filterwarnings("ignore") 

parser = argparse.ArgumentParser(prog="Set up", description='This program for testing')
parser.add_argument("--idx", type=int, default=0, help="Give eposide index")
args = parser.parse_args()

num = str(args.idx)

path = os.getcwd()
# color = cv2.imread('/home/austin/DataSet/grasp_drl/datasets/episode_'+num+'/rgb/rgb_'+num+'_0.jpg')
# color = cv2.resize(color,(224,224))
# depth = np.load('/home/austin/DataSet/grasp_drl/datasets/episode_'+num+'/depth/depth_'+num+'_0.npy')

color = cv2.imread('/home/austin/DataSet/grasp_drl/Datasets_test/episode_'+num+'/rgb/rgb_'+num+'_0.jpg')
color = cv2.resize(color,(224,224))
depth = np.load('/home/austin/DataSet/grasp_drl/Datasets_test/episode_'+num+'/depth/depth_'+num+'_0.npy')
# size = color.shape[0]

# depth[depth>1000] = 0

net = reinforcement_net(use_cuda=True)

model_name = path+'/model/behavior_half_v3_2_1000.pth'
net.load_state_dict(torch.load(model_name))
net = net.cuda().eval()

# Preprocessing
color_tensor, depth_tensor, pad = preprocessing(color, depth)
color_tensor = color_tensor.cuda()
depth_tensor = depth_tensor.cuda()
prediction = net.forward(color_tensor, depth_tensor, is_volatile=True)

size = color.shape[0]
s = color_tensor.shape[2]
lower = int(pad/2)
upper = int(s/2-pad/2)

# prediction: list with length 4
# | index | tool |
# | --- | --- |
# | 0 | gripper with -90 deg |
# | 1 | gripper with -45 deg |
# | 2 | gripper with 0 deg |
# | 3 | gripper with 45 deg |

tool_0 = prediction[0][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
tool_1 = prediction[1][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
tool_2 = prediction[2][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
tool_3 = prediction[3][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy()

plot_figures([tool_0, tool_1, tool_2, tool_3], color, depth, show=True, save=True)