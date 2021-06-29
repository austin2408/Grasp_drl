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

parser = argparse.ArgumentParser(prog="Set up", description='This program for testing')
parser.add_argument("--idx", type=int, default=0, help="Give eposide index")
args = parser.parse_args()

num = str(args.idx)
color = cv2.imread('/home/austin/DataSet/grasp_drl/datasets/episode_'+num+'/rgb/rgb_'+num+'_0.jpg')
depth = np.load('/home/austin/DataSet/grasp_drl/datasets/episode_'+num+'/depth/depth_'+num+'_0.npy')
size = color.shape[0]

net = reinforcement_net(use_cuda=True)

model_name = "/home/austin/Grasp_drl/grasp/src/sean_approach/weight/behavior_500_0.0016054634004831314.pth"
net.load_state_dict(torch.load(model_name))
net = net.cuda().eval()

def preprocessing(color, depth):
	# Zoom 2 times
	color_img_2x = ndimage.zoom(color, zoom=[2, 2, 1], order=0)
	depth_img_2x = ndimage.zoom(depth, zoom=[2, 2],    order=0)
	# Add extra padding to handle rotations inside network
	diag_length = float(color_img_2x.shape[0])*np.sqrt(2)
	diag_length = np.ceil(diag_length/32)*32 # Shrink 32 times in network
	padding_width = int((diag_length - color_img_2x.shape[0])/2)
	# Convert BGR (cv) to RGB
	color_img_2x_b = np.pad(color_img_2x[:, :, 0], padding_width, 'constant', constant_values=0)
	color_img_2x_b.shape = (color_img_2x_b.shape[0], color_img_2x_b.shape[1], 1)
	color_img_2x_g = np.pad(color_img_2x[:, :, 1], padding_width, 'constant', constant_values=0)
	color_img_2x_g.shape = (color_img_2x_g.shape[0], color_img_2x_g.shape[1], 1)
	color_img_2x_r = np.pad(color_img_2x[:, :, 2], padding_width, 'constant', constant_values=0)
	color_img_2x_r.shape = (color_img_2x_r.shape[0], color_img_2x_r.shape[1], 1)
	color_img_2x = np.concatenate((color_img_2x_r, color_img_2x_g, color_img_2x_b), axis = 2)
	depth_img_2x = np.pad(depth_img_2x, padding_width, 'constant', constant_values=0)
	# Normalize color image with ImageNet data
	image_mean = [0.485, 0.456, 0.406] # for sim: [0.20414721, 0.17816422, 0.15419899]
	image_std  = [0.229, 0.224, 0.225] # for sim: [0.1830081 , 0.16705943, 0.17520182]
	input_color_img = color_img_2x.astype(float)/255 # np.uint8 to float
	for c in range(3):
		input_color_img[:, :, c] = (input_color_img[:, :, c] - image_mean[c]) / image_std[c]
	# Normalize depth image
	depth_mean = 0.0909769548291 # for sim: 0.032723393
	depth_std = 0.0397293901695 # for sim: 0.056900032
	tmp = depth_img_2x.astype(float)
	tmp = (tmp-depth_mean)/depth_std
	# Duplicate channel to DDD
	tmp.shape = (tmp.shape[0], tmp.shape[1], 1)
	input_depth_img = np.concatenate((tmp, tmp, tmp), axis = 2)
	# Convert to tensor
	# H, W, C - > N, C, H, W
	input_color_img.shape = (input_color_img.shape[0], input_color_img.shape[1], input_color_img.shape[2], 1)
	input_depth_img.shape = (input_depth_img.shape[0], input_depth_img.shape[1], input_depth_img.shape[2], 1)
	input_color_data = torch.from_numpy(input_color_img.astype(np.float32)).permute(3, 2, 0, 1)
	input_depth_data = torch.from_numpy(input_depth_img.astype(np.float32)).permute(3, 2, 0, 1)
	return input_color_data, input_depth_data, padding_width

# Preprocessing
color_tensor, depth_tensor, pad = preprocessing(color, depth)
color_tensor = color_tensor.cuda()
depth_tensor = depth_tensor.cuda()
prediction = net.forward(color_tensor, depth_tensor, is_volatile=True)
# print(type(prediction))
# print(type(prediction[2]))
# print(len(prediction))
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

def vis_affordance(predictions):
	tmp = np.copy(predictions)
	# View the value as probability
	tmp[tmp<0] = 0
	tmp /= 5
	tmp[tmp>1] = 1
	tmp = (tmp*255).astype(np.uint8)
	tmp.shape = (tmp.shape[0], tmp.shape[1], 1)
	heatmap = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)
	return heatmap

def plot_figures(tool):
    combine = []
    tool_cmap = []
    tt = []
    i = 0
    max_ = []
    pos = []
    theta_ = [90, -45, 0, 45]
    for object in tool:
        tool_cmap_ = vis_affordance(object)
        combine_ = cv2.addWeighted(color, 1.0, tool_cmap_, 0.8, 0.0)
        best = np.where(object == np.max(object))
        maxx = np.max(object)
        u, v = best[1][0], best[0][0]
        pos.append([u, v])
        combine_ = cv2.circle(combine_, (u, v), 3, (0, 0, 0), 2)
        tt_ = color.copy()
        cv2.circle(tt_,(u, v), 5, (255, 255, 0), -1)
        tool_cmap.append(tool_cmap_)
        combine.append(combine_)
        tt.append(tt_)
        print('angle : ', theta_[i], ' ',(u, v), ' max : ',maxx)
        max_.append(maxx)
        i += 1

    Max = max(max_)
    angle = theta_[max_.index(Max)]
    positions = pos[max_.index(Max)]
    f, axarr = plt.subplots(4,4) 
    plt.suptitle('Resurt : Angle : '+str(angle)+' Position : '+str(positions))
    axarr[0][0].set_title('90')
    axarr[0][0].imshow(combine[0][:,:,::-1])
    axarr[0][1].imshow(tool_cmap[0][:,:,[2,1,0]])
    axarr[0][2].imshow(tt[0][:,:,[2,1,0]])
    axarr[0][3].imshow(depth)

    axarr[1][0].set_title('-45')
    axarr[1][0].imshow(combine[1][:,:,::-1])
    axarr[1][1].imshow(tool_cmap[1][:,:,[2,1,0]])
    axarr[1][2].imshow(tt[1][:,:,[2,1,0]])
    axarr[1][3].imshow(depth)

    axarr[2][0].set_title('0')
    axarr[2][0].imshow(combine[2][:,:,::-1])
    axarr[2][1].imshow(tool_cmap[2][:,:,[2,1,0]])
    axarr[2][2].imshow(tt[2][:,:,[2,1,0]])
    axarr[2][3].imshow(depth)

    axarr[3][0].set_title('45')
    axarr[3][0].imshow(combine[3][:,:,::-1])
    axarr[3][1].imshow(tool_cmap[3][:,:,[2,1,0]])
    axarr[3][2].imshow(tt[3][:,:,[2,1,0]])
    axarr[3][3].imshow(depth)
    plt.savefig('/home/austin/Grasp_drl/grasp/src/sean_approach/result/image_'+num+'.png', dpi=300)
    plt.show()

tool_0 = prediction[0][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
tool_1 = prediction[1][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
tool_2 = prediction[2][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
tool_3 = prediction[3][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy()

plot_figures([tool_0, tool_1, tool_2, tool_3])