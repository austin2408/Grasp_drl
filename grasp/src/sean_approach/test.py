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
from matplotlib import pyplot as plt

from model_v2 import reinforcement_net
# color_name = "/content/data/train_experience/logger_000/images/color_000000.jpg"
# depth_name = "/content/data/train_experience/logger_000/depth_data/depth_data_000000.npy"


color = cv2.imread('/home/austin/DataSet/grasp_drl/datasets/episode_100/rgb/rgb_100_0.jpg')
depth = np.load('/home/austin/DataSet/grasp_drl/datasets/episode_100/depth/depth_100_0.npy')
size = color.shape[0]

net = reinforcement_net(use_cuda=True)

model_name = "/home/austin/Test2/model_50.pth"
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
print(type(prediction))
print(type(prediction[2]))
print(prediction[2].shape)
size = color.shape[0]
s = color_tensor.shape[2]
lower = int(pad/2)
upper = int(s/2-pad/2)
# prediction: list with length 6
# | index | tool |
# | --- | --- |
# | 0 | small suction cup |
# | 1 | medium suction cup |
# | 2 | gripper with -90 deg |
# | 3 | gripper with -45 deg |
# | 4 | gripper with 0 deg |
# | 5 | gripper with 45 deg |

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

def plot_figures(tool, theta_):
    tool_cmap = vis_affordance(tool)
    combine = cv2.addWeighted(color, 1.0, tool_cmap, 0.8, 0.0)
    best = np.where(tool == np.max(tool))
    max = np.max(tool)
    u, v = best[1][0], best[0][0]
    combine = cv2.circle(combine, (u, v), 3, (255, 255, 255), 2)
    tt = color.copy()

    # plt.figure()
    cv2.circle(tt,(u, v), 5, (255, 0, 0), -1)
    f, axarr = plt.subplots(1,4) 
    axarr[0].imshow(combine[:,:,::-1])
    axarr[1].imshow(tool_cmap[:,:,[2,1,0]])
    axarr[2].imshow(tt[:,:,[2,1,0]])
    axarr[3].imshow(depth)
    # plt.savefig('/home/austin/DataSet/grasp_drl/0/image_'+str(theta_)+'.png', dpi=300)

    print('angle : ', theta_, ' ',(u, v), ' max : ',max)
    plt.show()
# gripper with -90 deg
tool_3 = prediction[2][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
plot_figures(tool_3, -90)
# gripper with -45 deg
tool_4 = prediction[3][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
plot_figures(tool_4, -45)
# gripper with 0 deg
tool_5 = prediction[4][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
plot_figures(tool_5, 0)
# gripper with 45 deg
tool_6 = prediction[5][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy()
plot_figures(tool_6, 45)