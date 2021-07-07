import os
import sys
import numpy as np
import yaml
import cv2
import struct
import ctypes
import rospkg
import argparse
import multiprocessing as mp
from scipy import ndimage
from collections import namedtuple
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from prioritized_memory import Memory


def standarization(prediction):
	if prediction.shape[0] is not 1:
		for i in range(prediction.shape[0]):
			mean = np.nanmean(prediction[i])
			std  = np.nanstd(prediction[i])
			prediction[i] = (prediction[i]-mean)/std
	else:
		mean = np.nanmean(prediction)
		std = np.nanstd(prediction)
		prediction = (prediction-mean)/std
	return prediction

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

def plot_figures(tool, color, depth, show=False, save=False):
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
        tool_cmap.append(tool_cmap_)
        combine.append(combine_)
        tt.append(tt_)
        # print('angle : ', theta_[i], ' ',(u, v), ' max : ',maxx)
        max_.append(maxx)
        i += 1

    Max = max(max_)
    angle = theta_[max_.index(Max)]
    positions = pos[max_.index(Max)]
    f, axarr = plt.subplots(2,5)
    plt.suptitle('Result : Angle : '+str(angle)+' Position : '+str(positions))
    axarr[0][0].set_title('color and depth')
    axarr[0][0].imshow(tt[0][:,:,[2,1,0]])
    axarr[1][0].imshow(depth)

    axarr[0][1].set_title('90')
    axarr[0][1].imshow(combine[0][:,:,::-1])
    axarr[1][1].imshow(tool_cmap[0][:,:,[2,1,0]])


    axarr[0][2].set_title('-45')
    axarr[0][2].imshow(combine[1][:,:,::-1])
    axarr[1][2].imshow(tool_cmap[1][:,:,[2,1,0]])

    axarr[0][3].set_title('0')
    axarr[0][3].imshow(combine[2][:,:,::-1])
    axarr[1][3].imshow(tool_cmap[2][:,:,[2,1,0]])

    axarr[0][4].set_title('45')
    axarr[0][4].imshow(combine[3][:,:,::-1])
    axarr[1][4].imshow(tool_cmap[3][:,:,[2,1,0]])
    if save:
        plt.savefig('/home/austin/Grasp_drl/grasp/src/sean_approach/result/sample.png', dpi=300)
    if show:
        plt.show()

    return [angle, positions[1], positions[0]]


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
	image_mean = [0.33638567, 0.33638567, 0.33638567]
	image_std  = [0.2603763,  0.2443466,  0.24258484]
	# image_mean = [0.485, 0.456, 0.406] # for sim: [0.20414721, 0.17816422, 0.15419899]
	# image_std  = [0.229, 0.224, 0.225] # for sim: [0.1830081 , 0.16705943, 0.17520182]
	input_color_img = color_img_2x.astype(float)/255 # np.uint8 to float
	for c in range(3):
		input_color_img[:, :, c] = (input_color_img[:, :, c] - image_mean[c]) / image_std[c]
	# Normalize depth image
	depth_mean = 1.3136337 # for sim: 0.032723393
	depth_std = 1.9633287 # for sim: 0.056900032
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