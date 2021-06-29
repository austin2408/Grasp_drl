import os
import time
import copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import ndimage
from PIL import Image
from torchvision import transforms

from model import reinforcement_net

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

class Trainer():
    def __init__(self, args, run):

        self.args = args

        self.behavior_net = reinforcement_net(args.cuda)
        self.target_net   = reinforcement_net(args.cuda)

        # Huber Loss
        self.criterion = nn.SmoothL1Loss(reduce = False)

        if args.cuda:
            self.behavior_net = self.behavior_net.cuda()
            self.target_net   = self.target_net.cuda()
            self.criterion = self.criterion.cuda()
            print("using cuda")

        self.discount_factor = args.discount_factor

        # Set model to train mode
        self.behavior_net.train()
        self.target_net.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD([{'params': self.behavior_net.grasp_net.parameters(), 'lr': args.learning_rate},
                                          {'params': self.behavior_net.grasp_color_feat_extractor.parameters(), 'lr': args.densenet_lr}, 
                                          {'params': self.behavior_net.grasp_depth_feat_extractor.parameters(), 'lr': args.densenet_lr},
                                          ], lr = args.learning_rate, momentum = 0.9, weight_decay = 2e-5)

        # self.optimizer = torch.optim.Adam([{'params': self.behavior_net.grasp_net.parameters(), 'lr': args.learning_rate},
        #                                   {'params': self.behavior_net.grasp_color_feat_extractor.parameters(), 'lr': args.densenet_lr}, 
        #                                   {'params': self.behavior_net.grasp_depth_feat_extractor.parameters(), 'lr': args.densenet_lr},
        #                                   ], lr = args.learning_rate, weight_decay = 2e-5)

        # load model if need 
        if(args.load_model != None):
            artifact = run.use_artifact(args.load_model, type='model')
            artifact_dir = artifact.download()
            files = os.listdir(artifact_dir)
            for f in files:
                if "behavior" in f:
                    self.behavior_net.load_state_dict(torch.load(os.path.join(artifact_dir, f)))
                    self.target_net.load_state_dict(torch.load(self.behavior_net.state_dict()))

    def preprocessing(self, color, depth):
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
        image_mean = [0.805, 0.456, 0.406]
        image_std  = [0.229, 0.224, 0.225]
        input_color_img = color_img_2x.astype(float)/255 # np.uint8 to float
        for c in range(3):
            input_color_img[:, :, c] = (input_color_img[:, :, c] - image_mean[c]) / image_std[c]
        # Normalize depth image
        depth_mean = 0.0909769580291
        #depth_std  = 0.0005 # Terrible value...
        depth_std = 0.0398093901695
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

    # Forward pass through image, get Q value
    def forward(self, color_img, depth_img, is_volatile = False, specific_rotation=-1, network = "behavior", clear_grad = False):

        input_color_data, input_depth_data, padding_width = self.preprocessing(color_img, depth_img)
        # Pass input data to model
        if network == "behavior":
            output_prob = self.behavior_net.forward(input_color_data, input_depth_data, \
                                                    is_volatile=is_volatile, specific_rotation = specific_rotation, clear_grad=clear_grad)
        else: # Target
            output_prob = self.target_net.forward(input_color_data, input_depth_data, \
                                                  is_volatile=is_volatile, specific_rotation = specific_rotation, clear_grad=True)

        lower = int(padding_width/2)
        upper = int(input_color_data.shape[2]/2-padding_width/2)
        
        if is_volatile == False:
            return output_prob.cpu().detach().numpy()[:, 0, lower:upper, lower:upper] # only one array

        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                grasp_prediction = output_prob[rotate_idx].cpu().detach().numpy()[:, 0, lower:upper, lower:upper]
            else:
                grasp_prediction = np.concatenate((grasp_prediction, output_prob[rotate_idx].cpu().detach().numpy()[:, 0, lower:upper, lower:upper]))
        return grasp_prediction
        
    # Get TD target given reward and next state
    def get_label_value(self, reward, next_color, next_depth, is_empty):

        current_reward = reward
        # Compute TD target
        ''' 
        Double DQN 
        TD target = R + discount * Q_target(next state, argmax(Q_behavior(next state, a)))
        '''
        # Use behavior net to find best action in next state
        next_grasp_prediction = self.forward(next_color, next_depth, is_volatile = True)

        tmp = np.where(next_grasp_prediction==np.max(next_grasp_prediction))
        next_best_pixel = [tmp[0][0], tmp[1][0], tmp[2][0]]
        rotation = next_best_pixel[0]

        next_prediction = self.forward(next_color, next_depth, is_volatile = False, specific_rotation = rotation, network = "target")

        future_reward = 0.0
        if not is_empty:
            future_reward = next_prediction[0, next_best_pixel[1], next_best_pixel[2]]
        td_target = current_reward + self.discount_factor * future_reward

        del next_prediction
        return td_target

    # Do backwardpropagation
    def backprop(self, color_img, depth_img, action_pix_idx, label_value, is_weight, batch_size, first = False, update=False):

        label = np.zeros((1, 320, 320))
        label[0, action_pix_idx[1]+48, action_pix_idx[2]+48] = label_value # Extra padding
        label_weight = np.zeros((1, 320, 320))
        label_weight[0, action_pix_idx[1]+48, action_pix_idx[2]+48] = 1 # Extra padding
        if first: self.optimizer.zero_grad()
        loss_value = 0.0
        out_str = "({}, {}, {})| TD Target: {:.3f}\t Weight: {:.3f}\t".format(action_pix_idx[0], action_pix_idx[1], action_pix_idx[2], label_value, is_weight)
        # Forward pass to save gradient
        '''
            0 -> grasp, -90
            1 -> grasp, -45
            2 -> grasp, 0
            3 -> grasp, 45
        '''
        # grasp
        rotation = action_pix_idx[0]
        prediction = self.forward(color_img, depth_img, is_volatile = False, specific_rotation = rotation, network = "behavior", clear_grad = False)
        out_str += "Q: {:.3f}\t".format(prediction[0, action_pix_idx[1], action_pix_idx[2]])
        if self.args.cuda:
            loss = self.criterion(self.behavior_net.output_prob.view(1, 320, 320), Variable(torch.from_numpy(label).float().cuda()))* \
                                Variable(torch.from_numpy(label_weight).float().cuda(), requires_grad = False)* \
                                Variable(torch.from_numpy(np.array([is_weight])).float().cuda(), requires_grad = False)* \
                                Variable(torch.from_numpy(np.array([1./batch_size])).float().cuda(), requires_grad = False)
        else:
            loss = self.criterion(self.behavior_net.output_prob.view(1, 320, 320), Variable(torch.from_numpy(label).float()))* \
                                Variable(torch.from_numpy(label_weight).float(), requires_grad = False)* \
                                Variable(torch.from_numpy(np.array([is_weight])).float(), requires_grad = False)* \
                                Variable(torch.from_numpy(np.array([1./batch_size])).float(), requires_grad = False)
        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()
        # Grasping is symmetric
        rotation += 4
        prediction = self.forward(color_img, depth_img, is_volatile = False, specific_rotation = rotation, network = "behavior", clear_grad = False)
        out_str += "Q (symmetric): {:.3f}\t".format(prediction[0, action_pix_idx[1], action_pix_idx[2]])
        if self.args.cuda:
            loss = self.criterion(self.behavior_net.output_prob.view(1, 320, 320), Variable(torch.from_numpy(label).float().cuda()))* \
                                Variable(torch.from_numpy(label_weight).float().cuda(), requires_grad = False)* \
                                Variable(torch.from_numpy(np.array([is_weight])).float().cuda(), requires_grad = False)* \
                                Variable(torch.from_numpy(np.array([1./batch_size])).float().cuda(), requires_grad = False)
        else:
            loss = self.criterion(self.behavior_net.output_prob.view(1, 320, 320), Variable(torch.from_numpy(label).float()))* \
                                Variable(torch.from_numpy(label_weight).float(), requires_grad = False)* \
                                Variable(torch.from_numpy(np.array([is_weight])).float(), requires_grad = False)* \
                                Variable(torch.from_numpy(np.array([1./batch_size])).float(), requires_grad = False)
        loss = loss.sum()
        loss.backward()
        loss_value += loss.cpu().data.numpy()
        
        loss_value = loss_value/2
        
        out_str += "Training loss: {}".format(loss_value)
        if update: self.optimizer.step()
        return loss_value