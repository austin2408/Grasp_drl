import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms as TF
from collections import OrderedDict

def rotate_heightmap(color_tensor, depth_tensor, theta, use_cuda):
    # theta in radian
    affine_mat_before = np.asarray([[ np.cos(-theta), -np.sin(-theta), 0],
                                    [ np.sin(-theta),  np.cos(-theta), 0]])
    affine_mat_before.shape = (2, 3, 1)
    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
    if use_cuda:
        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), color_tensor.size())
        rotate_color_tensor = F.grid_sample(Variable(color_tensor, volatile=True).cuda(), flow_grid_before, mode="nearest")
        rotate_depth_tensor = F.grid_sample(Variable(depth_tensor, volatile=True).cuda(), flow_grid_before, mode="nearest")
    else:
        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), color_tensor.size())
        rotate_color_tensor = F.grid_sample(Variable(color_tensor, volatile=True), flow_grid_before, mode="nearest")
        rotate_depth_tensor = F.grid_sample(Variable(depth_tensor, volatile=True), flow_grid_before, mode="nearest")
    return rotate_color_tensor, rotate_depth_tensor
    
def rotate_featuremap(feature_tensor, theta, use_cuda):
    # theta in radian
    affine_mat_after = np.asarray([[ np.cos(-theta), -np.sin(-theta), 0],
                                   [ np.sin(-theta),  np.cos(-theta), 0]])
    affine_mat_after.shape = (2, 3, 1)
    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
    if use_cuda:
        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), feature_tensor.size())
        rotate_feature = F.grid_sample(feature_tensor, flow_grid_after, mode="nearest")
    else:
        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), feature_tensor.size())
        rotate_feature = F.grid_sample(feature_tensor, flow_grid_after, mode="nearest")
    return rotate_feature
    
class reinforcement_net(nn.Module):
    def __init__(self, use_cuda, num_rotations=4):
        super(reinforcement_net, self).__init__()
        self.use_cuda = use_cuda
        self.num_rotations = num_rotations
        
        # Initialize Densenet pretrained on ImageNet
        self.grasp_color_feat_extractor = torchvision.models.densenet121(pretrained = True)
        self.grasp_depth_feat_extractor = torchvision.models.densenet121(pretrained = True)
        # We don't need there fully connected layers
        del self.grasp_color_feat_extractor.classifier,  self.grasp_depth_feat_extractor.classifier
        
        # Gripper, corresponding to tool ID 1
        self.grasp_net = nn.Sequential(OrderedDict([
          ('grasp-norm0', nn.BatchNorm2d(2048)),
          ('grasp-relu0', nn.ReLU(inplace = True)),
          ('grasp-conv0', nn.Conv2d(2048, 64, kernel_size = 1, stride = 1, bias = False)),
          ('grasp-norm1', nn.BatchNorm2d(64)),
          ('grasp-relu1', nn.ReLU(inplace = True)),
          ('grasp-upsample0', nn.Upsample(scale_factor = 4, mode="bilinear")),
          ("grasp-conv1", nn.Conv2d(64, 1, kernel_size = 1, stride = 1, bias = False)),
          ('grasp-norm2', nn.BatchNorm2d(1)),
          ("grasp-upsample1", nn.Upsample(scale_factor = 4, mode="bilinear"))
        ]))

        self.grasp_net[6].weight.data.uniform_(-1e-5, 1e-5)

        self.output_prob = None
        
    def forward(self, input_color_data, input_depth_data, is_volatile = False, specific_rotation = -1, clear_grad = False):
        if is_volatile: # For choosing action
            output_prob = []
            if self.use_cuda:
                input_color_data = input_color_data.cuda()
                input_depth_data = input_depth_data.cuda()
           
            # Rotation
            for rotate_idx in range(self.num_rotations):
                theta = np.radians(-90.0+(180.0/self.num_rotations)*rotate_idx)
                rotate_color, rotate_depth = rotate_heightmap(input_color_data, input_depth_data, theta, self.use_cuda)
                with torch.no_grad():
                    interm_color_feat_rotated = self.grasp_color_feat_extractor.features(rotate_color)
                    interm_depth_feat_rotated = self.grasp_depth_feat_extractor.features(rotate_depth)
                    interm_feat_rotated = torch.cat((interm_color_feat_rotated, interm_depth_feat_rotated), dim=1)
                    output_prob.append(rotate_featuremap(self.grasp_net(interm_feat_rotated), -theta, self.use_cuda))
            return output_prob

        else: # For backpropagation, or computing TD target
            self.output_prob = None
            if self.use_cuda:
                input_color_data = input_color_data.cuda()
                input_depth_data = input_depth_data.cuda()
            
            rotate_idx = specific_rotation
            theta = np.radians(-90.0+(180.0/self.num_rotations)*rotate_idx)
            rotate_color, rotate_depth = rotate_heightmap(input_color_data, input_depth_data, theta, self.use_cuda)
            interm_color_feat_rotated = self.grasp_color_feat_extractor.features(rotate_color)
            interm_depth_feat_rotated = self.grasp_depth_feat_extractor.features(rotate_depth)
            if clear_grad:
                interm_color_feat_rotated.detach()
                interm_depth_feat_rotated.detach()
            interm_feat_rotated = torch.cat((interm_color_feat_rotated, interm_depth_feat_rotated), dim=1)
            self.output_prob = rotate_featuremap(self.grasp_net(interm_feat_rotated), -theta, self.use_cuda)

            return self.output_prob