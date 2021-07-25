#!/usr/bin/env python3
from scipy import ndimage
import scipy.misc
import numpy as np
import os
import sys
import cv2
import copy
import random
from sean_approach.model import reinforcement_net
from sean_approach.utils import plot_figures, preprocessing
import warnings
warnings.filterwarnings("ignore")

# ROS
import rospy
import roslib
import rospkg
import message_filters
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from medical_msgs.msg import *
from medical_msgs.srv import *
from scipy.spatial.transform import Rotation

# Torch
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets ,transforms
import torch.nn.functional as F

class DRL_Predict():
    def __init__(self):
        self.bridge = CvBridge()
        r = rospkg.RosPack()
        self.path = r.get_path("grasp")

        # DRL model
        self.net = reinforcement_net(use_cuda=True)
        self.net.load_state_dict(torch.load(self.path+'/src/sean_approach/model/behavior_half_v3_2_1000.pth'))
        self.net = self.net.cuda().eval()
        rospy.loginfo('Load model complete')

        # Publisher
        self.pose_pub = rospy.Publisher("DRL/hand_object_pose", HandObjectPose, queue_size=1)
        self.image_pub = rospy.Publisher("DRL/image", Image, queue_size=1)
        # self.predict_hero = rospy.Publisher("DRL/affordanceMap", Image, queue_size = 1)
        # self.affordance_pub = rospy.Publisher("DRL/affordance", Image, queue_size = 1)

        # Camera info
        info = rospy.wait_for_message('camera/color/camera_info', CameraInfo)
        self.fx = info.P[0]
        self.fy = info.P[5]
        self.cx = info.P[2]
        self.cy = info.P[6]

        # Subscriber
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        image_sub = message_filters.Subscriber('/camera/color/image_raw/compressed', CompressedImage)

        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 5, 5)
        ts.registerCallback(self.callback)

    def callback(self, Color, Depth):
        # Define msgs
        pose_msgs = HandObjectPose()

        # Ros image to cv2
        Color = self.bridge.compressed_imgmsg_to_cv2(Color, "bgr8")
        Depth = self.bridge.imgmsg_to_cv2(Depth, "16UC1")
        Color = cv2.resize(Color, (224,224))
        Depth = cv2.resize(Depth, (224,224))
        size = Color.shape[0]

        # Make prediction
        color_tensor, depth_tensor, pad = preprocessing(Color, Depth)
        color_tensor = color_tensor.cuda()
        depth_tensor = depth_tensor.cuda()
        with torch.no_grad():
            prediction = self.net.forward(color_tensor, depth_tensor, is_volatile=True)

        tool_0 = prediction[0][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy()
        tool_1 = prediction[1][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy()
        tool_2 = prediction[2][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy()
        tool_3 = prediction[3][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy()

        result = plot_figures([tool_0, tool_1, tool_2, tool_3], Color, Depth)
        
        # ================= Get Gripping Point ================== 
        x = result[2] 
        y = result[1] 
        z = Depth[y, x]/1000

        grasp_point_x, grasp_point_y, grasp_point_z = self.getXYZ(x * 2.857, y * 2.14285, z)
        pose_msgs.pose.position.x = grasp_point_x
        pose_msgs.pose.position.y = grasp_point_y
        pose_msgs.pose.position.z = grasp_point_z
        
        # ================= Get Gripping Angle ==================
        rot = Rotation.from_euler('xyz', [float(result[0]), 0, 0], degrees=True)
        rot_quat = rot.as_quat()
        pose_msgs.pose.orientation.x = rot_quat[0]
        pose_msgs.pose.orientation.y = rot_quat[1]
        pose_msgs.pose.orientation.z = rot_quat[2]
        pose_msgs.pose.orientation.w = rot_quat[3]

        # ================= Pub ==================
        if pose_msgs.pose.position.x != 0:
            cv2.circle(Color,(x, y), 5, (0, 0, 255), -1)
            self.pose_pub.publish(pose_msgs)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(Color, "bgr8"))
            print(pose_msgs)
            print("Angle: ", result[0])
            print("==========================")


    def getXYZ(self, x, y, zc):
        x = float(x)
        y = float(y)
        zc = float(zc)
        inv_fx = 1.0/self.fx
        inv_fy = 1.0/self.fy
        x = (x - self.cx) * zc * inv_fx
        y = (y - self.cy) * zc * inv_fy 
        z = zc 

        return z, -1*x, -1*y

    def onShutdown(self):
        rospy.loginfo("Shutdown.")


if __name__ == '__main__':
    rospy.init_node('DRL_Predict')
    foo = DRL_Predict()
    rospy.on_shutdown(foo.onShutdown)
    rospy.spin()






