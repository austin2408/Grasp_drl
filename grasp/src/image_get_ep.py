#!/usr/bin/env python
import numpy as np
import rospy
import os
from cv_bridge import CvBridge
bridge = CvBridge()
from sensor_msgs.msg import Image
import cv2
import time
import message_filters
import argparse


class Collect(object):
    def __init__(self, ep, dir, name):
        self.save_path = dir +'/' + name
        os.mkdir(self.save_path)
        self.episode = ep
        self.catch = 0
        self.rgb = None
        self.depth = None
        self.aff = None
        self.img_rgb = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.img_depth = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        # self.img_aff = message_filters.Subscriber('',Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.img_rgb, self.img_depth], 5, 5)
        self.ts.registerCallback(self.register)
        self.save()

    def save(self):
        rospy.loginfo('Create new episode '+str(self.episode))
        path = self.save_path + '/episode_' + str(self.episode)
        Dir = [path, path+'/rgb', path+'/depth']
        for p in Dir:
            os.mkdir(p)
        i = 0
        while True:
            self.catch = input('')
            if self.catch == 1:
                self.episode += 1
                rospy.loginfo('Create next episode '+str(self.episode))
                path = self.save_path + '/episode_' + str(self.episode)
                Dir = [path,path+'/rgb',path+'/depth']
                for p in Dir:
                    os.mkdir(p)
                i = 0

            if self.catch == 2:
                # self.depth = self.depth/(np.max(self.depth)/255.0)
                # self.depth = np.round((self.depth/np.max(self.depth))*255).astype('float').reshape(1,self.depth.shape[0],self.depth.shape[1])
                # print(self.depth)
                self.rgb = cv2.resize(self.rgb, (224,224))
                self.depth = cv2.resize(self.depth, (224,224))
                np.save(path+'/depth'+'/depth_'+str(self.episode)+'_'+str(i), self.depth)
                cv2.imwrite(path+'/rgb'+'/rgb_'+str(self.episode)+'_'+str(i)+'.jpg', self.rgb[:,:,[2,1,0]])
                # cv2.imwrite(path+'/aff'+'/aff_'+str(self.episode)+'_'+str(i)+'.jpg', self.aff[:,:,[2,1,0]])
                rospy.loginfo('Save one state')
                i += 1

            self.catch == 0


    def register(self, RGB, DEPTH):
        self.rgb = bridge.imgmsg_to_cv2(RGB)
        self.depth = bridge.imgmsg_to_cv2(DEPTH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up')
    parser.add_argument('--ep', type=int, default = 0)
    parser.add_argument('--name', type=str, default = 'Datasets')
    parser.add_argument('--dir', type=str, default = '/home/austin/DataSet/grasp_drl')
    args = parser.parse_args()
    print(args)
    
    rospy.init_node("collect")
    print('Enter 1 to create new episode , Enter 2 to save current images')
    collecter = Collect(args.ep, args.dir, args.name)
    rospy.spin()