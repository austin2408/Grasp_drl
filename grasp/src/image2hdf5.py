from ast import iter_child_nodes
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import random
import json

Path = '/home/austin/DataSet/grasp_drl/datasets'
ratio = 1
count = [0,0]

# Divide origin angle into 4 categories
# prediction: list with length 4
# | index | tool |
# | --- | --- |
# | 0 | gripper with -90 deg |
# | 1 | gripper with -45 deg |
# | 2 | gripper with 0 deg |
# | 3 | gripper with 45 deg |
def angle_class_4(theta_):
    angle = [0, 45, 90, 135]
    if (theta_ >angle[0] -22.5) and (theta_ < angle[0] + 22.5): # 0 +- 22.5
        return 2
    if (theta_ > angle[1] - 22.5) and (theta_ < angle[1] + 22.5): # 45 +- 22.5
        return 1
    if (theta_ > angle[2] - 22.5) and (theta_ < angle[2] + 22.5): # 90 +- 22.5
        return 0
    if (theta_ > angle[3] - 22.5) and (theta_ < angle[3] + 22.5): # 135 +- 22.5
        return 3
    if (theta_ > angle[3] + 22.5) and (theta_ < 180): # 180 - 180-22.5
        return 2
# Divide origin angle into 8 categories
# prediction: list with length 4
# | index | tool |
# | --- | --- |
# | 0 | gripper with -90 deg |
# | 1 | gripper with -45 deg |
# | 2 | gripper with 0 deg |
# | 3 | gripper with 45 deg |
# | 4 | gripper with -22.5 deg |
# | 5 | gripper with 22.5 deg |
# | 6 | gripper with -67.5 deg |
# | 7 | gripper with 67.5 deg |
def angle_class_8(theta_):
    angle = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    if (theta_ > angle[0] - 11.25) and (theta_ < angle[0] + 11.25): # 0 +- 11.25 ( real angle )
        return 2
    if (theta_ > angle[1] - 11.25) and (theta_ < angle[1] + 11.25): # 22.5 +- 11.25
        return 4
    if (theta_ > angle[2] - 11.25) and (theta_ < angle[2] + 11.25): # 45 +- 11.25
        return 1
    if (theta_ > angle[3] - 11.25) and (theta_ < angle[3] + 11.25): # 67.5 +- 11.25
        return 6
    if (theta_ > angle[4] - 11.25) and (theta_ < angle[4] + 11.25): # 90 +- 11.25
        return 0
    if (theta_ > angle[5] - 11.25) and (theta_ < angle[5] + 11.25): # 112.5 +- 11.25
        return 7
    if (theta_ > angle[6] - 11.25) and (theta_ < angle[6] + 11.25): # 135 +- 11.25
        return 3
    if (theta_ > angle[7] - 11.25) and (theta_ < angle[7] + 11.25): # 157.5 +- 11.25
        return 5
    if (theta_ >= angle[7] + 11.25) and (theta_ <= 180 ):
        return 2



def logger(path):
    name_list = os.listdir(path)
    with h5py.File('/home/austin/DataSet/grasp_drl/Logger05_8.hdf5','w') as f:
        for name in name_list:
            num = name.split('_')[1]

            # ------------------------------Success transition------------------------------ #
            g1=f.create_group("iter_"+num)

            # Get gripper action from json file
            with open(path+'/episode_'+num+'/rgb/rgb_'+num+'_0.json',"r") as F:
                data = json.load(F)
                coord = data['shapes'][0]['points']
                if data['shapes'][0]['label'] == 'good':
                    x = int((int(coord[0][0]) + int(coord[1][0]))/2)
                    y = int((int(coord[0][1]) + int(coord[1][1]))/2)
                    x_ = coord[0][0] - coord[1][0]
                    y_ = coord[0][1] - coord[1][1]
                    if y_ > 0:
                        x_ = -x_
                    y_ = abs(y_)

                    if y_ < 0.001:
                        theta = 0
                    else:
                        theta = math.atan2(y_, x_)
                        theta = int(math.degrees(theta))
            
            # Set action, reward, origin angle
            a_t = angle_class_8(theta)
            g1["reward"] = np.array([5])
            g1["origin_theta"] = np.array([theta])
            g1["action"] = np.array([a_t, int(y), int(x)])
            if (int(y)>224) or (int(x)>224):
                print('The image shape in '+name+' is wrong !')

            # Get state
            g2 = g1.create_group("state")
            color = cv2.imread(path+'/episode_'+num+'/rgb/rgb_'+num+'_0.jpg')
            color = color[:,:,[2,1,0]]
            color = cv2.resize(color, (224,224))
            g2.create_dataset('color', (224,224,3), data=color)
            depth = np.load(path+'/episode_'+num+'/depth/depth_'+num+'_0.npy')
            g2.create_dataset('depth', (224,224), data=depth)

            # Get next state
            g3 = g1.create_group("next_state")
            color2 = cv2.imread(path+'/episode_'+num+'/rgb/rgb_'+num+'_1.jpg')
            color2 = color2[:,:,[2,1,0]]
            color2 = cv2.resize(color2, (224,224))
            g3.create_dataset('color', (224,224,3), data=color2)
            depth2 = np.load(path+'/episode_'+num+'/depth/depth_'+num+'_1.npy')
            g3.create_dataset('depth', (224,224), data=depth2)
            g3["empty"] = np.array([True])
            count[0] += 1
            
            # ------------------------------Fail transition------------------------------ #
            Do = True if random.random() < ratio else False
            if Do:
                g1=f.create_group("iter_"+num+"_2")
                with open(path+'/episode_'+num+'/rgb/rgb_'+num+'_0.json',"r") as F:
                    data = json.load(F)
                    coord = data['shapes'][1]['points']
                    if data['shapes'][0]['label'] == 'bad':
                        x = int((int(coord[0][0]) + int(coord[1][0]))/2)
                        y = int((int(coord[0][1]) + int(coord[1][1]))/2)
                        x_ = coord[0][0] - coord[1][0]
                        y_ = coord[0][1] - coord[1][1]
                        if y_ > 0:
                            x_ = -x_
                        y_ = abs(y_)

                        if y_ < 0.001:
                            theta = 0
                        else:
                            theta = math.atan2(y_, x_)
                            theta = int(math.degrees(theta))
                
                a_t = angle_class_8(theta)
                g1["reward"] = np.array([-5])
                g1["origin_theta"] = np.array([theta])
                g1["action"] = np.array([a_t, int(y), int(x)])

                g2 = g1.create_group("state")
                color = cv2.imread(path+'/episode_'+num+'/rgb/rgb_'+num+'_0.jpg')
                color = color[:,:,[2,1,0]]
                color = cv2.resize(color, (224,224))
                g2.create_dataset('color', (224,224,3), data=color)
                depth = np.load(path+'/episode_'+num+'/depth/depth_'+num+'_0.npy')
                g2.create_dataset('depth', (224,224), data=depth)

                g3 = g1.create_group("next_state")
                g3.create_dataset('color', (224,224,3), data=color)
                g3.create_dataset('depth', (224,224), data=depth)
                g3["empty"] = np.array([False])
                count[1] += 1

        f.close()

logger(Path)
print('done')
f = h5py.File('/home/austin/DataSet/grasp_drl/Logger05_8.hdf5', "r")
print('Get ',len(f.keys()), ' transitions')
print('Success : ',count[0], ' Fail : ', count[1])
print('========================')
# Show structure
# print(f.keys())
group = f['iter_459']
for key in group.keys():
    print(key)
print('========================')
print(group['state'])
print(group['action'])
print(group['reward'])
print(group['next_state'])
print('========================')
for key in group['next_state']:
    print(key)

color = f['iter_459/state/color'].value
depth = f['iter_459/state/depth'].value
colorn = f['iter_459/next_state/color'].value
depthn = f['iter_459/next_state/depth'].value

print('========================')
print(group['next_state/empty'])
em = group['next_state/empty']
print(em.value)
print(color.shape)
print(depth.shape)
action = group['action']
reward = group['reward']
theta = group['origin_theta']
print(action.value)
print(reward.value)
print(theta.value)

_, axarr = plt.subplots(2,2) 
axarr[0][0].imshow(color)
axarr[0][1].imshow(depth)
axarr[1][0].imshow(colorn)
axarr[1][1].imshow(depthn)
plt.show()
