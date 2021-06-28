import sys
import os
from scipy import ndimage
import scipy.misc
import numpy as np
import torch
import cv2
import h5py
import gdown
from matplotlib import pyplot as plt
from zipfile import ZipFile
import time
from trainer import Trainer
from prioritized_memory import Memory
from collections import namedtuple

memory_capacity = [1064, 1064, 1064]
reward = 5.0
discount_factor = 0.99
mini_batch_size = 10
copy_target_net = 10
train_iter = 50
save_freq = 5
primitive_lr = 2.5e-4
dexnet_lr = 5e-5
save_root = "/home/austin/Test2"

Transition = namedtuple('Transition', ['color', 'depth', 'pixel_idx', 'reward', 'next_color', 'next_depth', 'is_empty'])

# suction_1_sampled = np.zeros(memory_capacity[0])
# suction_2_sampled = np.zeros(memory_capacity[1])
gripper_sampled = np.zeros(memory_capacity[2])
# suction_1_memory = Memory(memory_capacity[0])
# suction_2_memory = Memory(memory_capacity[1])
gripper_memory   = Memory(memory_capacity[2])

hdf5_path = '/home/austin/DataSet/grasp_drl/logger.hdf5'
f = h5py.File(hdf5_path, "r")

for key in f.keys():
  group = f[key]
  color = group['state/color'].value
  depth = group['state/depth'].value
  pixel_index = group['action'].value
  reward = group['reward'].value
  next_color = group['next_state/color'].value
  next_depth = group['next_state/depth'].value
  is_empty = group['next_state/empty'].value

  transition = Transition(color, depth, pixel_index, reward, next_color, next_depth, is_empty)
  gripper_memory.add(transition)
#   if pixel_index[0] == 0:
#     suction_1_memory.add(transition)
#   elif pixel_index[0] == 1:
#     suction_2_memory.add(transition)
#   else:
#     gripper_memory.add(transitio  n)
print ("Gripper_Buffer: {}".format(gripper_memory.length))
# print ("Suction_1_Buffer: {} | Suction_2_Buffer: {} | Gripper_Buffer: {}".format(suction_1_memory.length, suction_2_memory.length, gripper_memory.length))

trainer = Trainer(reward, discount_factor, False, primitive_lr, dexnet_lr)
trainer.target_net.load_state_dict(trainer.behavior_net.state_dict())

def sample_data(memory, batch_size):
	done = False
	mini_batch = []; idxs = []; is_weight = []
	while not done:
		success = True
		mini_batch, idxs, is_weight = memory.sample(batch_size)
		for transition in mini_batch:
			success = success and isinstance(transition, Transition)
		if success: done = True
	return mini_batch, idxs, is_weight

def get_action_info(pixel_index):
	if pixel_index[0] == 0:
		action_str = "suck_1"; rotate_idx = -1
	elif pixel_index[0] == 1:
		action_str = "suck_2"; rotate_idx = -1
	else:
		action_str = "grasp"; rotate_idx = pixel_index[0]-2
	return action_str, rotate_idx

# torch.save(trainer.behavior_net.state_dict(), save_root+"/model.pth")
def train():
  for i in range(train_iter):
    print('Epoch : ', i)
    ts = time.time()
    print ("[{}%]".format(i/float(train_iter)*100))
    mini_batch = []; idxs = []; is_weight = []; old_q = []
    # _mini_batch, _idxs, _is_weight = sample_data(suction_1_memory, mini_batch_size); mini_batch += _mini_batch; idxs += _idxs; is_weight += list(_is_weight); tmp = [idx-memory_capacity[0]-1 for idx in _idxs]; suction_1_sampled[tmp] += 1
    # _mini_batch, _idxs, _is_weight = sample_data(suction_2_memory, mini_batch_size); mini_batch += _mini_batch; idxs += _idxs; is_weight += list(_is_weight); tmp = [idx-memory_capacity[1]-1 for idx in _idxs]; suction_2_sampled[tmp] += 1
    _mini_batch, _idxs, _is_weight = sample_data(gripper_memory, mini_batch_size);   mini_batch += _mini_batch; idxs += _idxs; is_weight += list(_is_weight); tmp = [idx-memory_capacity[2]-1 for idx in _idxs]; gripper_sampled[tmp] += 1
    for j in range(len(mini_batch)):
      # print('1 : ',j)
      color = mini_batch[j].color
      depth =mini_batch[j].depth
      pixel_index = mini_batch[j].pixel_idx
      next_color = mini_batch[j].next_color
      next_depth = mini_batch[j].next_depth
      action_str, rotate_idx = get_action_info(pixel_index)
      # print(pixel_index[1], pixel_index[2])
      old_q.append(trainer.forward(color, depth, action_str, False, int(rotate_idx), clear_grad=True)[0, int(pixel_index[1]), int(pixel_index[2])])
      reward = mini_batch[j].reward
      td_target = trainer.get_label_value(reward, next_color, next_depth, mini_batch[j].is_empty, pixel_index[0])
#       print(pixel_index, '/',td_target[0], '/', is_weight[j], '/', mini_batch_size)
      loss_ = trainer.backprop(color, depth, pixel_index, int(td_target[0]), int(is_weight[j]), mini_batch_size, j==0, j==len(mini_batch)-1)
      file = open('/home/austin/Test2/rl_pnp/src/grasp_suck/src//Train_loss.txt', "a+")
      file.write(str(loss_)+'\n')
      print(loss_)
    # Update priority
    for j in range(len(mini_batch)):
      # print('2 : ',j)
      color = mini_batch[j].color
      depth = mini_batch[j].depth
      pixel_index = mini_batch[j].pixel_idx
      next_color = mini_batch[j].next_color
      next_depth = mini_batch[j].next_depth
      reward = mini_batch[j].reward
      td_target = trainer.get_label_value(reward, next_color, next_depth, mini_batch[j].is_empty, pixel_index[0])
      action_str, rotate_idx = get_action_info(pixel_index)
      new_value = trainer.forward(color, depth, action_str, False, rotate_idx, clear_grad=True)[0, pixel_index[1], pixel_index[2]]
      gripper_memory.update(idxs[j], td_target-new_value)
    #   if j/mini_batch_size==0: suction_1_memory.update(idxs[j], td_target-new_value)
    #   elif j/mini_batch_size==1: suction_2_memory.update(idxs[j], td_target-new_value)
    #   else: gripper_memory.update(idxs[j], td_target-new_value)
    if (i+1)==train_iter:
      print ("Save model")
      torch.save(trainer.behavior_net.state_dict(), save_root+"/model_{}.pth".format(i+1))
    if (i+1)%copy_target_net==0:
      trainer.target_net.load_state_dict(trainer.behavior_net.state_dict())
    print ("Took {} seconds".format(time.time()-ts))

train()