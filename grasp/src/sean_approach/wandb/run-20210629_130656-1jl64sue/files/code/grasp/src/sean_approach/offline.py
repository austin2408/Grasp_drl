import os
import numpy as np
import cv2
import torch
import argparse
import h5py
import wandb
from collections import namedtuple

from trainer import Trainer
from prioritized_memory import Memory

# Define transition tuple
Transition = namedtuple('Transition', ['color', 'depth', 'pixel_idx', 'reward', 'next_color', 'next_depth', 'is_empty'])

class Option():
    def __init__(self):
        parser = argparse.ArgumentParser(prog="DLP final project", description='This program for offline learning')

        # training hyper parameters
        parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Learning rate for the trainer, default is 2.5e-4")
        parser.add_argument("--densenet_lr", type=float, default=5e-5, help="Learning rate for the densenet block, default is 5e-5")
        parser.add_argument("--mini_batch_size", type=int, default=10, help="How many transitions should used for learning, default is 10") # K
        parser.add_argument("--save_freq", type=int, default=10, help="Every how many update should save the model, default is 5")
        parser.add_argument("--updating_freq", type=int, default=10, help="Frequency for updating target network, default is 6") # C
        parser.add_argument("--iteration", type=int, default=500, help="The train iteration, default is 30") # M
        parser.add_argument("--memory_size", type=int, default=None, help="The memory size, default is None")
        parser.add_argument("--discount_factor", type=float, default=0.9, help="The memory size, default is None")
        # parser.add_argument("gripper_memory", type=str, default=None, help="The pkl file for save experience")

        # save name and load model path
        parser.add_argument("--save_folder", type=str, default=os.getcwd(), help="save model in save folder, default is current path")
        parser.add_argument("--load_model",  type=str, default=None, help="load model from wandb, ex. 'kuolunwang/DLP_final_project/model:v0', default is None")

        # cuda
        parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training, default is False')
        
        self.parser = parser

    def create(self):

        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args 

Transition = namedtuple('Transition', ['color', 'depth', 'pixel_idx', 'reward', 'next_color', 'next_depth', 'is_empty'])
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
	action_str = "grasp"; rotate_idx = pixel_index[0]-2
	return action_str, rotate_idx

class Offline_training():
    def __init__(self, args):
        hdf5_path = '/home/austin/DataSet/grasp_drl/logger.hdf5'
        f = h5py.File(hdf5_path, "r")
        args.memory_size = len(f.keys())

        self.gripper_memory = Memory(args.memory_size)

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
            self.gripper_memory.add(transition)

        print ("Gripper_Buffer: {}".format(self.gripper_memory.length))

        run = wandb.init(project="Grasp_drl")
        config = wandb.config

        config.learning_rate = args.learning_rate
        config.iteration = args.iteration
        config.memory_size = args.memory_size
        config.updating_freq = args.updating_freq
        config.mini_batch_size = args.mini_batch_size
        config.densenet_lr = args.densenet_lr
        config.save_freq = args.save_freq
        config.discount_factor = args.discount_factor
        # config.gripper_memory = args.gripper_memory

        self.trainer = Trainer(args, run)

        #crate folder
        self.weight_path = os.path.join(args.save_folder,"weight")
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)

        self.training(args)

    def training(self, args):
        print('Start training ...')
        for i in range(args.iteration):
            mini_batch = []
            idxs = []
            is_weight = []
            old_q = []
            loss_list = []

            _mini_batch, _idxs, _is_weight = sample_data(self.gripper_memory, args.mini_batch_size);   mini_batch += _mini_batch; idxs += _idxs; is_weight += list(_is_weight)

            for j in range(len(mini_batch)):
                color = mini_batch[j].color
                depth =mini_batch[j].depth
                pixel_index = mini_batch[j].pixel_idx
                next_color = mini_batch[j].next_color
                next_depth = mini_batch[j].next_depth

                action_str, rotate_idx = get_action_info(pixel_index)

                old_q.append(self.trainer.forward(color, depth, action_str, False, int(rotate_idx), clear_grad=True)[0, int(pixel_index[1]), int(pixel_index[2])])
                reward = mini_batch[j].reward
                td_target = self.trainer.get_label_value(reward, next_color, next_depth, mini_batch[j].is_empty)
                loss_ = self.trainer.backprop(color, depth, pixel_index, int(td_target[0]), int(is_weight[j]), args.mini_batch_size, j==0, j==len(mini_batch)-1)
                loss_list.append(loss_)

            # Update priority
            for j in range(len(mini_batch)):
                color = mini_batch[j].color
                depth = mini_batch[j].depth
                pixel_index = mini_batch[j].pixel_idx
                next_color = mini_batch[j].next_color
                next_depth = mini_batch[j].next_depth
                reward = mini_batch[j].reward
                td_target = self.trainer.get_label_value(reward, next_color, next_depth, mini_batch[j].is_empty)
                action_str, rotate_idx = get_action_info(pixel_index)
                new_value = self.trainer.forward(color, depth, action_str, False, rotate_idx, clear_grad=True)[0, pixel_index[1], pixel_index[2]]
                self.gripper_memory.update(idxs[j], td_target-new_value)

            if (i+1) % args.save_freq == 0:

                torch.save(self.trainer.behavior_net.state_dict(), os.path.join(self.weight_path, "behavior_{}_{}.pth".format(i+1, sum(loss_list)/len(loss_list))))

            if (i+1) % args.updating_freq == 0:
                self.trainer.target_net.load_state_dict(self.trainer.behavior_net.state_dict())
            
            print('Epoch : ', i, ' | Loss : ', sum(loss_list)/len(loss_list))
            wandb.log({"loss mean": np.mean(loss_list)})

if __name__ == "__main__":

    args = Option().create()

    offline_learning = Offline_training(args)