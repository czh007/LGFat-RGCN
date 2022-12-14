# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

from modelUtils import ActionSelection, VanillaConv, PathEncoder
import torch.optim as optim
import numpy as np
import math
from discriminator import *
from torch.distributions import Bernoulli
import discriminator
from scipy.stats import entropy
from torch.optim import lr_scheduler
from options import args
from models import pick_model
from MultiCNN_utils import load_lookups


class Generator(nn.Module):
    def __init__(self, args, data, g):
        super(Generator, self).__init__()
        self.args = args
        self.data = data
        self.g = g
        # self.cnn =VanillaConv(args,vocab_size=data.size())
        self.cnn = pick_model(args, load_lookups(args))
        # self.cnn=EncoderRNN(args, num_layers=1, vocab_size=data.size())
        self.pathEncoder = PathEncoder(args, self.g)
        self.pathHist = []  # Saves the paths that have been selected (only the last ICD is kept)

        # Modules related to reinforcement learning
        self.ActionSelection = ActionSelection(args)
        self.gamma = 0.99  # Calculating discount rates for future awards
        self.lamda = 1.0

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=0.0001)
        self.schedule = lr_scheduler.StepLR(self.optimizer, 10, 0.1)

    def forward(self, d_model, ehrs, hier_labels):
        paths_hops = []
        ehrs_hops = []
        ground_truths_hops = []
        # Randomly selected batch paths
        ehrs, randomPaths = random_sample(hier_labels, ehrs)
        true_label_level_0 = [row[1] for row in randomPaths]
        true_label_level_1 = [row[2] for row in randomPaths]
        true_label_level_2 = [row[3] for row in randomPaths]
        true_label_level_3 = [row[4] for row in randomPaths]

        # Individual sampling for each sample in batch (each EHR)
        # 1. Get a representation of the electronic medical record
        ehrs = torch.Tensor(ehrs).long().to(self.args.device)
        ehrRrep = self.cnn(ehrs)  # [64,300]
        paths = [[self.args.node2id.get('ROOT')] for i in range(len(ehrs))]
        pathRep = self.pathEncoder(paths)  # [64,600]
        children, children_len = action_space(paths, self.args)  # [64,2023] [64]
        rewards_episode = [[] for i in range(len(ehrs))]
        log_action_episode = torch.zeros((len(ehrs), self.args.hops)).float().to(self.args.device)
        batchRewards = 0.0

        actions, log_action = self.ActionSelection.act(self.pathEncoder, ehrRrep, pathRep, children,
                                                       hop=0)  # [64,229],[64]
        # Execute the selected action to get the status and reward value
        paths = [row + [action] for row, action in zip(paths, actions)]
        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())
        # Get reward
        rewards_groundTruth = self.getReward(actions, true_label_level_0)
        ground_truths_hops.append(rewards_groundTruth)
        rewards = d_model.batchClassify(self, ehrs, paths).detach().cpu().numpy().tolist()  # [64,1]
        batchRewards = batchRewards + sum([sum(r) for r in rewards])
        rewards_episode = [row + r for row, r in zip(rewards_episode, rewards)]
        log_action_episode[:, 0] = log_action

        # hop==1
        # Select action down according to the new state
        pathRep = self.pathEncoder(paths)  # [64,600]
        children, children_len = action_space(actions, self.args)  # [64,2023] [64]
        actions, log_action = self.ActionSelection.act(self.pathEncoder, ehrRrep, pathRep, children,
                                                       hop=1)  # [64,229],[64,1]
        # Execute the selected action to get the status and reward value
        paths = [row + [action] for row, action in zip(paths, actions)]
        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())
        # Get reward
        rewards_groundTruth = self.getReward(actions, true_label_level_1)
        ground_truths_hops.append(rewards_groundTruth)
        rewards = d_model.batchClassify(self, ehrs, paths).detach().cpu().numpy().tolist()
        batchRewards = batchRewards + sum([sum(r) for r in rewards])
        rewards_episode = [row + r for row, r in zip(rewards_episode, rewards)]
        log_action_episode[:, 1] = log_action

        # hop==2
        # Select action down according to the new state
        pathRep = self.pathEncoder(paths)  # [64,100]
        children, children_len = action_space(actions, self.args)  # [64,2023] [64]
        actions, log_action = self.ActionSelection.act(self.pathEncoder, ehrRrep, pathRep, children,
                                                       hop=2)  # [64,229],[64,1]
        # Execute the selected action to get the status and reward value
        paths = [row + [action] for row, action in zip(paths, actions)]
        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())
        # Get reward
        rewards_groundTruth = self.getReward(actions, true_label_level_2)
        ground_truths_hops.append(rewards_groundTruth)
        rewards = d_model.batchClassify(self, ehrs, paths).detach().cpu().numpy().tolist()
        # rewards = rewards_groundTruth
        batchRewards = batchRewards + sum([sum(r) for r in rewards])
        rewards_episode = [row + r for row, r in zip(rewards_episode, rewards)]
        log_action_episode[:, 2] = log_action

        # hop==3
        # Select action down according to the new state
        pathRep = self.pathEncoder(paths)  # [64,100]
        children, children_len = action_space(actions, self.args)  # [64,2023] [64]
        actions, log_action = self.ActionSelection.act(self.pathEncoder, ehrRrep, pathRep, children,
                                                       hop=3)  # [64,229],[64,1]
        # print('hop:3,actions:', actions)
        # print('hop:3,state:', state)
        # Execute the selected action to get the status and reward value
        paths = [row + [action] for row, action in zip(paths, actions)]
        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())

        # Get reward
        rewards_groundTruth = self.getReward(actions, true_label_level_3)
        ground_truths_hops.append(rewards_groundTruth)
        rewards = d_model.batchClassify(self, ehrs, paths).detach().cpu().numpy().tolist()
        # rewards = rewards_groundTruth
        batchRewards = batchRewards + sum([sum(r) for r in rewards])
        rewards_episode = [row + r for row, r in zip(rewards_episode, rewards)]
        log_action_episode[:, 3] = log_action

        # The resulting paths are the paths predicted by the agent for each sample
        return batchRewards, rewards_episode, log_action_episode, ehrs_hops, paths_hops, ground_truths_hops

    def getReward(self, actions, hier_labels):
        # Calculate whether to give bonus value by level
        rewards = [[1] if p == h else [0] for p, h in zip(actions, hier_labels)]
        return rewards

    def update_policy(self, d_model, rewards_episode, log_action, ehrs_hops, paths_n):
        # batch_rewards=[]
        # # For each sample in this batch
        # for i in range(len(rewards_episode)):
        #     R = 0
        #     rewards = []
        #     # Discount future rewards back to the present using gamma
        #     for r in rewards_episode[i][::-1]:
        #         R = r + self.gamma * R
        #         rewards.insert(0, R)
        #     batch_rewards.append(rewards)

        # Scale rewards
        batch_rewards = torch.FloatTensor(rewards_episode).float().to(self.args.device)  # [64,4]

        # Calculate loss
        pg_loss = torch.mean(torch.mul(log_action, batch_rewards))
        self.optimizer.zero_grad()
        pg_loss.backward()

        self.optimizer.step()
        return pg_loss.item()

    # Pass in the real action to get the real (state,path) pair
    def teacher_force(self, d_model, ehrs, hier_labels):
        paths_hops = []
        ehrs_hops = []
        ground_truths_hops = []
        # Randomly selected batch paths
        ehrs, randomPaths = random_sample(hier_labels, ehrs)
        true_label_level_0 = [row[1] for row in randomPaths]
        true_label_level_1 = [row[2] for row in randomPaths]
        true_label_level_2 = [row[3] for row in randomPaths]
        true_label_level_3 = [row[4] for row in randomPaths]

        ehrs = torch.Tensor(ehrs).long().to(self.args.device)
        ehrRrep = self.cnn(ehrs)  # [64,300]
        # print('ehrRrep:',ehrRrep)
        paths = [[self.args.node2id.get('ROOT')] for i in range(len(ehrs))]
        pathRep = self.pathEncoder(paths)  # [64,600]
        children, children_len = action_space(paths, self.args)  # [64,2023] [64]
        rewards_episode = [[] for i in range(len(ehrs))]
        log_action_episode = torch.zeros((len(ehrs), self.args.hops)).float().to(self.args.device)
        batchRewards = 0.0

        # hop == 0
        actions, log_action = self.ActionSelection.act(self.pathEncoder, ehrRrep, pathRep, children,
                                                       hop=0)  # [64,229],[64]

        # Execute the selected action to get the status and reward value
        paths = [row + [action] for row, action in zip(paths, true_label_level_0)]

        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())
        # Get reward
        rewards_groundTruth = self.getReward(actions, true_label_level_0)
        ground_truths_hops.append(rewards_groundTruth)
        rewards = d_model.batchClassify(self, ehrs, paths).detach().cpu().numpy().tolist()  # [64,1]
        batchRewards = batchRewards + sum([sum(r) for r in rewards])
        rewards_episode = [row + r for row, r in zip(rewards_episode, rewards)]
        log_action_episode[:, 0] = log_action

        # hop==1
        # Select action down according to the new state
        pathRep = self.pathEncoder(paths)  # [64,600]
        children, children_len = action_space(true_label_level_0, self.args)  # [64,2023] [64]
        actions, log_action = self.ActionSelection.act(self.pathEncoder, ehrRrep, pathRep, children,
                                                       hop=1)  # [64,229],[64,1]
        # print('hop:1,actions:', actions)
        # print('hop:1,state:', state)
        # Execute the selected action to get the status and reward value
        paths = [row + [action] for row, action in zip(paths, true_label_level_1)]
        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())
        # Get reward
        rewards_groundTruth = self.getReward(actions, true_label_level_1)
        ground_truths_hops.append(rewards_groundTruth)
        rewards = d_model.batchClassify(self, ehrs, paths).detach().cpu().numpy().tolist()
        # rewards = rewards_groundTruth
        batchRewards = batchRewards + sum([sum(r) for r in rewards])
        rewards_episode = [row + r for row, r in zip(rewards_episode, rewards)]
        log_action_episode[:, 1] = log_action

        # hop==2
        # Select action down according to the new state
        pathRep = self.pathEncoder(paths)  # [64,100]
        children, children_len = action_space(true_label_level_1, self.args)  # [64,2023] [64]
        actions, log_action = self.ActionSelection.act(self.pathEncoder, ehrRrep, pathRep, children,
                                                       hop=2)  # [64,229],[64,1]
        # print('hop:2,actions:', actions)
        # print('hop:2,state:', state)
        # Execute the selected action to get the status and reward value
        paths = [row + [action] for row, action in zip(paths, true_label_level_2)]
        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())
        # Get reward
        rewards_groundTruth = self.getReward(actions, true_label_level_2)
        ground_truths_hops.append(rewards_groundTruth)
        rewards = d_model.batchClassify(self, ehrs, paths).detach().cpu().numpy().tolist()
        # rewards = rewards_groundTruth
        batchRewards = batchRewards + sum([sum(r) for r in rewards])
        rewards_episode = [row + r for row, r in zip(rewards_episode, rewards)]
        log_action_episode[:, 2] = log_action

        # hop==3
        # Select action down according to the new state
        pathRep = self.pathEncoder(paths)  # [64,100]
        children, children_len = action_space(true_label_level_2, self.args)  # [64,2023] [64]
        actions, log_action = self.ActionSelection.act(self.pathEncoder, ehrRrep, pathRep, children,
                                                       hop=3)  # [64,229],[64,1]
        # print('hop:3,actions:', actions)
        # print('hop:3,state:', state)
        # Execute the selected action to get the status and reward value
        paths = [row + [action] for row, action in zip(paths, true_label_level_3)]
        paths_hops.append(paths)
        ehrs_hops.append(ehrs.detach().cpu().numpy())

        # Get reward
        rewards_groundTruth = self.getReward(actions, true_label_level_3)
        ground_truths_hops.append(rewards_groundTruth)
        rewards = d_model.batchClassify(self, ehrs, paths).detach().cpu().numpy().tolist()
        # rewards = rewards_groundTruth
        batchRewards = batchRewards + sum([sum(r) for r in rewards])
        rewards_episode = [row + r for row, r in zip(rewards_episode, rewards)]
        log_action_episode[:, 3] = log_action

        return batchRewards, rewards_episode, log_action_episode, ehrs_hops, paths_hops, ground_truths_hops


def getHopAction(hier_labels, hop):
    selected_actions = [list(set([row[hop] for row in sample])) for sample in
                        hier_labels]  # selected_actions:[[**,**,**],[**,**,**,**],..]
    return selected_actions


def getHopActionNext(paths, hier_labels, hop, args):
    new_hier_labels = []
    for sample in hier_labels:
        new_hier_labels.append([row[:hop + 1] for row in sample])
    next_paths = []
    next_actions = []
    # print('new_hier_labels:',new_hier_labels)
    for i in range(len(paths)):  # Find the path that starts with each selected_actions (note: there may be more than one)
        # print('selected_actions:',selected_actions[i])
        selected_paths = [row for row in new_hier_labels[i] if row[:hop] == paths[i][1:]]
        selected_path = random.choice(selected_paths)
        selected_path = [args.node2id.get('ROOT')] + selected_path
        next_paths.append(selected_path)
        next_actions.append([selected_path[-1]])

    return next_paths, next_actions


def action_space(parients, args):
    childrens = torch.zeros((len(parients), len(args.node2id))).float().to(args.device)  # [64,229]
    children_lens = torch.zeros(len(parients)).long().to(args.device)  # [64]
    for sample_i in range(len(parients)):
        children = args.adj[parients[sample_i]]
        children_len = len(torch.nonzero(children))
        childrens[sample_i] = children
        children_lens[sample_i] = children_len
    return childrens, children_lens


def generate_samples(gen_model, d_model, ehrs, hier_labels):
    # Generate state,hidden(representation of path) from g_model
    batchRewards_n, rewards_episode_n, log_action_episode_n, ehrs_hops, paths_n, ground_truths_hops = gen_model(d_model,
                                                                                                                ehrs,
                                                                                                                hier_labels)
    batchRewards_p, rewards_episode_p, log_action_episode_p, ehrs_hops, paths_p, ground_truths_hops = gen_model.teacher_force(
        d_model, ehrs, hier_labels)

    return batchRewards_n, log_action_episode_n, log_action_episode_p, rewards_episode_n, rewards_episode_p, ehrs_hops, paths_n, ground_truths_hops, paths_p


def random_sample(paths, ehrs):
    selected_paths = []
    ehrs_ = []
    # Select a path from the paths of each sample
    for i in range(len(paths)):
        path = random.choice(paths[i])
        selected_paths.append(path)
        ehrs_.append(ehrs[i])
    return ehrs_, selected_paths
