from utils import *
from network import Policy_net, ValueF_net

import numpy as np
import random
import tensorflow as tf
import time
import os
import logging
import gym
from gym import envs, scoreboard
from gym.spaces import Discrete, Box
from space_conversion import SpaceConversionEnv
import tempfile
import sys




class TRPO_Updater():

	TRPO_CONFIG= {
		'gamma':0.99
	}
	def __init__(self, env, config):
		
		self.vf = network.ValueF_net(env)
		self.policy_net = network.Policy_net(env)

		self. = tf.placeholder(tf.float32)

		self.gf = GetFlat


	def learn(self, sess, paths):
		#everypath is a list of dict, where each contain rewards

		self.action_dist = self.policy_net.predict(sess, paths)

		for path in paths:
			feat_map = path2feat(path)
            path["baseline"] = self.vf.predict(path)            
            path["returns"] = discount(path["rewards"], config.gamma)
            path["advant"] = path["returns"] - path["baseline"]
