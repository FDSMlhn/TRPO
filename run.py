import tensorflow as tf
import argparse
import gym

import os.path as osp
import gym, logging
from baselines import logger
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.cmd_util import atari_arg_parser

import trpo
from atari_setter import *


def train(env_id, num_timesteps, seed = 1209 ):

    sess = tf.Session()

    workerseed = seed
    set_global_seeds(workerseed)

    env = make_atari(env_id)
	
	trpo_updater = trpo.TRPO_Updater(sess, env)

	env.render()


    #env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
    #env.seed(workerseed)

    #env = wrap_deepmind(env)
    #env.seed(workerseed)


    #trpo_mpi.learn(env, policy_fn, timesteps_per_batch=512, max_kl=0.001, cg_iters=10, cg_damping=1e-3,
    #    max_timesteps=int(num_timesteps * 1.1), gamma=0.98, lam=1.0, vf_iters=3, vf_stepsize=1e-4, entcoeff=0.00)
    env.close()

def main():
    args = atari_setter.atari_arg_parser()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
	main()
