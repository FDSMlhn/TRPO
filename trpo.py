from utils import *
from network import Policy_net, ValueF_net
import time

import numpy as np
import random
import tensorflow as tf
import time, os



class TRPO_Updater():

    TRPO_CONFIG= {
        'gamma':0.99,
        "max_kl": 0.01,
        "cg_damping": 0.1,
        'cg_iters':10, 
        'residual_tol':1e-10}

    def __init__(self, sess, env, usr_cfg={}):
        self.env = env
        self.sess = sess
        self.config = update_cfg(TRPO_Updater.TRPO_CONFIG, usr_cfg)

        #print('my config', self.config)

        self.value_net = ValueF_net(self.sess, env, self.config)
        self.policy_net = Policy_net(self.sess, env, self.config)
        
        self.end_count = 0
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.train=True


    def learn(self, paths):
        #everypath is a list of dict, where each contain rewards

        start_time = time.time()

        for path in paths:

            #it's taken as the value fct
            path["baseline"] = self.value_net.predict(path)         

            path["returns"] = discount(path["rewards"], self.config['gamma'])
            path["advant"] = path["returns"] - path["baseline"]


        act_dist_n = np.concatenate([path["act_dists"] for path in paths])
        obs_n = np.concatenate([path["obs"] for path in paths])
        act_n = np.concatenate([path["acts"] for path in paths])
        baseline_n = np.concatenate([path["baseline"] for path in paths])
        returns_n = np.concatenate([path["returns"] for path in paths])

        #print(act_dist_n.shape)
        #print(baseline_n.shape)
        #print(returns_n.shape)
        #return

        # Standardize the advantage function to have mean=0 and std=1.
        advant_n = np.concatenate([path["advant"] for path in paths])
        advant_n -= advant_n.mean()

        # Computing baseline function for next iter.
        advant_n /= (advant_n.std() + 1e-8)

        self.policy_net.feed = {self.policy_net.input: obs_n,
                self.policy_net.action: act_n,
            self.policy_net.advant: advant_n,
                self.policy_net.old_act_dist: act_dist_n}


        episoderewards = np.array(
            [path["rewards"].sum() for path in paths])

        if not self.train:
            print("Episode mean: %f" % episoderewards.mean())            
            return


        if self.train:
            self.value_net.train(paths)
            
            #应该是只有policy network里的参数
            theta_prev = self.policy_net.gf(self.sess)

            g = self.policy_net.compute_flat_gradient()
            
            print(g)
            stepdir = self._conjugate_gradient(-g)

            #.5 is for computing beta in the paper
            shs = .5 * stepdir.dot(self._fisher_vector_product(stepdir))
            lm = np.sqrt(shs / self.config['max_kl'])
            fullstep = stepdir / lm
            neggdotstepdir = (-g).dot(stepdir)


            theta = self._line_search(theta_prev, fullstep, neggdotstepdir / lm)
            
            #assign new value to the theta
            self.policy_net.sff(theta, self.sess)

            surrafter, kloldnew, entropy = self.policy_net.all_losses()
        
        #ensure the we do satisfy KL constraint
            if kloldnew > 2.0 * self.config['max_kl']:
                self.policy_net.sff(theta_prev)

            stats = {}

            numeptotal = len(episoderewards)
            stats["Total number of episodes"] = numeptotal
            stats["Average sum of rewards per episode"] = episoderewards.mean()
            stats["Entropy"] = entropy
            exp = explained_variance(np.array(baseline_n), np.array(returns_n))
            stats["Baseline explained"] = exp
            stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
            stats["KL between old and new distribution"] = kloldnew
            stats["Surrogate loss"] = surrafter
            for k, v in stats.items():
                print(k + ": " + " " * (40 - len(k)) + str(v))
            if exp > 0.8:
                self.train = False
            	


    def _fisher_vector_product(self, p):
        self.policy_net.feed[self.policy_net.flat_gradient] = p
        return self.sess.run(self.policy_net.flat_gradient_Obj_KL, self.policy_net.feed) + self.config['cg_damping'] * p

    def _conjugate_gradient(self, g):
        '''
        The input g is the outcome of tf.gradient, so it's numpy array
        also notice it ensure the gradient is deriving the exact outcome we want in
        the paper
        
        The exact algorithm can be found here:
        https://web.stanford.edu/class/ee364b/lectures/conj_grad_slides.pdf

        '''
        p = g.copy()
        r = g.copy()
        x = np.zeros(g.shape)
        rho = r.dot(r)
        for i in range(self.config['cg_iters']):
            w = self._fisher_vector_product(g)
            print('w is ')
            print(w)
            print('p is ')
            print(p)
            alpha = rho / p.dot(w)
            x += alpha * p
            r -= alpha * w
            new_rho = r.dot(r)
            ratio = new_rho / rho
            p = r + ratio * p
            rho = new_rho
            if rho < self.config['residual_tol']:
                break
        return x

    def _line_search(self, theta_prev, fullstep, expected_improve_rate):
        '''
        Notice the expected_improve_rate is the fullstep dot product with gradient
        I don't manage to find exactly why we should do so, only a rough inituition
        '''
        accept_ratio = .1
        max_backtracks = 10
        #return the surrogate loss for new parameter x 
        fval = self.policy_net.loss(theta_prev)
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
            theta_new = theta_prev + stepfrac * fullstep
            newfval = self.policy_net.loss(theta_new)
            actual_improve = fval - newfval
        
            #stepfrac is 1, 0.5, 0.25, 
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            if ratio > accept_ratio and actual_improve > 0:
                return theta_new
        return theta_prev












