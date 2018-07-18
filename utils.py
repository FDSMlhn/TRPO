import tensorflow as tf
import numpy as np
import scipy.signal
import copy

EPS = 1e-7



def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.
    inputs
    ------
    x: ndarray
    gamma: float
    outputs
    -------
    y: ndarray with same shape as x, satisfying
        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1
    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]


ROLLOUT_CFG = {
    'max_pathlength': 10000,   #how maximal length of each trajectory will be.
    'timesteps_per_batch':1000 #how many trajectorys for each batch
}


def rollout(env, agent, max_pathlength, n_timesteps):
    paths = []
    timesteps_sofar = 0
    while timesteps_sofar < n_timesteps:
        obs, acts, rewards, act_dists = [], [], [], []
        ob = env.reset()

        for _ in range(max_pathlength):
            #第一次iter，出来的ob才是新的observation
            #dimension of ob is 1, reshape to 2-d to feed to our network 
            
            #print('obs dim is', ob.shape)
            act_dist, act = agent.act(ob.reshape(1,-1))
            #print("action is:", act)
            #print("action distribution is ", act_dist)

            obs.append(ob)
            acts.append(act)
            act_dists.append(act_dist)
            res = env.step(act)
            ob = res[0]
            rewards.append(res[1])
            if res[2]:
                        #"obs" 大概是 N, n_s, 2-d array
                path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
                        "act_dists": np.concatenate(act_dists),
                        "rewards": np.array(rewards),
                        "acts": np.array(acts)}

                paths.append(path)
                break

        timesteps_sofar += len(path["rewards"])
    return paths




def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary



def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def path2feat_vf(path):
    return path['obs']

def path2feat_pn(path):
    return path['obs']




def update_cfg(def_cfg, usr_cfg):
    #input: dict of default cfg and dict of usr cfg, but not change the original one
    if usr_cfg is None:
        return def_cfg

    out_cfg = copy.deepcopy(def_cfg)
    out_cfg.update(usr_cfg)
    #for key, value in def_cfg.items():
    #    out_cfg[key] = usr_cfg.get(key, value)

    return out_cfg

def var_shape(x):
    #calculate the shape of a tensor and return a list 
    return [k.value for k in x.get_shape()]


def num_ele(x):
    #calculate the number of elements in a tensor
    return np.prod(var_shape(x))

#compute grad and stretch it to 1-dimension tf.Tensor
def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(grad, [num_ele(v)]) for (v, grad) in zip(var_list, grads)], 0)

def slice_2d(x, inds0, inds1):

    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)



class GetFlat(object):
    '''
    notice everytime you concatenate your variable, you convert variable to tensor and 
    make it unable to be assigned to new value.
    Thus, it's necessary to have a convert a 
    '''
    def __init__(self, var_list):
        self.op = tf.concat([tf.reshape(v, [num_ele(v)]) for v in var_list], 0)

    def __call__(self,session):
        return self.op.eval(session=session)


class SetFromFlat(object):

    def __init__(self, var_list):
        assigns = []
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(tf.float64, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(
                    v,
                    tf.reshape(
                        theta[
                            start:start +
                            size],
                        shape)))
            start += size
        self.op = tf.group(*assigns)

        #change variable in the var_list to value of corresponding value in theta.    theta -> var_list
    def __call__(self, theta, session):
        session.run(self.op, feed_dict={self.theta: theta})






class ProbType(object):
    def sampled_variable(self):
        raise NotImplementedError
    def prob_variable(self):
        raise NotImplementedError
    def likelihood(self, a, prob):
        raise NotImplementedError
    def loglikelihood(self, a, prob):
        raise NotImplementedError
    def kl(self, prob0, prob1):
        raise NotImplementedError
    def entropy(self, prob):
        raise NotImplementedError
    def maxprob(self, prob):
        raise NotImplementedError


class Categorical(ProbType):
    def __init__(self, n):
        self.n = n
    def sampled_variable(self):
        return T.ivector('a')
    def prob_variable(self):
        return T.matrix('prob')
    def likelihood(self, a, prob):
        return prob[T.arange(prob.shape[0]), a]
    def loglikelihood(self, a, prob):
        return T.log(self.likelihood(a, prob))
    def kl(self, prob0, prob1):
        return (prob0 * T.log(prob0/prob1)).sum(axis=1)
    def entropy(self, prob0):
        return - (prob0 * T.log(prob0)).sum(axis=1)
    def sample(self, prob):
        return distributions.categorical_sample(prob)
    def maxprob(self, prob):
        return prob.argmax(axis=1)