import tensorflow as tf 
import os
from utils import *



def value_model(features, labels, mode, params):

    #features: This is the x-arg from the input_fn.

    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    y_hat = tf.cast(tf.layers.dense(net, 1, activation=None), tf.float64)
    y_hat = tf.reshape(y_hat,[-1])

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predictions': y_hat,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.mean_squared_error(labels=labels, predictions=y_hat)

    # Compute evaluation metrics.
    mean_sq_error = tf.metrics.mean_squared_error(labels=labels,
                                   predictions=y_hat,
                                   name='error_op')
    metrics = {'mean_squared_error': mean_sq_error}
    tf.summary.scalar('mean_squared_error', mean_sq_error[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


class ValueF_net(object):
    #This class is used to predict the value function

    VF_CONFIG = {
             'hidden_units':[20,20],
             'feature_columns':None,
             'model_dir':'',
             'steps':2000,
             'batch_size':32
            }

    def __init__(self, sess, env, usr_cfg= None):

        self.sess = sess
        self.env = env 

        self.config = update_cfg(ValueF_net.VF_CONFIG, usr_cfg)
        self.obs_D = self.env.observation_space.shape[0]
        self.act_D = self.env.action_space.n 


        #don't forget to change later
        self.untrained= True

        self._net_init()



    def _net_init(self):

        #print('feature columns are', self.config['feature_columns'])
        self.classifier = tf.estimator.Estimator(
        model_fn=value_model,
        params={
            'feature_columns': self.config['feature_columns'],
            'hidden_units': self.config['hidden_units']
        },
        model_dir = self.config['model_dir']
        )



    #steps are for the number of iteration(not epoch)

    def train(self, paths):
        #convert from paths -> input_fn 
        print('We have {} number of paths for training Value network.'.format(len(paths)))

        feat_mat = np.concatenate([path2feat_vf(path) for path in paths])
        y = np.concatenate([path['returns'] for path in paths])

        #print(feat_mat.shape)
        #print(y.shape)
        input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": feat_mat},
                y=y,
                num_epochs=30,
                shuffle=True)

        self.classifier.train(input_fn)
        self.untrained=False


    def predict(self, path):
        #this function should return the value of each state in this path
        N = path['obs'].shape[0]
        feat = path2feat_vf(path)

        if self.untrained:
            return np.zeros(shape= (N,))
        else:
            input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": feat},
                    y=None,
                    num_epochs=1,
                    shuffle=False,
                    batch_size = self.config['batch_size'])
            gen = self.classifier.predict(input_fn= input_fn)
            out = []
            for n in range(N):
                out.append(gen.__next__()['predictions'])

            return np.array(out)




class Policy_net(object):
    '''
    In this network I choose not to use high-level API, otherwise you have to save 
    every checkpoint in the middle and training and updating it will require tons of I/O
    ops in the middle.(at least by now 07.16.18)
    '''

    #config for test: cartpole
    PN_CONFIG = {'hidden_units':[20,20],
             'input_D':4,
             'n_classes':2
            }
    def __init__(self, sess, env, usrcfg=None):
        self.sess= sess
        self.env = env
        self.config = update_cfg(Policy_net.PN_CONFIG, usrcfg)
        self.obs_D = self.env.observation_space.shape[0]
        self.act_D = self.env.action_space.n

        self.input = tf.placeholder(tf.float32, shape = [None, self.config['input_D']])
        self.N = tf.cast(tf.shape(self.input)[0], tf.float32)
        self._net_init()
        self._loss_init()

    def _net_init(self):
        #only take care of discrete case in this version
        #self.out = self.input_ph
        net = self.input
        with tf.variable_scope('policy_net'):
            for num, units in enumerate(self.config['hidden_units']):
                net = tf.layers.dense(net, units=units,
                    activation=tf.nn.relu, name = 'dense_'+str(num))
            logits = tf.layers.dense(net, units=self.act_D, 
                activation=tf.nn.relu, name='dense_'+str(num+1), )
        
        
        self.act_dist = tf.nn.softmax(logits)
        self.predicted_classes = tf.argmax(logits, 1)

        #important property, will be used in the TRPO update
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'policy_net')


    def _loss_init(self):
        '''
        Overall there are 5 placeholder in this function:
        1. old_act_dist
        2. action
        3. advant
        4. flat_gradient(objective gradient)
        '''
        self.old_act_dist = tf.placeholder(tf.float32, shape=[None, self.act_D], name="oldaction_dist")

        #notice its a one-dimensional object
        self.action = tf.placeholder(tf.int32, shape = [None], name='action')
        self.advant = tf.placeholder(tf.float32, shape = [None], name='action')

        p_n = slice_2d(self.act_dist, tf.range(0, self.N), self.action)
        oldp_n = slice_2d(self.old_act_dist, tf.range(0, self.N), self.action)
        
        ratio_n = p_n/oldp_n

        #we use loss here, and we may add a negative sign later
        surr = -tf.reduce_mean(ratio_n * self.advant)
        kl = tf.reduce_sum(self.old_act_dist * tf.log((self.old_act_dist + EPS) / (self.act_dist + EPS))) / self.N
        ent = tf.reduce_sum(-self.act_dist * tf.log(self.act_dist + EPS)) / self.N
        
        self.losses = [surr, kl, ent]

        self.pg = flatgrad(surr, self.var_list)
        self.feed = {}

        '''
        following lines are mainly for the update to the theta in the constrained optimization problem
        to derive the search direction and proper step size
        '''

        #this is for the derivative later on
        kl_firstfixed = tf.reduce_sum(tf.stop_gradient(
            self.act_dist) * tf.log(tf.stop_gradient(self.act_dist + EPS) / (self.act_dist + EPS))) / self.N
        
        
        #this will be the stretched gradient of the objective: g
        self.flat_gradient = tf.placeholder(tf.float32, shape=[None])

        shapes = map(var_shape, self.var_list)# list of list of shapes [[N, D], ...]
        start = 0
        gradient = []
        
        #reshape新的placeholder 到param的形状
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_gradient[start:(start + size)], shape)
            gradient.append(param)
            start += size


        #this is kind of shortcut to compute the search direction
        #grads is the derivative of KL divergence
        grads = tf.gradients(kl_firstfixed, self.var_list)
        kl_g_obj_g = tf.add_n([tf.reduce_sum(g * t) for (g, t) in zip(grads, gradient)])
        
        #flat gradient of Objective gradient and KL gradient multiplication
        self.flat_gradient_Obj_KL = flatgrad(kl_g_obj_g, self.var_list)
        self.gf = GetFlat(self.var_list)

        #will be used to assign value to the variable later on
        self.sff = SetFromFlat(self.var_list)

    def predict(self, input):
        return 
        raise NotImplementedError


    def compute_flat_gradient(self):
        return self.sess.run(self.pg, feed_dict=self.feed) 

    def loss(self, theta):
        '''
        compute value of objective for specific theta setting
        '''
        self.sff(theta, self.sess)
        return self.sess.run(self.losses[0], feed_dict=self.feed)


    def all_losses(self):
        return self.sess.run(self.losses, self.feed)


    def act(self, obs):
        act_dist, predicted_classes = self.sess.run([self.act_dist, self.predicted_classes],
        feed_dict= {self.input: obs})
        return act_dist, predicted_classes.squeeze()














