import tensorflow as tf 
from utils import update_cfg



def value_model(features, labels, mode, params):

    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    y_hat = tf.layers.dense(net, 1, activation=None)

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
             'hidden_units':[100,100],
             'n_classes':3
            }

    def __init__(self, env, usr_cfg= None):
        self.env = env 
        self.config = update_cfg(VG_CONFIG, usr_cfg)
        self.obs_D = self.env.observation_space.shape[0]
        self.act_D = self.env.action_space.n 

        self._net_init()



    def _net_init(self):

    	self.classifier = tf.estimator.Estimator(
        model_fn=value_model,
        params={
            'feature_columns': self.config['feature_columns'],
            'hidden_units': self.config['hidden_units'],
            'n_classes': self.config['n_classes'],
        },
        model_dir = self.config['model_dir']
        )



    #steps are for the number of iteration(not epoch)

    def train(self, input_fn, steps=2000):
        self.classifier.train(input_fn, steps = steps)
        raise NotImplementedError


    def predict(self, path)




class Policy_net(object):
    '''
    In this network I choose not to use high-level API, otherwise you have to save 
    every checkpoint in the middle and training and updating it will require tons of I/O
    ops in the middle.(at least by now 07.16.18)
    '''

    #config for test
    PN_CONFIG = {'hidden_units':[100,100],
             'input_D':5,
             'n_classes':3
            }
    def __init__(self, env, usrcfg=None):
        self.env = env
        self.config = update_cfg(PN_CONFIG, usrcfg)
        self.obs_D = self.env.observation_space.shape[0]
        self.act_D = self.env.action_space.n

        self.input = tf.placeholder(tf.float32, shape = [None, self.config['input_D']])
        self._net_init()

    def _net_init(self):
        #only take care of discrete case in this version
        #self.out = self.input_ph
        net = self.input
        with tf.variable_scope('policy_net'):
            for num, units in enumerate(self.config['hidden_units']):
                net = tf.layers.dense(net, units=units, activation=tf.nn.relu,name = 'dense_'+str(num))
            logits = tf.layers.dense(net, units=self.config['input_D'], activation=tf.nn.relu, name='dense_'+str(num+1))
        
        predicted_classes = tf.argmax(logits, 1)

        #important property, will be used in the TRPO update
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'policy_net')

        #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=predict)
        #optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        #self.train_op = optimizer.minimize(loss)

    #def train(self, sess, input):
    #    for num_iter in self.config['num_iters']:
    #        sess.run(self.train_op, {self.input:input})

    def predict(self, sess, input):
        return 
        raise NotImplementedError













